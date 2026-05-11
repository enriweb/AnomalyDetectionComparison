"""
RepViT-EfficientAD — EfficientAD student-teacher distillation with timm
RepViT-M1 (teacher, frozen ImageNet-pretrained) and RepViT-M0.9 (student).

Replaces canonical PDN-S teacher/student with thin RepViT-backed wrappers
that upsample stage-3 feature maps (stride 32) back to AE-compatible
(stride 4) spatial resolution. Autoencoder branch is left unchanged.

NOTE: This deviates from canonical EfficientAD in two ways:
    1. Teacher is NOT distilled from WideResNet-101 on ImageNet patches;
       it is the off-the-shelf timm-pretrained RepViT-M1 backbone with a
       1x1 projection layer.
    2. PDN-specific pretrained weights are skipped (no
       `pretrained_teacher_*.pth` load).

Usage:
    python repvit_efficientad.py          # 1 run (default)
    python repvit_efficientad.py 3        # 3 runs
"""

import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries")
import torch
import torch.nn as nn
import torch.nn.functional as F
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import (
    EfficientAdModel, EfficientAdModelSize, AutoEncoder, imagenet_norm_batch,
)
from lightning.pytorch.callbacks import TQDMProgressBar
import timm

try:
    from anomalib.metrics import F1Max
    HAS_F1MAX = True
except ImportError:
    HAS_F1MAX = False

try:
    from thop import profile as _thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

from utils.efficiency import measure_all

# ── Config ────────────────────────────────────────────────────
N_RUNS: int        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TEACHER_BACKBONE   = "repvit_m1"
STUDENT_BACKBONE   = "repvit_m0_9"
TEACHER_OUT_CH     = 384
IMAGE_SIZE         = (256, 256)
MAX_EPOCHS         = 20
LR                 = 1e-4
WEIGHT_DECAY       = 1e-5
MAX_VRAM_GB: float = 0
torch.set_float32_matmul_precision("high")


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _start_mem_sampler(samples: list, interval: float = 0.05) -> threading.Event:
    stop = threading.Event()
    def _run():
        while not stop.is_set():
            if torch.cuda.is_available():
                samples.append(torch.cuda.memory_allocated() / 1e6)
            stop.wait(interval)
    threading.Thread(target=_run, daemon=True).start()
    return stop


def vram_ok() -> bool:
    if not torch.cuda.is_available() or MAX_VRAM_GB <= 0:
        return True
    return torch.cuda.memory_reserved() / 1e9 < MAX_VRAM_GB


class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm(); bar.leave = False; return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm(); bar.leave = False; return bar
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm(); bar.leave = False; return bar


# ── RepViT-based teacher / student wrappers ───────────────────
class RepViTPDN(nn.Module):
    """Thin wrapper around a timm RepViT backbone.

    Extracts the last stage's feature map (stride 32) and bilinear-upsamples
    it to the EfficientAD-AE spatial resolution (input // 4 when padding=True).
    Optionally projects channels to ``out_channels`` via a 1x1 conv.
    """
    def __init__(
        self,
        model_name: str,
        out_channels: int,
        out_stride: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.out_stride = out_stride
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=(3,),
        )
        native_c = self.backbone.feature_info.channels()[0]
        self.proj = (
            nn.Conv2d(native_c, out_channels, kernel_size=1)
            if native_c != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = imagenet_norm_batch(x)
        feat = self.backbone(x)[0]
        feat = self.proj(feat)
        h, w = x.shape[-2], x.shape[-1]
        target = (h // self.out_stride, w // self.out_stride)
        return F.interpolate(feat, size=target, mode="bilinear", align_corners=False)


# ── RepViT-EfficientAD torch model ────────────────────────────
class RepViTEfficientAdModel(EfficientAdModel):
    """EfficientAdModel with RepViT teacher + student replacing PDN-S."""
    def __init__(
        self,
        teacher_out_channels: int = TEACHER_OUT_CH,
        padding: bool = True,
        pad_maps: bool = False,
        teacher_backbone: str = TEACHER_BACKBONE,
        student_backbone: str = STUDENT_BACKBONE,
    ) -> None:
        # Bypass parent init: we don't want PDN-S teacher/student built.
        nn.Module.__init__(self)
        self.pad_maps = pad_maps

        self.teacher = RepViTPDN(
            model_name=teacher_backbone,
            out_channels=teacher_out_channels,
            out_stride=4,
            pretrained=True,
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = RepViTPDN(
            model_name=student_backbone,
            out_channels=teacher_out_channels * 2,
            out_stride=4,
            pretrained=True,
        )

        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels: int = teacher_out_channels

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": nn.Parameter(torch.zeros((1, teacher_out_channels, 1, 1)), requires_grad=False),
                "std":  nn.Parameter(torch.zeros((1, teacher_out_channels, 1, 1)), requires_grad=False),
            },
        )
        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": nn.Parameter(torch.tensor(0.0), requires_grad=False),
                "qb_st": nn.Parameter(torch.tensor(0.0), requires_grad=False),
                "qa_ae": nn.Parameter(torch.tensor(0.0), requires_grad=False),
                "qb_ae": nn.Parameter(torch.tensor(0.0), requires_grad=False),
            },
        )


# ── RepViT-EfficientAD Lightning module ───────────────────────
class RepViTEfficientAd(EfficientAd):
    """EfficientAd Lightning module wired to RepViTEfficientAdModel.

    Overrides `prepare_pretrained_model` to a no-op — RepViT backbones come
    pre-loaded with timm-trained ImageNet weights, no PDN .pth file exists.
    """
    def __init__(
        self,
        imagenet_dir: str = "./datasets/imagenette",
        teacher_out_channels: int = TEACHER_OUT_CH,
        padding: bool = True,
        pad_maps: bool = False,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        **kw,
    ) -> None:
        super().__init__(
            imagenet_dir=imagenet_dir,
            teacher_out_channels=teacher_out_channels,
            model_size=EfficientAdModelSize.S,
            lr=lr,
            weight_decay=weight_decay,
            padding=padding,
            pad_maps=pad_maps,
            **kw,
        )
        self.model = RepViTEfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            padding=padding,
            pad_maps=pad_maps,
            teacher_backbone=TEACHER_BACKBONE,
            student_backbone=STUDENT_BACKBONE,
        )

    def prepare_pretrained_model(self) -> None:
        """Skip PDN teacher .pth load — RepViT timm weights already pretrained."""
        pass


# ── Dataset config ────────────────────────────────────────────
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

DATASETS = [
    ("mvtec", MVTecAD, MVTEC_CATEGORIES, "./datasets/MVTecAD"),
    ("visa",  Visa,    VISA_CATEGORIES,  "./datasets/VisA"),
]

total_categories = sum(len(c) for _, _, c, _ in DATASETS)

os.makedirs("results", exist_ok=True)
PROGRESS_FILE = "results/repvit_efficientad_progress.json"

print("=" * 60)
print(f"  RepViT-EfficientAD  |  T={TEACHER_BACKBONE}  |  S={STUDENT_BACKBONE}")
print(f"  teacher_out={TEACHER_OUT_CH}  padding=True  epochs={MAX_EPOCHS}")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
print(f"  FLOPs (thop)      : {'available' if HAS_THOP else 'not installed — skipped'}")
print(f"  F1Max metric      : {'available' if HAS_F1MAX else 'not in this anomalib version — skipped'}")
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device            : {gpu_info}")
print("=" * 60)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"\n[resume] Found checkpoint — resuming from {PROGRESS_FILE}")
else:
    results = {ds: {} for ds, *_ in DATASETS}
    print("\n[start] No checkpoint found — starting fresh.")

oom_skips: list[dict] = []

for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
    print(f"\n{'─'*60}")
    print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
    print(f"{'─'*60}")

    for cat_idx, category in enumerate(categories, 1):
        runs_done = len(results.get(ds_name, {}).get(category, []))
        if runs_done >= N_RUNS:
            print(f"  [skip] {category} ({cat_idx}/{len(categories)}) — {runs_done}/{N_RUNS} runs already done")
            continue

        if ds_name not in results:
            results[ds_name] = {}
        if category not in results[ds_name]:
            results[ds_name][category] = []

        print(f"\n  [{cat_idx}/{len(categories)}] {category}  — {N_RUNS - runs_done} run(s) remaining")

        for run in range(runs_done, N_RUNS):
            print(f"\n  {'='*48}")
            print(f"  {ds_name.upper()}  |  {category}  |  Run {run+1}/{N_RUNS}")
            print(f"  {'='*48}")

            if not vram_ok():
                free_gpu()
                if not vram_ok():
                    print(f"  [SKIP] VRAM over threshold. Skipping run {run+1}.")
                    continue

            model = engine = datamodule = evaluator = None
            image_auroc = pixel_auroc = pixel_pro = None
            image_f1max = pixel_f1max = test_results = metrics = None
            try:
                print(f"  → Loading datamodule (train_batch_size=1 per EfficientAD)...")
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=1,
                    eval_batch_size=16,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                pixel_pro   = AUPRO(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                test_metrics = [image_auroc, pixel_auroc, pixel_pro]
                if HAS_F1MAX:
                    image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                    pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                    test_metrics += [image_f1max, pixel_f1max]
                evaluator = Evaluator(test_metrics=test_metrics)

                print(f"  → Building model ({TEACHER_BACKBONE} + {STUDENT_BACKBONE})...")
                model = RepViTEfficientAd(
                    imagenet_dir="./datasets/imagenette",
                    teacher_out_channels=TEACHER_OUT_CH,
                    padding=True,
                    pad_maps=False,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    pre_processor=EfficientAd.configure_pre_processor(image_size=IMAGE_SIZE),
                    evaluator=evaluator,
                    visualizer=False,
                )

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  → Parameters: {n_params:,}")

                print(f"  → Training ({MAX_EPOCHS} epochs)...")
                engine = Engine(max_epochs=MAX_EPOCHS, devices="auto", strategy="auto",
                                logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)

                flops_M = None
                if HAS_THOP:
                    try:
                        device = next(model.parameters()).device
                        _dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _flops, _ = _thop_profile(
                                model.model.teacher, inputs=(_dummy,), verbose=False
                            )
                        flops_M = round(_flops / 1e6, 1)
                        print(f"  → Teacher FLOPs: {flops_M:.1f} M")
                        del _dummy
                    except Exception:
                        pass

                eff_extra = None
                try:
                    eff_extra = measure_all(
                        model.model.teacher,
                        input_shape=(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        skip_flops=True,
                        gpu_batch=8,
                    )
                    print(f"  → GPU throughput: {eff_extra.get('gpu_throughput'):.1f} img/s"
                          f"  |  CPU latency: {eff_extra.get('cpu_latency_ms'):.1f} ms")
                except Exception as e:
                    print(f"  [warn] measure_all failed: {e}")

                print(f"  → Running inference on test set (GPU)...")
                _gpu_samples: list = []
                _gpu_stop = _start_mem_sampler(_gpu_samples)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                test_results = engine.test(model=model, datamodule=datamodule)
                elapsed_gpu = time.time() - t0
                _gpu_stop.set()

                peak_gpu_mb = 0.0
                mean_gpu_mb = 0.0
                if torch.cuda.is_available():
                    peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if _gpu_samples:
                    mean_gpu_mb = round(statistics.mean(_gpu_samples), 1)

                metrics = test_results[0]
                img_auc  = metrics.get("image_AUROC", 0) * 100
                pxl_auc  = metrics.get("pixel_AUROC", 0) * 100
                pxl_pro  = metrics.get("pixel_AUPRO", 0) * 100
                img_f1   = (metrics["image_F1Max"] * 100) if metrics.get("image_F1Max") is not None else None
                pxl_f1   = (metrics["pixel_F1Max"] * 100) if metrics.get("pixel_F1Max") is not None else None

                print(f"  → Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                      f" | AUPRO {pxl_pro:.1f}%"
                      + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else ""))
                print(f"  → GPU inf: {elapsed_gpu:.3f}s  |  Peak GPU: {peak_gpu_mb:.1f} MB"
                      f"  |  Mean GPU: {mean_gpu_mb:.1f} MB")

                run_record = {
                    "image_AUROC":     img_auc,
                    "pixel_AUROC":     pxl_auc,
                    "pixel_AUPRO":     pxl_pro,
                    "image_F1Max":     img_f1,
                    "pixel_F1Max":     pxl_f1,
                    "n_params":        n_params,
                    "flops_M":         flops_M,
                    "inference_gpu_s": elapsed_gpu,
                    "peak_gpu_mb":     peak_gpu_mb,
                    "mean_gpu_mb":     mean_gpu_mb,
                }
                if eff_extra:
                    run_record["gpu_throughput"] = eff_extra.get("gpu_throughput")
                    run_record["cpu_latency_ms"] = eff_extra.get("cpu_latency_ms")
                    run_record["peak_gpu_mem_b"] = eff_extra.get("peak_gpu_mem")

                results[ds_name][category].append(run_record)

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → Checkpoint saved.")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_skips.append({"ds": ds_name, "category": category, "run": run + 1})
                    print(f"\n  [OOM] CUDA OOM — {ds_name}/{category} run {run+1}. Skipping.")
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max
                del test_results, metrics
                free_gpu()

print(f"\n{'='*60}")
print("  All runs complete.")

if oom_skips:
    print(f"\n  [OOM summary] {len(oom_skips)} run(s) skipped:")
    for s in oom_skips:
        print(f"    • {s['ds']}/{s['category']}  run {s['run']}")
else:
    print("  No OOM skips.")

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"\n  Checkpoint deleted ({PROGRESS_FILE}).")

print(f"{'='*60}\n")


def _mean(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else float("nan")


def _std(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _fmt(mean: float, std: float, decimals: int = 1) -> str:
    if mean != mean:
        return "n/a".rjust(12 if N_RUNS > 1 else 10)
    fmt = f"{{:{5+decimals}.{decimals}f}}±{{:4.{decimals}f}}" if N_RUNS > 1 else f"{{:10.{decimals}f}}"
    return fmt.format(mean, std) if N_RUNS > 1 else fmt.format(mean)


out_path = "results/repvit_efficientad_combined.txt"
C  = 12 if N_RUNS > 1 else 10
CE = 11

lines = []
SEP = "=" * 80
lines.append(SEP)
lines.append(f"  RepViT-EfficientAD Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  Teacher: {TEACHER_BACKBONE}  |  Student: {STUDENT_BACKBONE}")
lines.append(SEP)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"

    lines.append(f"\n--- {label} — Accuracy ---")
    HDR_A = (f"{'Category':<15} {'Img AUROC':>{C}} {'Pxl AUROC':>{C}}"
             f" {'AUPRO':>{C}} {'Img F1Max':>{C}} {'Pxl F1Max':>{C}}")
    lines.append(HDR_A)
    lines.append("-" * len(HDR_A))

    cat_acc: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        r = {k: (_mean(runs, k), _std(runs, k)) for k in
             ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max")}
        cat_acc[cat] = {k: v[0] for k, v in r.items()}
        lines.append(
            f"{cat:<15}"
            f" {_fmt(*r['image_AUROC']):>{C}}"
            f" {_fmt(*r['pixel_AUROC']):>{C}}"
            f" {_fmt(*r['pixel_AUPRO']):>{C}}"
            f" {_fmt(*r['image_F1Max']):>{C}}"
            f" {_fmt(*r['pixel_F1Max']):>{C}}"
        )
    if cat_acc:
        avg = {k: statistics.mean(v for v in (c[k] for c in cat_acc.values()) if v == v)
               for k in ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max")}
        lines.append("-" * len(HDR_A))
        lines.append(
            f"{'MEAN':<15}"
            f" {avg['image_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUPRO']:>{C}.1f}"
            f" {avg.get('image_F1Max', float('nan')):>{C}.1f}"
            f" {avg.get('pixel_F1Max', float('nan')):>{C}.1f}"
        )

    lines.append(f"\n--- {label} — Efficiency ---")
    HDR_E = (f"{'Category':<15} {'Params':>{CE}} {'FLOPs(M)':>{CE}}"
             f" {'GPU inf(s)':>{CE}} {'GPU th(i/s)':>{CE}}"
             f" {'CPU(ms)':>{CE}} {'PkGPU(MB)':>{CE}}")
    lines.append(HDR_E)
    lines.append("-" * len(HDR_E))

    def _fe(v, fmt) -> str:
        try:
            return (fmt.format(v) if v is not None and v == v else "n/a").rjust(CE)
        except Exception:
            return "n/a".rjust(CE)

    cat_eff: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        np_  = _mean(runs, "n_params")
        fl_  = _mean(runs, "flops_M")
        gi_  = _mean(runs, "inference_gpu_s")
        gt_  = _mean(runs, "gpu_throughput")
        cl_  = _mean(runs, "cpu_latency_ms")
        pgm_ = _mean(runs, "peak_gpu_mb")
        cat_eff[cat] = {"np": np_, "fl": fl_, "gpu": gi_, "th": gt_, "cpu": cl_, "pgm": pgm_}
        lines.append(
            f"{cat:<15}"
            f" {_fe(np_,  '{:,.0f}')}"
            f" {_fe(fl_,  '{:.1f}')}"
            f" {_fe(gi_,  '{:.3f}')}"
            f" {_fe(gt_,  '{:.1f}')}"
            f" {_fe(cl_,  '{:.1f}')}"
            f" {_fe(pgm_, '{:.1f}')}"
        )
    if cat_eff:
        lines.append("-" * len(HDR_E))
        def _avg_eff(key):
            vals = [v[key] for v in cat_eff.values() if v[key] is not None and v[key] == v[key]]
            return statistics.mean(vals) if vals else float("nan")
        _first = next(iter(cat_eff.values()))
        lines.append(
            f"{'MEAN':<15}"
            f" {_fe(_first['np'],  '{:,.0f}')}"
            f" {_fe(_first['fl'],  '{:.1f}')}"
            f" {_fe(_avg_eff('gpu'), '{:.3f}')}"
            f" {_fe(_avg_eff('th'),  '{:.1f}')}"
            f" {_fe(_avg_eff('cpu'), '{:.1f}')}"
            f" {_fe(_avg_eff('pgm'), '{:.1f}')}"
        )

lines.append("=" * 80)
output = "\n".join(lines)
print(output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
