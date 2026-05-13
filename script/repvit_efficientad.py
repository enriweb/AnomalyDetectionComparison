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

import copy
import gc
import json
import logging
import os
import sys
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
from anomalib.metrics import F1Max
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.flop_counter import FlopCounterMode
import timm
import pandas as pd

# ── Config ────────────────────────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1
torch.set_float32_matmul_precision("high")


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Inline efficiency helpers ─────────────────────────────────
@torch.no_grad()
def _gpu_throughput(model,
                    input_shape: tuple = (1, 3, 224, 224),
                    batch: int = 8,
                    warmup: int = 10,
                    iters: int = 50):
    """Images per second on CUDA. Returns None if no GPU."""
    if not torch.cuda.is_available():
        return None
    _model = model.cuda().eval()
    _, c, h, w = input_shape
    x = torch.randn(batch, c, h, w, device="cuda")
    for _ in range(warmup):
        _model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _model(x)
    torch.cuda.synchronize()
    return (batch * iters) / (time.perf_counter() - t0)


@torch.no_grad()
def _cpu_latency(model,
                 input_shape: tuple = (1, 3, 224, 224),
                 warmup: int = 5,
                 iters: int = 20):
    """Mean ms per single-image forward on CPU."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*input_shape)
    try:
        for _ in range(warmup):
            cpu_model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_model(x)
        return (time.perf_counter() - t0) / iters * 1000.0
    finally:
        torch.set_num_threads(prev_threads)
        del cpu_model


@torch.no_grad()
def _peak_gpu_mem(model,
                  input_shape: tuple = (1, 3, 224, 224)):
    """Peak GPU allocated bytes for a single forward pass. None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    _model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(*input_shape, device="cuda")
    _model(x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


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
        teacher_out_channels: int = 384,
        padding: bool = True,
        pad_maps: bool = False,
        teacher_backbone: str = "repvit_m1",
        student_backbone: str = "repvit_m0_9",
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
        teacher_out_channels: int = 384,
        padding: bool = True,
        pad_maps: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
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
            teacher_backbone="repvit_m1",
            student_backbone="repvit_m0_9",
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

# ── Startup banner ────────────────────────────────────────────
print("=" * 60)
print(f"  RepViT-EfficientAD  (N={N_RUNS}, Teacher: repvit_m1, Student: repvit_m0_9)")
print("  teacher_out=384  padding=True  epochs=20")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
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

# ── OOM skip tracking ─────────────────────────────────────────
oom_skips: list[dict] = []   # {"ds": ..., "category": ..., "run": ...}

# ── Experiment loop (run-outer) ───────────────────────────────
for run in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"  RUN {run+1}/{N_RUNS}")
    print(f"{'='*60}")

    for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
        print(f"\n{'─'*60}")
        print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
        print(f"{'─'*60}")

        for cat_idx, category in enumerate(categories, 1):
            # Skip if this run index already completed for this category
            if len(results.get(ds_name, {}).get(category, [])) > run:
                print(f"  [skip] {category} ({cat_idx}/{len(categories)}) — run {run+1} already done")
                continue

            if ds_name not in results:
                results[ds_name] = {}
            if category not in results[ds_name]:
                results[ds_name][category] = []

            print(f"\n  [{cat_idx}/{len(categories)}] {category}  —  Run {run+1}/{N_RUNS}")

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
                image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro,
                                                    image_f1max, pixel_f1max])

                print("  → Building model (repvit_m1 + repvit_m0_9)...")
                model = RepViTEfficientAd(
                    imagenet_dir="./datasets/imagenette",
                    teacher_out_channels=384,
                    padding=True,
                    pad_maps=False,
                    lr=1e-4,
                    weight_decay=1e-5,
                    pre_processor=EfficientAd.configure_pre_processor(image_size=(256, 256)),
                    evaluator=evaluator,
                    visualizer=False,
                )

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  → Parameters: {n_params:,}")

                print("  → Training (20 epochs)...")
                engine = Engine(max_epochs=20, devices="auto", strategy="auto",
                                logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)

                # ── FLOPs: teacher backbone forward pass on one image (PyTorch native) ─────────
                flops_M = None
                try:
                    device = next(model.parameters()).device
                    _dummy = torch.randn(1, 3, 256, 256, device=device)
                    with FlopCounterMode(display=False) as fcm:
                        _ = model.model.teacher(_dummy)
                    flops_M = round(fcm.get_total_flops() / 1e6, 1)
                    print(f"  → Teacher FLOPs: {flops_M:.1f} M")
                    del _dummy
                except Exception:
                    pass

                # ── Throughput, CPU latency, single-forward peak mem ─────────
                extractor = model.model.teacher
                input_shape = (1, 3, 256, 256)

                try:
                    gpu_tput = _gpu_throughput(extractor, input_shape=input_shape, batch=8)
                    print(f"  → GPU throughput: {gpu_tput:.1f} img/s" if gpu_tput else "")
                except Exception:
                    gpu_tput = None

                try:
                    cpu_lat = _cpu_latency(extractor, input_shape=input_shape)
                    print(f"  → CPU latency: {cpu_lat:.1f} ms" if cpu_lat else "")
                except Exception:
                    cpu_lat = None

                try:
                    peak_mem_1fwd = _peak_gpu_mem(extractor, input_shape=input_shape)
                    print(f"  → Peak mem (1 fwd): {peak_mem_1fwd/1e6:.1f} MB" if peak_mem_1fwd else "")
                except Exception:
                    peak_mem_1fwd = None

                # ── GPU inference on test set ──────────────────────────────────
                print(f"  → Running inference on test set (GPU)...")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                test_results = engine.test(model=model, datamodule=datamodule)
                elapsed_gpu = time.time() - t0

                peak_gpu_mb = 0.0
                if torch.cuda.is_available():
                    peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

                metrics = test_results[0]
                img_auc  = metrics.get("image_AUROC", 0) * 100
                pxl_auc  = metrics.get("pixel_AUROC", 0) * 100
                pxl_pro  = metrics.get("pixel_AUPRO", 0) * 100
                img_f1   = (metrics["image_F1Max"] * 100) if metrics.get("image_F1Max") is not None else None
                pxl_f1   = (metrics["pixel_F1Max"] * 100) if metrics.get("pixel_F1Max") is not None else None

                print(f"  → Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                      f" | AUPRO {pxl_pro:.1f}%"
                      + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else ""))
                print(f"  → GPU inf: {elapsed_gpu:.3f}s  |  Peak GPU: {peak_gpu_mb:.1f} MB")

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
                    "gpu_throughput":  gpu_tput,
                    "cpu_latency_ms":  cpu_lat,
                    "peak_gpu_mem_b":  peak_mem_1fwd,
                }
                results[ds_name][category].append(run_record)

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → Checkpoint saved.")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_skips.append({"ds": ds_name, "category": category, "run": run + 1})
                    print(f"\n  [OOM] CUDA out of memory — {ds_name}/{category} run {run+1}. "
                          f"Skipping. (total OOM skips so far: {len(oom_skips)})")
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max
                del test_results, metrics
                free_gpu()

# ── Post-loop: clean up checkpoint ───────────────────────────
print(f"\n{'='*60}")
print("  All runs complete.")
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"  Checkpoint deleted ({PROGRESS_FILE}).")
    print("  Next invocation will start fresh.")
print(f"{'='*60}\n")

# ── Build pandas DataFrame & unified summary ──────────────────
rows = []
for ds_name, _, categories, _ in DATASETS:
    for cat in categories:
        runs_list = results.get(ds_name, {}).get(cat, [])
        for run_idx, rec in enumerate(runs_list):
            rows.append({
                "dataset": ds_name, "category": cat, "run": run_idx + 1,
                "img_auroc": rec.get("image_AUROC"), "pxl_auroc": rec.get("pixel_AUROC"),
                "aupro": rec.get("pixel_AUPRO"), "img_f1max": rec.get("image_F1Max"),
                "pxl_f1max": rec.get("pixel_F1Max"),
                "params": rec.get("n_params"), "flops_M": rec.get("flops_M"),
                "inf_s": rec.get("inference_gpu_s"), "peak_gpu_mb": rec.get("peak_gpu_mb"),
                "gpu_tput": rec.get("gpu_throughput"), "cpu_lat_ms": rec.get("cpu_latency_ms"),
                "peak_mem_b": rec.get("peak_gpu_mem_b"),
            })

if not rows:
    print("No results to display.")
    sys.exit(0)

df = pd.DataFrame(rows)
csv_path = "results/repvit_efficientad_results.csv"
df.to_csv(csv_path, index=False)

metric_cols = ["img_auroc", "pxl_auroc", "aupro", "img_f1max", "pxl_f1max",
               "params", "flops_M", "inf_s", "peak_gpu_mb",
               "gpu_tput", "cpu_lat_ms", "peak_mem_b"]

if N_RUNS > 1:
    cat_agg = df.groupby(["dataset", "category"])[metric_cols].agg(["mean", "std"])
    cat_agg.columns = [f"{c[0]}" if c[1] == "mean" else f"{c[0]}_std" for c in cat_agg.columns]
    std_cols = [f"{c}_std" for c in metric_cols]
else:
    cat_agg = df.groupby(["dataset", "category"])[metric_cols].mean()
    std_cols = []
display_cols = metric_cols + std_cols

ds_rows = []
for ds_name, _, _, _ in DATASETS:
    ds_df = df[df["dataset"] == ds_name]
    if ds_df.empty:
        continue
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    row = {"dataset": ds_name, "category": f"~ {label}"}
    row.update(ds_df[metric_cols].mean().to_dict())
    ds_rows.append(row)
ds_summary = pd.DataFrame(ds_rows).set_index(["dataset", "category"]) if ds_rows else None

ov = df[metric_cols].mean().to_dict()
ov.update({"dataset": "", "category": "~ OVERALL"})
overall_df = pd.DataFrame([ov]).set_index(["dataset", "category"])

def _align(frame, cols):
    if frame is None: return None
    frame = frame.copy()
    for c in cols:
        if c not in frame.columns: frame[c] = float("nan")
    return frame[cols]

pieces = [_align(cat_agg, display_cols)]
if ds_summary is not None: pieces.append(_align(ds_summary, display_cols))
pieces.append(_align(overall_df, display_cols))
summary = pd.concat(pieces)

print(f"\n{'=' * 140}")
print(f"  RepViT-EfficientAD  (N={N_RUNS}, Teacher: repvit_m1, Student: repvit_m0_9)")
print(f"  Raw CSV: {csv_path}")
print(f"{'=' * 140}")
print(summary.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "—"))

if oom_skips:
    print(f"\n{'=' * 80}\n  OOM Skips ({len(oom_skips)} total)\n{'=' * 80}")
    for s in oom_skips: print(f"  • {s['ds']}/{s['category']}  run {s['run']}")
else:
    print("\n  No OOM skips recorded.")
print("=" * 80)
