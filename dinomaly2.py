"""
Dinomaly2

Usage:
    python dinomaly2.py          # 1 run (default)
    python dinomaly2.py 3        # 3 runs, results averaged

NOTE: Paper trains one model for all categories jointly (MUAD).
      This script trains per-category to match other thesis runners.

Metrics recorded per run:
    Accuracy  - image AUROC, pixel AUROC, pixel AUPRO,
                image F1Max (optimal threshold), pixel F1Max
    Efficiency - n_params, FLOPs (encoder, thop optional),
                 inference time GPU (encoder),
                 peak + mean GPU MB

Progress is checkpointed to results/dinomaly2_progress.json after every
completed run. On normal exit the checkpoint is deleted. On crash / OOM
the checkpoint survives and the next invocation auto-resumes.
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

# - Silence warnings and noisy loggers -------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries...")
import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Dinomaly
from lightning.pytorch.callbacks import TQDMProgressBar

torch.set_float32_matmul_precision("high")

# F1Max: anomalib >= 1.x; graceful fallback if missing
try:
    from anomalib.metrics import F1Max
    HAS_F1MAX = True
except ImportError:
    HAS_F1MAX = False

# FLOPs counting (pip install thop)
try:
    from thop import profile as _thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# - Config -----------------------------------------------------
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_VRAM_GB: float = 0  # reserved VRAM limit before each run; 0 = disabled

# Dinomaly defaults with paper-aligned reasonable settings
ENCODER = "dinov2reg_vit_base_14"
BOTTLENECK_DROPOUT = 0.2
DECODER_DEPTH = 8
TRAIN_STEPS = 40000  # paper protocol (Guo+ 2025, Table I)
IMAGE_SIZE = (448, 448)
CROP_SIZE = 392
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

# Dinomaly2-specific settings from arXiv:2510.17611
# Paper uses middle 8 encoder layers {3..10} in 1-based indexing.
TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]  # 0-based for implementation
# Loose group-to-group reconstruction with 2 groups (Eq. 13-14, Fig. 5e).
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
# Context-Aware Recentering (Sec. 3.4, Eq. 12).
USE_CONTEXT_RECENTERING = True

# - GPU helpers ------------------------------------------------
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _start_mem_sampler(samples: list, interval: float = 0.05) -> threading.Event:
    """Spawn a daemon thread that appends GPU memory samples every `interval` s."""
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


# - Single-line progress bar -----------------------------------
class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.leave = False
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.leave = False
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.leave = False
        return bar


# - Dataset config ----------------------------------------------
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
    ("visa", Visa, VISA_CATEGORIES, "./datasets/VisA"),
]

total_categories = sum(len(c) for _, _, c, _ in DATASETS)

os.makedirs("results", exist_ok=True)
PROGRESS_FILE = "results/dinomaly2_progress.json"

# - Startup banner ----------------------------------------------
print("=" * 60)
print(f"  Dinomaly2  |  Encoder: {ENCODER}")
print(f"  dropout={BOTTLENECK_DROPOUT}  |  decoder_depth={DECODER_DEPTH}  |  train_steps={TRAIN_STEPS}")
print(f"  context_recentering={USE_CONTEXT_RECENTERING}  |  layer_groups={len(FUSE_LAYER_ENCODER)}")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
print(f"  Max VRAM limit    : {MAX_VRAM_GB:.1f} GB  ({'disabled' if MAX_VRAM_GB <= 0 else 'active'})")
print(f"  FLOPs (thop)      : {'available' if HAS_THOP else 'not installed - skipped'}")
print(f"  F1Max metric      : {'available' if HAS_F1MAX else 'not in this anomalib version - skipped'}")
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device            : {gpu_info}")
print("=" * 60)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"\n[resume] Found checkpoint - resuming from {PROGRESS_FILE}")
else:
    results = {ds: {} for ds, *_ in DATASETS}
    print("\n[start] No checkpoint found - starting fresh.")

# - OOM skip tracking -------------------------------------------
oom_skips: list[dict] = []

# - Experiment loop ---------------------------------------------
for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
    print(f"\n{'-' * 60}")
    print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
    print(f"{'-' * 60}")

    for cat_idx, category in enumerate(categories, 1):
        runs_done = len(results.get(ds_name, {}).get(category, []))
        if runs_done >= N_RUNS:
            print(f"  [skip] {category} ({cat_idx}/{len(categories)}) - {runs_done}/{N_RUNS} runs already done")
            continue

        if ds_name not in results:
            results[ds_name] = {}
        if category not in results[ds_name]:
            results[ds_name][category] = []

        print(f"\n  [{cat_idx}/{len(categories)}] {category}  - {N_RUNS - runs_done} run(s) remaining")

        for run in range(runs_done, N_RUNS):
            print(f"\n  {'=' * 48}")
            print(f"  {ds_name.upper()}  |  {category}  |  Run {run + 1}/{N_RUNS}")
            print(f"  {'=' * 48}")

            if not vram_ok():
                vram_used = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [WARN] VRAM {vram_used:.2f} GB > limit {MAX_VRAM_GB:.1f} GB - freeing...")
                free_gpu()
                if not vram_ok():
                    print(f"  [SKIP] VRAM still over threshold after free. Skipping run {run + 1}.")
                    continue

            model = engine = datamodule = evaluator = None
            image_auroc = pixel_auroc = pixel_pro = None
            image_f1max = pixel_f1max = test_results = metrics = None
            try:
                print("  -> Loading datamodule...")
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=TRAIN_BATCH_SIZE,
                    eval_batch_size=EVAL_BATCH_SIZE,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
                pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
                test_metrics = [image_auroc, pixel_auroc, pixel_pro]
                if HAS_F1MAX:
                    image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                    pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
                    test_metrics += [image_f1max, pixel_f1max]
                evaluator = Evaluator(test_metrics=test_metrics)

                print(f"  -> Building model ({ENCODER})...")
                model = Dinomaly(
                    encoder_name=ENCODER,
                    bottleneck_dropout=BOTTLENECK_DROPOUT,
                    decoder_depth=DECODER_DEPTH,
                    target_layers=TARGET_LAYERS,
                    fuse_layer_encoder=FUSE_LAYER_ENCODER,
                    fuse_layer_decoder=FUSE_LAYER_DECODER,
                    remove_class_token=False,
                    pre_processor=Dinomaly.configure_pre_processor(
                        image_size=IMAGE_SIZE,
                        crop_size=CROP_SIZE,
                    ),
                    evaluator=evaluator,
                    visualizer=False,
                )

                # Context-Aware Recentering (Dinomaly2 paper, Sec 3.4, Eq.12):
                # Subtract class token from patch tokens before reconstruction.
                if USE_CONTEXT_RECENTERING:
                    _orig_encoder_forward = model.model.encoder.forward

                    def _recentered_forward(x):
                        result = _orig_encoder_forward(x)
                        if isinstance(result, dict):
                            for k in list(result.keys()):
                                v = result[k]
                                if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[1] > 1:
                                    cls = v[:, 0:1, :]
                                    result[k] = torch.cat([cls, v[:, 1:, :] - cls], dim=1)
                        return result

                    model.model.encoder.forward = _recentered_forward
                    print("  -> Context-aware recentering applied (encoder monkey-patch).")

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  -> Parameters: {n_params:,}")

                print(f"  -> Training (max_steps={TRAIN_STEPS})...")
                engine = Engine(max_steps=TRAIN_STEPS, devices="auto", strategy="auto", logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)
                print("  -> Training complete.")

                # - FLOPs: encoder forward pass on one image ----------
                flops_M = None
                if HAS_THOP:
                    try:
                        device = next(model.parameters()).device
                        _dummy = torch.randn(1, 3, 224, 224, device=device)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _flops, _ = _thop_profile(
                                model.model.encoder, inputs=(_dummy,), verbose=False
                            )
                        flops_M = round(_flops / 1e6, 1)
                        print(f"  -> Encoder FLOPs: {flops_M:.1f} M")
                        del _dummy
                    except Exception:
                        pass

                # - GPU inference timing + memory ---------------------
                print("  -> Running inference on test set (GPU)...")
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
                img_auc = metrics.get("image_AUROC", 0) * 100
                pxl_auc = metrics.get("pixel_AUROC", 0) * 100
                pxl_pro = metrics.get("pixel_AUPRO", 0) * 100
                img_f1 = (metrics["image_F1Max"] * 100) if metrics.get("image_F1Max") is not None else None
                pxl_f1 = (metrics["pixel_F1Max"] * 100) if metrics.get("pixel_F1Max") is not None else None

                print(
                    f"  -> Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                    f" | AUPRO {pxl_pro:.1f}%"
                    + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else "")
                )
                print(
                    f"  -> GPU inf: {elapsed_gpu:.3f}s  |  Peak GPU: {peak_gpu_mb:.1f} MB"
                    f"  |  Mean GPU: {mean_gpu_mb:.1f} MB"
                )

                results[ds_name][category].append(
                    {
                        # - Accuracy ------------------------------------
                        "image_AUROC": img_auc,
                        "pixel_AUROC": pxl_auc,
                        "pixel_AUPRO": pxl_pro,
                        "image_F1Max": img_f1,
                        "pixel_F1Max": pxl_f1,
                        # - Efficiency ----------------------------------
                        "n_params": n_params,
                        "flops_M": flops_M,
                        "inference_gpu_s": elapsed_gpu,
                        "peak_gpu_mb": peak_gpu_mb,
                        "mean_gpu_mb": mean_gpu_mb,
                    }
                )

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print("  -> Checkpoint saved.")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_skips.append({"ds": ds_name, "category": category, "run": run + 1})
                    print(
                        f"\n  [OOM] CUDA out of memory - {ds_name}/{category} run {run + 1}. "
                        f"Skipping. (total OOM skips so far: {len(oom_skips)})"
                    )
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max
                del test_results, metrics
                free_gpu()

# - Post-loop: cleanup checkpoint + OOM summary ----------------
print(f"\n{'=' * 60}")
print("  All runs complete.")

if oom_skips:
    print(f"\n  [OOM summary] {len(oom_skips)} run(s) skipped due to CUDA out of memory:")
    for s in oom_skips:
        print(f"    - {s['ds']}/{s['category']}  run {s['run']}")
else:
    print("  No OOM skips.")

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"\n  Checkpoint deleted ({PROGRESS_FILE}).")
    print("  Next invocation will start fresh.")

print(f"{'=' * 60}\n")

# - Result helpers ----------------------------------------------
def _mean(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else float("nan")


def _std(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _fmt(mean: float, std: float, decimals: int = 1) -> str:
    if mean != mean:
        return "n/a".rjust(12 if N_RUNS > 1 else 10)
    fmt = f"{{:{5 + decimals}.{decimals}f}}+-{{:4.{decimals}f}}" if N_RUNS > 1 else f"{{:10.{decimals}f}}"
    return fmt.format(mean, std) if N_RUNS > 1 else fmt.format(mean)


# - Final results tables ----------------------------------------
out_path = "results/dinomaly2_combined.txt"
C = 12 if N_RUNS > 1 else 10
CE = 11

lines = []
SEP = "=" * 80

lines.append(SEP)
lines.append(f"  Dinomaly2 Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  Encoder: {ENCODER}  |  dropout={BOTTLENECK_DROPOUT}  |  depth={DECODER_DEPTH}")
lines.append(f"  Context Recentering: {USE_CONTEXT_RECENTERING}  |  Groups: {len(FUSE_LAYER_ENCODER)}")
lines.append(SEP)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"

    # - Accuracy table -----------------------------------------
    lines.append(f"\n--- {label} - Accuracy ---")
    hdr_a = (
        f"{'Category':<15} {'Img AUROC':>{C}} {'Pxl AUROC':>{C}}"
        f" {'AUPRO':>{C}} {'Img F1Max':>{C}} {'Pxl F1Max':>{C}}"
    )
    lines.append(hdr_a)
    lines.append("-" * len(hdr_a))

    cat_acc: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        r = {
            k: (_mean(runs, k), _std(runs, k))
            for k in ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max")
        }
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
        avg = {
            k: statistics.mean(v for v in (c[k] for c in cat_acc.values()) if v == v)
            for k in ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max")
        }
        lines.append("-" * len(hdr_a))
        lines.append(
            f"{'MEAN':<15}"
            f" {avg['image_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUPRO']:>{C}.1f}"
            f" {avg.get('image_F1Max', float('nan')):>{C}.1f}"
            f" {avg.get('pixel_F1Max', float('nan')):>{C}.1f}"
        )

    # - Efficiency table ---------------------------------------
    lines.append(f"\n--- {label} - Efficiency ---")
    hdr_e = (
        f"{'Category':<15} {'Params':>{CE}} {'FLOPs(M)':>{CE}}"
        f" {'GPU inf(s)':>{CE}}"
        f" {'PkGPU(MB)':>{CE}} {'MnGPU(MB)':>{CE}}"
    )
    lines.append(hdr_e)
    lines.append("-" * len(hdr_e))

    def _fe(v, fmt) -> str:
        return (fmt.format(v) if v == v else "n/a").rjust(CE)

    cat_eff: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        np_ = _mean(runs, "n_params")
        fl_ = _mean(runs, "flops_M")
        gi_ = _mean(runs, "inference_gpu_s")
        pgm_ = _mean(runs, "peak_gpu_mb")
        mgm_ = _mean(runs, "mean_gpu_mb")
        cat_eff[cat] = {"np": np_, "fl": fl_, "gpu": gi_, "pgm": pgm_, "mgm": mgm_}
        lines.append(
            f"{cat:<15}"
            f" {_fe(np_, '{:,.0f}')}"
            f" {_fe(fl_, '{:.1f}')}"
            f" {_fe(gi_, '{:.3f}')}"
            f" {_fe(pgm_, '{:.1f}')}"
            f" {_fe(mgm_, '{:.1f}')}"
        )

    if cat_eff:
        lines.append("-" * len(hdr_e))
        _first = next(iter(cat_eff.values()))

        def _avg_eff(key):
            vals = [v[key] for v in cat_eff.values() if v[key] == v[key]]
            return statistics.mean(vals) if vals else float("nan")

        lines.append(
            f"{'MEAN':<15}"
            f" {_fe(_first['np'], '{:,.0f}')}"
            f" {_fe(_first['fl'], '{:.1f}')}"
            f" {_fe(_avg_eff('gpu'), '{:.3f}')}"
            f" {_fe(_avg_eff('pgm'), '{:.1f}')}"
            f" {_fe(_avg_eff('mgm'), '{:.1f}')}"
        )

# - Overall summary ---------------------------------------------
all_cats = [
    (ds, cat)
    for ds, _, cats, _ in DATASETS
    for cat in cats
    if results.get(ds, {}).get(cat)
]
n_all = len(all_cats)
if n_all:
    def _omean(key):
        vals = [_mean(results[ds][c], key) for ds, c in all_cats]
        vals = [v for v in vals if v == v]
        return statistics.mean(vals) if vals else float("nan")

    oi = _omean("image_AUROC")
    op = _omean("pixel_AUROC")
    opr = _omean("pixel_AUPRO")
    of1i = _omean("image_F1Max")
    of1p = _omean("pixel_F1Max")
    ogi = _omean("inference_gpu_s")
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Overall Mean ({n_all} categories, N={N_RUNS} runs each)")
    lines.append(f"{'=' * 80}")
    lines.append(f"  Img AUROC : {oi:.2f}%   Pxl AUROC : {op:.2f}%   AUPRO : {opr:.2f}%")
    lines.append(f"  Img F1Max : {of1i:.2f}%   Pxl F1Max : {of1p:.2f}%")
    lines.append(f"  GPU inf   : {ogi:.3f}s")

# - OOM section in results file --------------------------------
if oom_skips:
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  OOM Skips ({len(oom_skips)} total)")
    lines.append(f"{'=' * 80}")
    for s in oom_skips:
        lines.append(f"  - {s['ds']}/{s['category']}  run {s['run']}")
else:
    lines.append("\n  No OOM skips recorded.")

lines.append("=" * 80)

output = "\n".join(lines)
print(output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
