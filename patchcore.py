"""
PatchCore-25%

Usage:
    python patchcore.py          # 1 run (default)
    python patchcore.py 3        # 3 runs, results averaged

Metrics recorded per run:
    Accuracy  — image AUROC, pixel AUROC, pixel AUPRO,
                image F1Max (optimal threshold), pixel F1Max
    Efficiency — n_params, FLOPs (backbone, thop optional),
                 inference time GPU + CPU (backbone),
                 peak GPU MB, peak CPU MB (psutil optional)

Progress is checkpointed to results/patchcore_progress.json after every
completed run. On normal exit the checkpoint is deleted so the next
invocation starts fresh. On crash / OOM the checkpoint survives and the
next invocation auto-resumes from where it left off.
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

# ── Silence warnings and noisy loggers ───────────────────────
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries")
import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Patchcore
from lightning.pytorch.callbacks import TQDMProgressBar

# F1Max: anomalib ≥ 1.x; graceful fallback if missing
try:
    from anomalib.metrics import F1Max
    HAS_F1MAX = True
except ImportError:
    HAS_F1MAX = False

# FNR = FN/(FN+TP) at the F1-optimal threshold (lower is better)
try:
    from torchmetrics import Metric as _TMMetric

    class FNR(_TMMetric):
        """False Negative Rate at the threshold that maximises F1."""
        higher_is_better: bool = False
        full_state_update: bool = False

        def __init__(self, fields: list[str], prefix: str = "", num_thresh: int = 200):
            super().__init__()
            self.fields = fields
            self.prefix = prefix
            self._num_thresh = num_thresh
            self.add_state("preds",  default=[], dist_reduce_fx="cat")
            self.add_state("labels", default=[], dist_reduce_fx="cat")

        def update(self, **kwargs) -> None:
            self.preds.append(kwargs[self.fields[0]].float().detach().flatten())
            self.labels.append(kwargs[self.fields[1]].float().detach().flatten())

        def compute(self) -> torch.Tensor:
            p  = torch.cat(self.preds)
            lb = torch.cat(self.labels).long()
            ts = torch.linspace(p.min(), p.max(), self._num_thresh, device=p.device)
            best_fnr, best_f1 = torch.tensor(1.0, device=p.device), -1.0
            for t in ts:
                pb = (p >= t).long()
                tp = ((pb == 1) & (lb == 1)).sum().float()
                fp = ((pb == 1) & (lb == 0)).sum().float()
                fn = ((pb == 0) & (lb == 1)).sum().float()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
                if f1.item() > best_f1:
                    best_f1 = f1.item()
                    best_fnr = fn / (fn + tp + 1e-8)
            return best_fnr

    HAS_FNR = True
except Exception:
    HAS_FNR = False

# FLOPs counting (pip install thop)
try:
    from thop import profile as _thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# CPU memory tracking (pip install psutil)
try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Config ────────────────────────────────────────────────────
N_RUNS: int        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_VRAM_GB: float = 0   # reserved VRAM limit before each run; 0 = disabled
torch.set_float32_matmul_precision('high')
# ── GPU helpers ───────────────────────────────────────────────
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _start_mem_sampler(samples: list, use_gpu: bool, interval: float = 0.05) -> threading.Event:
    """Spawn a daemon thread that appends GPU or RSS memory samples every `interval` s."""
    stop = threading.Event()
    def _run():
        proc = _psutil.Process(os.getpid()) if (not use_gpu and HAS_PSUTIL) else None
        while not stop.is_set():
            if use_gpu and torch.cuda.is_available():
                samples.append(torch.cuda.memory_allocated() / 1e6)
            elif proc is not None:
                samples.append(proc.memory_info().rss / 1e6)
            stop.wait(interval)
    threading.Thread(target=_run, daemon=True).start()
    return stop

def vram_ok() -> bool:
    if not torch.cuda.is_available() or MAX_VRAM_GB <= 0:
        return True
    return torch.cuda.memory_reserved() / 1e9 < MAX_VRAM_GB

# ── Single-line progress bar ──────────────────────────────────
class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm(); bar.leave = False; return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm(); bar.leave = False; return bar
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm(); bar.leave = False; return bar

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
PROGRESS_FILE = "results/patchcore_progress.json"

# ── Startup banner ────────────────────────────────────────────
print("=" * 60)
print("  PatchCore-25%  |  WideResNet-50  |  Layers 2+3  |  Coreset 25%")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
print(f"  Max VRAM limit    : {MAX_VRAM_GB:.1f} GB  ({'disabled' if MAX_VRAM_GB <= 0 else 'active'})")
print(f"  FLOPs (thop)      : {'available' if HAS_THOP else 'not installed — skipped'}")
print(f"  CPU mem (psutil)  : {'available' if HAS_PSUTIL else 'not installed — skipped'}")
print(f"  F1Max metric      : {'available' if HAS_F1MAX else 'not in this anomalib version — skipped'}")
print(f"  FNR metric        : {'available' if HAS_FNR else 'torchmetrics missing — skipped'}")
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

# ── Experiment loop ───────────────────────────────────────────
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
                vram_used = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [WARN] VRAM {vram_used:.2f} GB > limit {MAX_VRAM_GB:.1f} GB — freeing...")
                free_gpu()
                if not vram_ok():
                    print(f"  [SKIP] VRAM still over threshold after free. Skipping run {run+1}.")
                    continue

            model = engine = datamodule = evaluator = None
            image_auroc = pixel_auroc = pixel_pro = None
            image_f1max = pixel_f1max = image_fnr = pixel_fnr = test_results = metrics = None
            try:
                print(f"  → Loading datamodule...")
                # (§4.1): "No data augmentation is applied"
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=32,
                    eval_batch_size=32,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                # fpr_limit=0.3 matches paper Table 3
                pixel_pro   = AUPRO(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                test_metrics = [image_auroc, pixel_auroc, pixel_pro]
                if HAS_F1MAX:
                    # F1 at optimal decision threshold (image + pixel level)
                    image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                    pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                    test_metrics += [image_f1max, pixel_f1max]
                if HAS_FNR:
                    image_fnr = FNR(fields=["pred_score", "gt_label"], prefix="image_")
                    pixel_fnr = FNR(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                    test_metrics += [image_fnr, pixel_fnr]
                evaluator = Evaluator(test_metrics=test_metrics)

                print(f"  → Building model (WRN50 backbone)...")
                # (§3.1, §3.2, §4.4.1)
                model = Patchcore(
                    # (§3.1): "WideResnet-50" — ImageNet-pretrained backbone
                    backbone="wide_resnet50_2",
                    # (§4.4.1, Fig.4 bottom): "2+3, which is chosen as the default setting"
                    layers=["layer2", "layer3"],
                    pre_trained=True,
                    # (§3.2, Table 1): PatchCore-25% = 25% coreset subsampling
                    coreset_sampling_ratio=0.25,
                    # (Eq.7): num_neighbors b for re-weighting (anomalib default=9)
                    num_neighbors=9,
                    # (§4.1): "images are resized and center cropped to 256×256 and 224×224"
                    pre_processor=Patchcore.configure_pre_processor(
                        image_size=(256, 256),
                        center_crop_size=(224, 224),
                    ),
                    evaluator=evaluator,
                    visualizer=False,
                )

                # Parameter count (backbone only; coreset is a buffer, not params)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  → Parameters: {n_params:,}")

                print(f"  → Training (building coreset)...")
                engine = Engine(max_epochs=1, logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)
                print(f"  → Coreset built.")

                # ── FLOPs: backbone forward pass on one image ─────────
                flops_M = None
                if HAS_THOP:
                    try:
                        device = next(model.parameters()).device
                        _dummy = torch.randn(1, 3, 224, 224, device=device)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _flops, _ = _thop_profile(
                                model.model.feature_extractor, inputs=(_dummy,), verbose=False
                            )
                        flops_M = round(_flops / 1e6, 1)
                        print(f"  → Backbone FLOPs: {flops_M:.1f} M")
                        del _dummy
                    except Exception:
                        pass

                # ── CPU inference: backbone on 1 image (warmup + avg of 5) ──
                # tracemalloc → peak alloc; RSS sampler thread → mean memory.
                inference_cpu_s = None
                peak_cpu_mb = None
                mean_cpu_mb = None
                try:
                    import tracemalloc
                    print(f"  → Timing CPU inference (backbone, 1 image × 5 runs)...")
                    _feat = model.model.feature_extractor.cpu().eval()
                    # verify weights actually moved to CPU
                    _dev = next(_feat.parameters()).device
                    if str(_dev) != "cpu":
                        raise RuntimeError(f"Backbone still on {_dev} after .cpu()")
                    _dummy_cpu = torch.randn(1, 3, 224, 224)
                    # warmup — 3 passes to stabilise branch prediction / caches
                    for _ in range(3):
                        with torch.no_grad():
                            _feat(_dummy_cpu)
                    # timed — avg 5 passes; RSS sampler runs concurrently for mean
                    _cpu_rss_samples: list = []
                    _cpu_stop = _start_mem_sampler(_cpu_rss_samples, use_gpu=False)
                    _baseline_rss = (_psutil.Process(os.getpid()).memory_info().rss / 1e6
                                     if HAS_PSUTIL else 0.0)
                    tracemalloc.start()
                    _times = []
                    for _ in range(5):
                        _t = time.perf_counter()
                        with torch.no_grad():
                            _feat(_dummy_cpu)
                        _times.append(time.perf_counter() - _t)
                    _, _peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    _cpu_stop.set()
                    inference_cpu_s = round(statistics.mean(_times), 4)
                    peak_cpu_mb     = round(_peak_bytes / 1e6, 1)
                    if _cpu_rss_samples:
                        mean_cpu_mb = round(
                            max(0.0, statistics.mean(_cpu_rss_samples) - _baseline_rss), 1
                        )
                    print(f"  → CPU inference: {inference_cpu_s:.4f}s (avg 5 runs)"
                          f"  |  peak: {peak_cpu_mb:.1f} MB"
                          + (f"  |  mean: {mean_cpu_mb:.1f} MB" if mean_cpu_mb is not None else ""))
                    del _dummy_cpu
                    # restore backbone to GPU for the upcoming engine.test()
                    if torch.cuda.is_available():
                        model.model.feature_extractor.cuda()
                except Exception as _cpu_err:
                    print(f"  [WARN] CPU timing failed: {_cpu_err}")

                # ── GPU inference with peak + mean VRAM tracking ─────
                print(f"  → Running inference on test set (GPU)...")
                _gpu_samples: list = []
                _gpu_stop = _start_mem_sampler(_gpu_samples, use_gpu=True)
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
                img_fnr  = (metrics["image_FNR"] * 100) if metrics.get("image_FNR") is not None else None
                pxl_fnr  = (metrics["pixel_FNR"] * 100) if metrics.get("pixel_FNR") is not None else None

                print(f"  → Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                      f" | AUPRO {pxl_pro:.1f}%"
                      + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else "")
                      + (f" | ImgFNR {img_fnr:.1f}%" if img_fnr is not None else ""))
                print(f"  → GPU inf: {elapsed_gpu:.3f}s  |  Peak GPU: {peak_gpu_mb:.1f} MB"
                      f"  |  Mean GPU: {mean_gpu_mb:.1f} MB")

                results[ds_name][category].append({
                    # ── Accuracy ──────────────────────────────────────
                    "image_AUROC":     img_auc,
                    "pixel_AUROC":     pxl_auc,
                    "pixel_AUPRO":     pxl_pro,
                    "image_F1Max":     img_f1,
                    "pixel_F1Max":     pxl_f1,
                    "image_FNR":       img_fnr,
                    "pixel_FNR":       pxl_fnr,
                    # ── Efficiency ────────────────────────────────────
                    "n_params":        n_params,
                    "flops_M":         flops_M,
                    "inference_gpu_s": elapsed_gpu,
                    "inference_cpu_s": inference_cpu_s,
                    "peak_gpu_mb":     peak_gpu_mb,
                    "mean_gpu_mb":     mean_gpu_mb,
                    "peak_cpu_mb":     peak_cpu_mb,
                    "mean_cpu_mb":     mean_cpu_mb,
                })

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
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max, image_fnr, pixel_fnr
                del test_results, metrics
                free_gpu()

# ── Post-loop: clean up checkpoint + OOM summary ─────────────
print(f"\n{'='*60}")
print("  All runs complete.")

if oom_skips:
    print(f"\n  [OOM summary] {len(oom_skips)} run(s) skipped due to CUDA out of memory:")
    for s in oom_skips:
        print(f"    • {s['ds']}/{s['category']}  run {s['run']}")
else:
    print("  No OOM skips.")

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"\n  Checkpoint deleted ({PROGRESS_FILE}).")
    print("  Next invocation will start fresh.")

print(f"{'='*60}\n")

# ── Result helpers ────────────────────────────────────────────
def _mean(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else float("nan")

def _std(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def _fmt(mean: float, std: float, decimals: int = 1) -> str:
    if mean != mean:  # nan
        return "n/a".rjust(12 if N_RUNS > 1 else 10)
    fmt = f"{{:{5+decimals}.{decimals}f}}±{{:4.{decimals}f}}" if N_RUNS > 1 else f"{{:10.{decimals}f}}"
    return fmt.format(mean, std) if N_RUNS > 1 else fmt.format(mean)

# ── Final results tables ──────────────────────────────────────
out_path = "results/patchcore_combined.txt"
C  = 12 if N_RUNS > 1 else 10
CE = 11  # efficiency column width

lines = []
SEP = "=" * 80

lines.append(SEP)
lines.append(f"  PatchCore-25% Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append("  Backbone: WideResNet-50  |  Layers: 2+3  |  Coreset: 25%")
lines.append(SEP)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"

    # ── Accuracy table ────────────────────────────────────────
    lines.append(f"\n--- {label} — Accuracy ---")
    HDR_A = (f"{'Category':<15} {'Img AUROC':>{C}} {'Pxl AUROC':>{C}}"
             f" {'AUPRO':>{C}} {'Img F1Max':>{C}} {'Pxl F1Max':>{C}}"
             f" {'Img FNR':>{C}} {'Pxl FNR':>{C}}")
    lines.append(HDR_A)
    lines.append("-" * len(HDR_A))

    cat_acc: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        r = {k: (_mean(runs, k), _std(runs, k)) for k in
             ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max",
              "image_FNR", "pixel_FNR")}
        cat_acc[cat] = {k: v[0] for k, v in r.items()}
        lines.append(
            f"{cat:<15}"
            f" {_fmt(*r['image_AUROC']):>{C}}"
            f" {_fmt(*r['pixel_AUROC']):>{C}}"
            f" {_fmt(*r['pixel_AUPRO']):>{C}}"
            f" {_fmt(*r['image_F1Max']):>{C}}"
            f" {_fmt(*r['pixel_F1Max']):>{C}}"
            f" {_fmt(*r['image_FNR']):>{C}}"
            f" {_fmt(*r['pixel_FNR']):>{C}}"
        )

    if cat_acc:
        avg = {k: statistics.mean(v for v in (c[k] for c in cat_acc.values()) if v == v)
               for k in ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max",
                         "image_FNR", "pixel_FNR")}
        lines.append("-" * len(HDR_A))
        lines.append(
            f"{'MEAN':<15}"
            f" {avg['image_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUROC']:>{C}.1f}"
            f" {avg['pixel_AUPRO']:>{C}.1f}"
            f" {avg.get('image_F1Max', float('nan')):>{C}.1f}"
            f" {avg.get('pixel_F1Max', float('nan')):>{C}.1f}"
            f" {avg.get('image_FNR', float('nan')):>{C}.1f}"
            f" {avg.get('pixel_FNR', float('nan')):>{C}.1f}"
        )

    # ── Efficiency table ──────────────────────────────────────
    lines.append(f"\n--- {label} — Efficiency ---")
    HDR_E = (f"{'Category':<15} {'Params':>{CE}} {'FLOPs(M)':>{CE}}"
             f" {'GPU inf(s)':>{CE}} {'CPU inf(s)':>{CE}}"
             f" {'PkGPU(MB)':>{CE}} {'MnGPU(MB)':>{CE}}"
             f" {'PkCPU(MB)':>{CE}} {'MnCPU(MB)':>{CE}}")
    lines.append(HDR_E)
    lines.append("-" * len(HDR_E))

    def _fe(v, fmt) -> str:
        return (fmt.format(v) if v == v else "n/a").rjust(CE)

    cat_eff: dict[str, dict] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        np_  = _mean(runs, "n_params")
        fl_  = _mean(runs, "flops_M")
        gi_  = _mean(runs, "inference_gpu_s")
        ci_  = _mean(runs, "inference_cpu_s")
        pgm_ = _mean(runs, "peak_gpu_mb")
        mgm_ = _mean(runs, "mean_gpu_mb")
        pcm_ = _mean(runs, "peak_cpu_mb")
        mcm_ = _mean(runs, "mean_cpu_mb")
        cat_eff[cat] = {"np": np_, "fl": fl_, "gpu": gi_, "cpu": ci_,
                        "pgm": pgm_, "mgm": mgm_, "pcm": pcm_, "mcm": mcm_}
        lines.append(
            f"{cat:<15}"
            f" {_fe(np_,  '{:,.0f}')}"
            f" {_fe(fl_,  '{:.1f}')}"
            f" {_fe(gi_,  '{:.3f}')}"
            f" {_fe(ci_,  '{:.4f}')}"
            f" {_fe(pgm_, '{:.1f}')}"
            f" {_fe(mgm_, '{:.1f}')}"
            f" {_fe(pcm_, '{:.1f}')}"
            f" {_fe(mcm_, '{:.1f}')}"
        )

    if cat_eff:
        lines.append("-" * len(HDR_E))
        _first = next(iter(cat_eff.values()))
        def _avg_eff(key):
            vals = [v[key] for v in cat_eff.values() if v[key] == v[key]]
            return statistics.mean(vals) if vals else float("nan")
        lines.append(
            f"{'MEAN':<15}"
            f" {_fe(_first['np'],  '{:,.0f}')}"
            f" {_fe(_first['fl'],  '{:.1f}')}"
            f" {_fe(_avg_eff('gpu'), '{:.3f}')}"
            f" {_fe(_avg_eff('cpu'), '{:.4f}')}"
            f" {_fe(_avg_eff('pgm'), '{:.1f}')}"
            f" {_fe(_avg_eff('mgm'), '{:.1f}')}"
            f" {_fe(_avg_eff('pcm'), '{:.1f}')}"
            f" {_fe(_avg_eff('mcm'), '{:.1f}')}"
        )

# ── Overall summary ───────────────────────────────────────────
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
    oi   = _omean("image_AUROC")
    op   = _omean("pixel_AUROC")
    opr  = _omean("pixel_AUPRO")
    of1i  = _omean("image_F1Max")
    of1p  = _omean("pixel_F1Max")
    ofnri = _omean("image_FNR")
    ofnrp = _omean("pixel_FNR")
    ogi   = _omean("inference_gpu_s")
    lines.append(f"\n{'='*80}")
    lines.append(f"  Overall Mean ({n_all} categories, N={N_RUNS} runs each)")
    lines.append(f"{'='*80}")
    lines.append(f"  Img AUROC : {oi:.2f}%   Pxl AUROC : {op:.2f}%   AUPRO : {opr:.2f}%")
    lines.append(f"  Img F1Max : {of1i:.2f}%   Pxl F1Max : {of1p:.2f}%")
    lines.append(f"  Img FNR   : {ofnri:.2f}%   Pxl FNR   : {ofnrp:.2f}%")
    lines.append(f"  GPU inf   : {ogi:.3f}s")

# ── OOM section in results file ───────────────────────────────
if oom_skips:
    lines.append(f"\n{'='*80}")
    lines.append(f"  OOM Skips ({len(oom_skips)} total)")
    lines.append(f"{'='*80}")
    for s in oom_skips:
        lines.append(f"  • {s['ds']}/{s['category']}  run {s['run']}")
else:
    lines.append("\n  No OOM skips recorded.")

lines.append("=" * 80)

output = "\n".join(lines)
print(output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
