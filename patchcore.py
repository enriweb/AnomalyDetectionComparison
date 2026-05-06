"""
PatchCore-25%

Usage:
    python patchcore.py          # 1 run (default)
    python patchcore.py 3        # 3 runs, results averaged
"""

import gc
import json
import logging
import os
import statistics
import sys
import time
import warnings

# ── Silence warnings and noisy loggers ───────────────────────
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)

import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Patchcore
from lightning.pytorch.callbacks import TQDMProgressBar

# ── Config ────────────────────────────────────────────────────
N_RUNS: int        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_VRAM_GB: float = 8.0   # reserved VRAM limit before each run; 0 = disabled

# ── GPU helpers ───────────────────────────────────────────────
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def vram_ok() -> bool:
    if not torch.cuda.is_available() or MAX_VRAM_GB <= 0:
        return True
    return torch.cuda.memory_reserved() / 1e9 < MAX_VRAM_GB

# ── Single-line progress bar ──────────────────────────────────
class CompactBar(TQDMProgressBar):
    """TQDMProgressBar with leave=False so each bar stays on one line."""
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

os.makedirs("results", exist_ok=True)
PROGRESS_FILE = "results/patchcore_progress.json"

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"Resuming from {PROGRESS_FILE}")
else:
    results = {ds: {} for ds, *_ in DATASETS}

# ── Experiment loop ───────────────────────────────────────────
for ds_name, DataModule, categories, root in DATASETS:
    for category in categories:
        runs_done = len(results.get(ds_name, {}).get(category, []))
        if runs_done >= N_RUNS:
            print(f"[skip] {ds_name}/{category} — {runs_done}/{N_RUNS} runs done")
            continue

        if ds_name not in results:
            results[ds_name] = {}
        if category not in results[ds_name]:
            results[ds_name][category] = []

        for run in range(runs_done, N_RUNS):
            print(f"\n{'='*50}")
            print(f"  {ds_name.upper()}  |  {category}  |  Run {run+1}/{N_RUNS}")
            print(f"{'='*50}")

            if not vram_ok():
                print(f"[WARN] VRAM > {MAX_VRAM_GB:.1f} GB — freeing before run.")
                free_gpu()
                if not vram_ok():
                    print("[SKIP] VRAM still over threshold. Skipping run.")
                    continue

            model = engine = datamodule = evaluator = None
            image_auroc = pixel_auroc = pixel_pro = test_results = metrics = None
            try:
                # (§4.1): "No data augmentation is applied"
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=32,
                    eval_batch_size=32,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
                pixel_pro   = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
                evaluator   = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro])

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

                engine = Engine(max_epochs=1, logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)

                t0 = time.time()
                test_results = engine.test(model=model, datamodule=datamodule)
                elapsed = time.time() - t0

                metrics = test_results[0]
                results[ds_name][category].append({
                    "image_AUROC": metrics.get("image_AUROC", 0) * 100,
                    "pixel_AUROC": metrics.get("pixel_AUROC", 0) * 100,
                    "pixel_AUPRO": metrics.get("pixel_AUPRO", 0) * 100,
                    "inference_s": elapsed,
                })

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] CUDA out of memory — {ds_name}/{category} run {run+1}. Skipping.")
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, test_results, metrics
                free_gpu()

# ── Result helpers ────────────────────────────────────────────
def _mean(runs: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in runs)

def _std(runs: list[dict], key: str) -> float:
    return statistics.stdev(r[key] for r in runs) if len(runs) > 1 else 0.0

def _fmt(mean: float, std: float) -> str:
    return f"{mean:5.1f}±{std:4.1f}" if N_RUNS > 1 else f"{mean:10.1f}"

# ── Final results table ───────────────────────────────────────
out_path = "results/patchcore_combined.txt"
COL = 12 if N_RUNS > 1 else 10
W   = 15 + COL * 4 + 3
HDR = f"{'Category':<15} {'Img AUROC':>{COL}} {'Pxl AUROC':>{COL}} {'PRO':>{COL}} {'Infer(s)':>{COL}}"

lines = []
lines.append("=" * W)
lines.append(f"  PatchCore-25% Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append("  Backbone: WideResNet-50  |  Layers: 2+3  |  Coreset: 25%")
lines.append("=" * W)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    lines.append(f"\n--- {label} ({len(categories)} categories) ---")
    lines.append(HDR)
    lines.append("-" * W)

    cat_means: dict[str, dict[str, float]] = {}
    for cat in categories:
        runs = results.get(ds_name, {}).get(cat)
        if not runs:
            continue
        m_img = _mean(runs, "image_AUROC");  s_img = _std(runs, "image_AUROC")
        m_pxl = _mean(runs, "pixel_AUROC");  s_pxl = _std(runs, "pixel_AUROC")
        m_pro = _mean(runs, "pixel_AUPRO");  s_pro = _std(runs, "pixel_AUPRO")
        m_t   = _mean(runs, "inference_s");  s_t   = _std(runs, "inference_s")
        cat_means[cat] = {"img": m_img, "pxl": m_pxl, "pro": m_pro, "t": m_t}
        lines.append(
            f"{cat:<15}"
            f" {_fmt(m_img, s_img):>{COL}}"
            f" {_fmt(m_pxl, s_pxl):>{COL}}"
            f" {_fmt(m_pro, s_pro):>{COL}}"
            f" {_fmt(m_t,   s_t  ):>{COL}}"
        )

    if cat_means:
        avg = {k: statistics.mean(v[k] for v in cat_means.values()) for k in ("img", "pxl", "pro", "t")}
        lines.append("-" * W)
        lines.append(
            f"{'MEAN':<15}"
            f" {avg['img']:>{COL}.1f}"
            f" {avg['pxl']:>{COL}.1f}"
            f" {avg['pro']:>{COL}.1f}"
            f" {avg['t']:>{COL}.2f}"
        )

all_cats = [
    (ds, cat)
    for ds, _, cats, _ in DATASETS
    for cat in cats
    if results.get(ds, {}).get(cat)
]
n_all = len(all_cats)
if n_all:
    oi  = statistics.mean(_mean(results[ds][c], "image_AUROC") for ds, c in all_cats)
    op  = statistics.mean(_mean(results[ds][c], "pixel_AUROC") for ds, c in all_cats)
    or_ = statistics.mean(_mean(results[ds][c], "pixel_AUPRO") for ds, c in all_cats)
    ot  = statistics.mean(_mean(results[ds][c], "inference_s") for ds, c in all_cats)
    lines.append(f"\n--- Overall Mean ({n_all} categories, N={N_RUNS} runs each) ---")
    lines.append(f"Img AUROC: {oi:.1f}  |  Pxl AUROC: {op:.1f}  |  PRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
