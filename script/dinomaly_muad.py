"""
Dinomaly — MUAD (Multi-class Unified Anomaly Detection) variant

Paper protocol: ONE Dinomaly trained jointly on all 15 MVTec categories,
then evaluated per-category. Reproduces the headline numbers in the
Dinomaly paper (Guo+ 2025) Table 1.

Usage:
    python dinomaly_muad.py          # 1 run (default)
    python dinomaly_muad.py 3        # 3 runs, results averaged

Differences vs. dinomaly2.py (per-category sibling):
  • Trains a single model on a merged datamodule (all MVTec categories
    concatenated via the dataset's __add__ operator).
  • Eval loop iterates per-category by constructing a fresh MVTecAD
    test datamodule for each category and calling engine.test on it.
  • VisA is not part of paper's MUAD protocol; this script targets MVTec only.

WARNING: ViT-B/14 + 448×448 + batch=16 needs ≳ 24 GB VRAM. If you OOM,
         lower the batch size or switch ENCODER to "dinov2reg_vit_small_14".
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
print("Importing torch-related libraries...")
import torch
from anomalib.data import MVTecAD
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.utils import Split
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Dinomaly
from lightning.pytorch.callbacks import TQDMProgressBar

torch.set_float32_matmul_precision('high')

from anomalib.metrics import F1Max

# ── Config ────────────────────────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROOT        = "./datasets/MVTecAD"
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

os.makedirs("results", exist_ok=True)
PROGRESS_FILE = "results/dinomaly_muad_progress.json"

# ── Helpers ───────────────────────────────────────────────────
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm(); bar.leave = False; return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm(); bar.leave = False; return bar
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm(); bar.leave = False; return bar

# ── Multi-category datamodule ─────────────────────────────────
class MultiCategoryMVTecAD(MVTecAD):
    """MVTecAD subclass that loads multiple categories into a single,
    concatenated train/test set via AnomalibDataset.__add__.
    Used only for the MUAD training pass; per-category evaluation
    re-uses the stock MVTecAD with category=<cat>.
    """
    def __init__(self, root, categories: list[str], **kwargs):
        super().__init__(root=root, category=categories[0], **kwargs)
        self._categories = list(categories)

    def _setup(self, _stage=None) -> None:
        train_data = test_data = None
        for cat in self._categories:
            tr = MVTecADDataset(split=Split.TRAIN, root=self.root, category=cat)
            te = MVTecADDataset(split=Split.TEST,  root=self.root, category=cat)
            train_data = tr if train_data is None else train_data + tr
            test_data  = te if test_data  is None else test_data  + te
        self.train_data = train_data
        self.test_data  = test_data


def make_evaluator():
    metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    ]
    return Evaluator(test_metrics=metrics)


def build_model(evaluator):
    return Dinomaly(
        encoder_name="dinov2reg_vit_base_14",
        bottleneck_dropout=0.2,
        decoder_depth=8,
        target_layers=None,
        fuse_layer_encoder=None,
        fuse_layer_decoder=None,
        remove_class_token=False,
        # Paper §4.1: 448×448 resize, 392 center crop
        pre_processor=Dinomaly.configure_pre_processor(
            image_size=(448, 448),
            crop_size=392,
        ),
        evaluator=evaluator,
        visualizer=False,
    )


# ── Banner ────────────────────────────────────────────────────
print("=" * 60)
print("  Dinomaly — MUAD  |  Encoder: dinov2reg_vit_base_14")
print(f"  batch=16  |  image=448  |  crop=392  |  max_steps=5000")
print(f"  Categories: {len(MVTEC_CATEGORIES)} (MVTec)")
print(f"  Runs: {N_RUNS}")
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device: {gpu_info}")
print("=" * 60)

# ── Resume ────────────────────────────────────────────────────
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"\n[resume] Loaded {PROGRESS_FILE}")
else:
    results = {"mvtec": {}}
    print("\n[start] No checkpoint found — starting fresh.")

# ── Main: per-run train + per-category eval ───────────────────
oom_skips: list[dict] = []

for run_idx in range(N_RUNS):
    runs_done_min = min(
        len(results["mvtec"].get(c, [])) for c in MVTEC_CATEGORIES
    ) if results["mvtec"] else 0
    if runs_done_min > run_idx:
        print(f"\n[skip] Run {run_idx+1}/{N_RUNS} already complete.")
        continue

    print(f"\n{'='*60}")
    print(f"  Run {run_idx+1}/{N_RUNS}  —  training MUAD model")
    print(f"{'='*60}")

    train_evaluator = make_evaluator()
    print(f"  → Building merged datamodule ({len(MVTEC_CATEGORIES)} cats)...")
    merged_dm = MultiCategoryMVTecAD(
        root=ROOT,
        categories=MVTEC_CATEGORIES,
        train_batch_size=16,
        eval_batch_size=16,
    )

    print(f"  → Building model...")
    model = build_model(train_evaluator)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  → Parameters: {n_params:,}")

    print(f"  → Training (max_steps=5000) on merged dataset...")
    engine = Engine(max_steps=5000, logger=False, callbacks=[CompactBar()])
    t_train = time.time()
    try:
        engine.fit(model=model, datamodule=merged_dm)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM] training OOM on run {run_idx+1}. Skipping.")
            oom_skips.append({"phase": "train", "run": run_idx + 1})
            del model, engine, merged_dm, train_evaluator
            free_gpu()
            continue
        raise
    train_secs = time.time() - t_train
    print(f"  → Training complete in {train_secs/60:.1f} min")

    # ── Per-category test ─────────────────────────────────────
    for cat_idx, category in enumerate(MVTEC_CATEGORIES, 1):
        runs_done = len(results["mvtec"].get(category, []))
        if runs_done > run_idx:
            continue
        if category not in results["mvtec"]:
            results["mvtec"][category] = []

        print(f"\n  [{cat_idx}/{len(MVTEC_CATEGORIES)}] eval {category}")
        cat_evaluator = make_evaluator()
        # Replace the model's evaluator so test-time metrics reset per cat.
        model.evaluator = cat_evaluator

        cat_dm = MVTecAD(
            root=ROOT,
            category=category,
            train_batch_size=16,
            eval_batch_size=16,
        )
        try:
            t0 = time.time()
            test_results = engine.test(model=model, datamodule=cat_dm)
            elapsed = time.time() - t0
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] eval OOM on {category}. Skipping.")
                oom_skips.append({"phase": "test", "run": run_idx + 1, "category": category})
                free_gpu()
                continue
            raise

        m = test_results[0]
        img_auc  = m.get("image_AUROC", 0) * 100
        pxl_auc  = m.get("pixel_AUROC", 0) * 100
        pxl_pro  = m.get("pixel_AUPRO", 0) * 100
        img_f1   = (m["image_F1Max"] * 100) if m.get("image_F1Max") is not None else None
        pxl_f1   = (m["pixel_F1Max"] * 100) if m.get("pixel_F1Max") is not None else None
        print(f"    Img AUROC {img_auc:.1f}%  Pxl AUROC {pxl_auc:.1f}%  AUPRO {pxl_pro:.1f}%"
              f"  ({elapsed:.1f}s)")

        results["mvtec"][category].append({
            "image_AUROC":     img_auc,
            "pixel_AUROC":     pxl_auc,
            "pixel_AUPRO":     pxl_pro,
            "image_F1Max":     img_f1,
            "pixel_F1Max":     pxl_f1,
            "n_params":        n_params,
            "inference_gpu_s": elapsed,
            "train_secs":      train_secs,
        })
        with open(PROGRESS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        free_gpu()

    del model, engine, merged_dm, train_evaluator
    free_gpu()

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  All MUAD runs complete.")
print(f"{'='*60}")

if oom_skips:
    print(f"\n  [OOM] {len(oom_skips)} skip(s):")
    for s in oom_skips:
        print(f"    • {s}")

# ── Tables ────────────────────────────────────────────────────
def _mean(runs, key):
    vs = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vs) if vs else float("nan")

def _std(runs, key):
    vs = [r[key] for r in runs if r.get(key) is not None]
    return statistics.stdev(vs) if len(vs) > 1 else 0.0

def _fmt(m, s, dec=1):
    if m != m:
        return "n/a".rjust(12 if N_RUNS > 1 else 10)
    fmt = f"{{:{5+dec}.{dec}f}}±{{:4.{dec}f}}" if N_RUNS > 1 else f"{{:10.{dec}f}}"
    return fmt.format(m, s) if N_RUNS > 1 else fmt.format(m)

C = 12 if N_RUNS > 1 else 10
lines = []
SEP = "=" * 80
lines.append(SEP)
lines.append(f"  Dinomaly MUAD Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  Encoder: dinov2reg_vit_base_14  |  one model jointly on {len(MVTEC_CATEGORIES)} categories")
lines.append(SEP)

HDR = (f"{'Category':<15} {'Img AUROC':>{C}} {'Pxl AUROC':>{C}} {'AUPRO':>{C}}"
       f" {'Img F1Max':>{C}} {'Pxl F1Max':>{C}}")
lines.append(HDR)
lines.append("-" * len(HDR))

cat_avg = {}
for cat in MVTEC_CATEGORIES:
    runs = results["mvtec"].get(cat)
    if not runs:
        continue
    r = {k: (_mean(runs, k), _std(runs, k)) for k in
         ("image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max")}
    cat_avg[cat] = {k: v[0] for k, v in r.items()}
    lines.append(
        f"{cat:<15}"
        f" {_fmt(*r['image_AUROC']):>{C}}"
        f" {_fmt(*r['pixel_AUROC']):>{C}}"
        f" {_fmt(*r['pixel_AUPRO']):>{C}}"
        f" {_fmt(*r['image_F1Max']):>{C}}"
        f" {_fmt(*r['pixel_F1Max']):>{C}}"
    )

if cat_avg:
    avg = {k: statistics.mean(v for v in (c[k] for c in cat_avg.values()) if v == v)
           for k in ("image_AUROC", "pixel_AUROC", "pixel_AUPRO",
                     "image_F1Max", "pixel_F1Max")}
    lines.append("-" * len(HDR))
    lines.append(
        f"{'MEAN':<15}"
        f" {avg['image_AUROC']:>{C}.1f}"
        f" {avg['pixel_AUROC']:>{C}.1f}"
        f" {avg['pixel_AUPRO']:>{C}.1f}"
        f" {avg['image_F1Max']:>{C}.1f}"
        f" {avg['pixel_F1Max']:>{C}.1f}"
    )

lines.append("=" * 80)
output = "\n".join(lines)
print(output)

with open("results/dinomaly_muad_combined.txt", "w") as f:
    f.write(output)

if os.path.exists(PROGRESS_FILE) and all(
    len(results["mvtec"].get(c, [])) >= N_RUNS for c in MVTEC_CATEGORIES
):
    os.remove(PROGRESS_FILE)
    print(f"\n  Checkpoint deleted ({PROGRESS_FILE}).")
