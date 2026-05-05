"""
PatchCore with MobileViT-S backbone

Usage:
    python mobilevit.py          # 1 run (default)
    python mobilevit.py 3        # 3 runs, results averaged
"""

import gc
import json
import os
import statistics
import sys
import time

import timm
import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Patchcore

# ── How many runs to average ──────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# ============================================================
#  BACKBONE CONFIGURATION
# ============================================================
# MobileViT-S from timm (ICLR 2022, §4.1: "5.6M params, 78.4% top-1")
BACKBONE = "mobilevit_s"

# --- Discover available feature extraction layers ---
_tmp = timm.create_model(BACKBONE, features_only=True, pretrained=True)
available_layers = _tmp.feature_info.module_name()
print(f"Available layers for {BACKBONE}: {available_layers}")
# Typical output: ['stages.0.0', 'stages.1.0', 'stages.2.0', 'stages.3.0', 'stages.4.0']
# We pick the two mid-level stages (analogous to layer2 + layer3 in ResNet)
LAYERS = [available_layers[2], available_layers[3]]
print(f"Using layers: {LAYERS}")
del _tmp
# ============================================================

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
PROGRESS_FILE = "results/mobilevit_progress.json"

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
            print(f"  Dataset: {ds_name.upper()}  |  Category: {category}  |  Run {run+1}/{N_RUNS}")
            print(f"{'='*50}")

            # Same preprocessing as PatchCore paper (§4.1)
            datamodule = DataModule(
                root=root,
                category=category,
                train_batch_size=4,
                eval_batch_size=16,
            )

            image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
            pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro])

            model = Patchcore(
                # MobileViT-S (ICLR 2022): lightweight ViT backbone
                backbone=BACKBONE,
                layers=LAYERS,
                pre_trained=True,
                # PatchCore paper (§3.2): 25% coreset subsampling
                coreset_sampling_ratio=0.25,
                # PatchCore paper (Eq.7): num_neighbors for re-weighting
                num_neighbors=9,
                # PatchCore paper (§4.1): resize 256, center crop 224
                pre_processor=Patchcore.configure_pre_processor(
                    image_size=(256, 256),
                    center_crop_size=(224, 224),
                ),
                evaluator=evaluator,
                visualizer=False,
            )

            engine = Engine(max_epochs=1, logger=False)
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

            del model, engine, datamodule, evaluator
            del image_auroc, pixel_auroc, pixel_pro, test_results, metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ── Helpers ───────────────────────────────────────────────────
def _mean(runs: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in runs)

def _std(runs: list[dict], key: str) -> float:
    return statistics.stdev(r[key] for r in runs) if len(runs) > 1 else 0.0

def _fmt(mean: float, std: float) -> str:
    if N_RUNS > 1:
        return f"{mean:5.1f}±{std:4.1f}"
    return f"{mean:10.1f}"

# ── Final results table ───────────────────────────────────────
out_path = "results/mobilevit_combined.txt"

COL = 12 if N_RUNS > 1 else 10
W = 15 + COL * 4 + 3
HDR = f"{'Category':<15} {'Img AUROC':>{COL}} {'Pxl AUROC':>{COL}} {'PRO':>{COL}} {'Infer(s)':>{COL}}"

lines = []
lines.append("=" * W)
lines.append(f"  PatchCore + MobileViT-S Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  Backbone: {BACKBONE}  |  Layers: {LAYERS}  |  Coreset: 25%")
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
