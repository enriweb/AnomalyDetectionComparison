"""
Dinomaly

Usage:
    python dinomaly.py          # 1 run (default)
    python dinomaly.py 3        # 3 runs, results averaged

NOTE: Paper trains ONE model for all 15 categories jointly (MUAD).
      This script trains per-category for consistency with other scripts.

WARNING: The default ViT-Base/14 + 392×392 input requires ~6+ GB VRAM.
         If you OOM on a small GPU, change ENCODER below to
         "dinov2reg_vit_small_14" (see comment).
"""

import gc
import json
import os
import statistics
import sys
import time

import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Dinomaly

# ── How many runs to average ──────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# ============================================================
#  ENCODER SELECTION
# ============================================================
# (§3.1, §4.1): "ViT-Base/14 pretrained by DINOv2-Register" (default)
# If you get OOM, switch to ViT-Small:
#   ENCODER = "dinov2reg_vit_small_14"
ENCODER = "dinov2reg_vit_base_14"
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
PROGRESS_FILE = "results/dinomaly_progress.json"

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

            datamodule = DataModule(
                root=root,
                category=category,
                # (§4.1): "batch size of 16" — reduced for small GPUs
                train_batch_size=2,
                eval_batch_size=2,
            )

            image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
            pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro])

            # (§3, §4.1)
            model = Dinomaly(
                # (§3.1, §4.1): "ViT-Base/14 pretrained by DINOv2-Register [7]"
                encoder_name=ENCODER,
                # (§3.2): "The drop rate of Noisy Bottleneck is 0.2 by default"
                bottleneck_dropout=0.2,
                # (§3.1): "The decoder [...] consisting of 8 Transformer layers"
                decoder_depth=8,
                # (§3.1): middle 8 layers M={3,...,10} for ViT-Base (12 layers)
                target_layers=None,
                # (§3.3): "Loose constraint with 2 groups"
                fuse_layer_encoder=None,
                fuse_layer_decoder=None,
                # (§3.1): keep class token
                remove_class_token=False,
                evaluator=evaluator,
                visualizer=False,
            )

            # (§4.1): "trained for 10,000 iterations (steps) on MVTec-AD"
            engine = Engine(logger=False)
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
out_path = "results/dinomaly_combined.txt"

COL = 12 if N_RUNS > 1 else 10
W = 15 + COL * 4 + 3
HDR = f"{'Category':<15} {'Img AUROC':>{COL}} {'Pxl AUROC':>{COL}} {'AUPRO':>{COL}} {'Infer(s)':>{COL}}"

lines = []
lines.append("=" * W)
lines.append(f"  Dinomaly Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  Encoder: {ENCODER}  |  dropout=0.2  |  2 groups")
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
    lines.append(f"Img AUROC: {oi:.1f}  |  Pxl AUROC: {op:.1f}  |  AUPRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
