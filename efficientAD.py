"""
EfficientAD

Usage:
    python efficientAD.py          # 1 run (default)
    python efficientAD.py 3        # 3 runs, results averaged
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
from anomalib.models import EfficientAd

# ── How many runs to average ──────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# ============================================================
#  SELECT VARIANT HERE
# ============================================================
# (§4): "EfficientAD-S uses the architecture displayed in Figure 2"
# (§4): "For EfficientAD-M, we double the number of kernels in the
#         hidden convolutional layers [...] and insert a 1×1 convolution"
VARIANT = "s"  # <-- change to "m" for EfficientAD-M
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
PROGRESS_FILE = f"results/efficientad_{VARIANT}_progress.json"

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
            print(f"  Dataset: {ds_name.upper()}  |  Category: {category}  |  Variant: EfficientAD-{VARIANT.upper()}  |  Run {run+1}/{N_RUNS}")
            print(f"{'='*50}")

            # (§3.1): image size 256×256
            # EfficientAD requires train_batch_size=1 (anomalib constraint)
            datamodule = DataModule(
                root=root,
                category=category,
                train_batch_size=1,
                eval_batch_size=32,
            )

            # Image-level AU-ROC (Table 2)
            image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
            # Pixel-level AU-PRO (Table 1) — fpr_limit=0.3, matching §4
            pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            evaluator = Evaluator(test_metrics=[image_auroc, pixel_pro])

            model = EfficientAd(
                # (§3.2): "we sample a random image P from the pretraining dataset"
                imagenet_dir="./datasets/imagenette",
                # (Fig.2): PDN output channels = 384
                teacher_out_channels=384,
                # (supplementary): learning rate = 1e-4
                lr=1e-4,
                # (supplementary): weight decay = 1e-5
                weight_decay=1e-5,
                # (§3.1): PDN has no padding — receptive field exactly 33×33
                padding=False,
                pad_maps=True,
                evaluator=evaluator,
                visualizer=False,
            )

            # (§5, supplementary): training takes ~20 min per scenario
            engine = Engine(max_steps=70000, logger=False)
            engine.fit(model=model, datamodule=datamodule)

            t0 = time.time()
            test_results = engine.test(model=model, datamodule=datamodule)
            elapsed = time.time() - t0

            metrics = test_results[0]
            results[ds_name][category].append({
                "image_AUROC": metrics.get("image_AUROC", 0) * 100,
                "pixel_AUPRO": metrics.get("pixel_AUPRO", 0) * 100,
                "inference_s": elapsed,
            })

            with open(PROGRESS_FILE, "w") as f:
                json.dump(results, f, indent=2)

            del model, engine, datamodule, evaluator
            del image_auroc, pixel_pro, test_results, metrics
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
variant_label = VARIANT.upper()
out_path = f"results/efficientad_{VARIANT}_combined.txt"

COL = 12 if N_RUNS > 1 else 10
W = 15 + COL * 3 + 3
HDR = f"{'Category':<15} {'Img AUROC':>{COL}} {'AU-PRO':>{COL}} {'Infer(s)':>{COL}}"

lines = []
lines.append("=" * W)
lines.append(f"  EfficientAD-{variant_label} Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append("  PDN teacher_out=384  |  image_size=256×256")
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
        m_pro = _mean(runs, "pixel_AUPRO");  s_pro = _std(runs, "pixel_AUPRO")
        m_t   = _mean(runs, "inference_s");  s_t   = _std(runs, "inference_s")
        cat_means[cat] = {"img": m_img, "pro": m_pro, "t": m_t}
        lines.append(
            f"{cat:<15}"
            f" {_fmt(m_img, s_img):>{COL}}"
            f" {_fmt(m_pro, s_pro):>{COL}}"
            f" {_fmt(m_t,   s_t  ):>{COL}}"
        )

    if cat_means:
        avg = {k: statistics.mean(v[k] for v in cat_means.values()) for k in ("img", "pro", "t")}
        lines.append("-" * W)
        lines.append(
            f"{'MEAN':<15}"
            f" {avg['img']:>{COL}.1f}"
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
    or_ = statistics.mean(_mean(results[ds][c], "pixel_AUPRO") for ds, c in all_cats)
    ot  = statistics.mean(_mean(results[ds][c], "inference_s") for ds, c in all_cats)
    lines.append(f"\n--- Overall Mean ({n_all} categories, N={N_RUNS} runs each) ---")
    lines.append(f"Img AUROC: {oi:.1f}  |  AU-PRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
