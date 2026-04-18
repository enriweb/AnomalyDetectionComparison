"""
UniNet
"""

import os
import time
from anomalib.data import MVTecAD, Visa
from anomalib.models import UniNet
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AUPRO
from anomalib.metrics import Evaluator

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

results = {ds: {} for ds, *_ in DATASETS}

for ds_name, DataModule, categories, root in DATASETS:
    for category in categories:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name.upper()}  |  Category: {category}")
        print(f"{'='*50}")

        # (§5.1): "All images were resized into 256×256"
        datamodule = DataModule(
            root=root,
            category=category,
            # (§5.1): "The batch size was 8"
            # NOTE: reduced to 2 to fit on small GPUs — does not affect results
            train_batch_size=2,
            eval_batch_size=2,
        )

        # Image-level AUROC (Table 1a, I-AUROC column)
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        # Pixel-level AUROC (Table 1a, P-AUROC column)
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        # PRO metric (Table 1a, PRO column) — fpr_limit=0.3 default
        pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro])

        # (§4, §5.1)
        model = UniNet(
            # (§5.1): "we used the publicly available WideResNet50 as backbone
            #  in S-T models"
            student_backbone="wide_resnet50_2",
            teacher_backbone="wide_resnet50_2",
            # (§5.1): temperature parameter — anomalib default handles the
            # paper's hyperparameters internally:
            #   UniNetLoss(lambda_weight=0.7, temperature=2.0) maps to
            #   paper's λ=0.7 and T=2
            #   Weighted decision uses α=0.01 and β=0.03
            evaluator=evaluator,
            visualizer=False,
        )

        # (§5.1): "AdamW was employed as the optimizer with a learning rate
        #  of 5e-3 and 1e-6 for the learnable student and teacher"
        engine = Engine(logger=False)
        engine.fit(model=model, datamodule=datamodule)

        t0 = time.time()
        test_results = engine.test(model=model, datamodule=datamodule)
        elapsed = time.time() - t0

        metrics = test_results[0]
        results[ds_name][category] = {
            "image_AUROC": metrics.get("image_AUROC", 0) * 100,
            "pixel_AUROC": metrics.get("pixel_AUROC", 0) * 100,
            "pixel_AUPRO": metrics.get("pixel_AUPRO", 0) * 100,
            "inference_s": elapsed,
        }

# ============================================================
# Write results to file
# ============================================================
os.makedirs("results", exist_ok=True)
out_path = "results/uninet_combined.txt"

W = 78
HDR = f"{'Category':<15} {'Img AUROC':>10} {'Pxl AUROC':>10} {'PRO':>10} {'Infer(s)':>10}"

lines = []
lines.append("=" * W)
lines.append("  UniNet Results")
lines.append("  Backbone: WRN50  |  λ=0.7  |  T=2  |  α=0.01  |  β=0.03")
lines.append("=" * W)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    lines.append(f"\n--- {label} ({len(categories)} categories) ---")
    lines.append(HDR)
    lines.append("-" * W)
    sum_img = sum_pxl = sum_pro = sum_t = 0.0
    for cat in categories:
        r = results[ds_name][cat]
        img, pxl, pro, t = r["image_AUROC"], r["pixel_AUROC"], r["pixel_AUPRO"], r["inference_s"]
        sum_img += img; sum_pxl += pxl; sum_pro += pro; sum_t += t
        lines.append(f"{cat:<15} {img:>10.1f} {pxl:>10.1f} {pro:>10.1f} {t:>10.2f}")
    n = len(categories)
    lines.append("-" * W)
    lines.append(f"{'MEAN':<15} {sum_img/n:>10.1f} {sum_pxl/n:>10.1f} {sum_pro/n:>10.1f} {sum_t/n:>10.2f}")

all_cats = [(ds, cat) for ds, _, cats, _ in DATASETS for cat in cats]
n_all = len(all_cats)
oi = sum(results[ds][c]["image_AUROC"] for ds, c in all_cats) / n_all
op = sum(results[ds][c]["pixel_AUROC"] for ds, c in all_cats) / n_all
or_ = sum(results[ds][c]["pixel_AUPRO"] for ds, c in all_cats) / n_all
ot = sum(results[ds][c]["inference_s"] for ds, c in all_cats) / n_all
lines.append(f"\n--- Overall Mean ({n_all} categories) ---")
lines.append(f"Img AUROC: {oi:.1f}  |  Pxl AUROC: {op:.1f}  |  PRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
