"""
CFA
"""

import os
import time
from anomalib.data import MVTecAD, Visa
from anomalib.models import Cfa
from anomalib.engine import Engine
from anomalib.metrics import AUROC
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

        # (§4.1): "each sample of the dataset is resized into 256×256,
        #     and is center-cropped into 224×224"
        datamodule = DataModule(
            root=root,
            category=category,
            # (§4.1): "The batch size was set to 4"
            train_batch_size=4,
            eval_batch_size=16,
        )

        # Image-level AUROC (Table 5, I-AUROC column)
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        # Pixel-level AUROC (Table 5, P-AUROC column)
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc])

        # (§3, §4.1)
        model = Cfa(
            # (§4.1): "All CNNs used in the experiments were pretrained with ImageNet"
            # (Table 5): WRN50-2 backbone
            backbone="wide_resnet50_2",
            # (§4.1, Table 6): "γ_c = 1, γ_d = 1" — no compression of memory bank
            gamma_c=1,
            gamma_d=1,
            # (§3.1, Eq.1, §4.1): "K and J were equally set to 3"
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            # (§3.1, Eq.1, §4.1): "r [...] was set to 1e-5"
            radius=1e-5,
            # (§4.1): resize 256×256, center-crop 224×224
            pre_processor=Cfa.configure_pre_processor(
                image_size=(256, 256),
                center_crop_size=(224, 224),
            ),
            evaluator=evaluator,
            visualizer=False,
        )

        # (§4.1): "Patch descriptor was trained for 30 epochs"
        engine = Engine(max_epochs=30, logger=False)
        engine.fit(model=model, datamodule=datamodule)

        t0 = time.time()
        test_results = engine.test(model=model, datamodule=datamodule)
        elapsed = time.time() - t0

        metrics = test_results[0]
        results[ds_name][category] = {
            "image_AUROC": metrics.get("image_AUROC", 0) * 100,
            "pixel_AUROC": metrics.get("pixel_AUROC", 0) * 100,
            "inference_s": elapsed,
        }

# ============================================================
# Write results to file
# ============================================================
os.makedirs("results", exist_ok=True)
out_path = "results/cfa_combined.txt"

W = 60
HDR = f"{'Category':<15} {'Img AUROC':>10} {'Pxl AUROC':>10} {'Infer(s)':>10}"

lines = []
lines.append("=" * W)
lines.append("  CFA Results")
lines.append("  Backbone: WRN50-2  |  K=J=3  |  r=1e-5  |  30 epochs")
lines.append("=" * W)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    lines.append(f"\n--- {label} ({len(categories)} categories) ---")
    lines.append(HDR)
    lines.append("-" * W)
    sum_img = sum_pxl = sum_t = 0.0
    for cat in categories:
        r = results[ds_name][cat]
        img, pxl, t = r["image_AUROC"], r["pixel_AUROC"], r["inference_s"]
        sum_img += img; sum_pxl += pxl; sum_t += t
        lines.append(f"{cat:<15} {img:>10.1f} {pxl:>10.1f} {t:>10.2f}")
    n = len(categories)
    lines.append("-" * W)
    lines.append(f"{'MEAN':<15} {sum_img/n:>10.1f} {sum_pxl/n:>10.1f} {sum_t/n:>10.2f}")

all_cats = [(ds, cat) for ds, _, cats, _ in DATASETS for cat in cats]
n_all = len(all_cats)
oi = sum(results[ds][c]["image_AUROC"] for ds, c in all_cats) / n_all
op = sum(results[ds][c]["pixel_AUROC"] for ds, c in all_cats) / n_all
ot = sum(results[ds][c]["inference_s"] for ds, c in all_cats) / n_all
lines.append(f"\n--- Overall Mean ({n_all} categories) ---")
lines.append(f"Img AUROC: {oi:.1f}  |  Pxl AUROC: {op:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
