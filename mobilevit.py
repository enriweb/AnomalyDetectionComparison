"""
PatchCore with MobileViT-S backbone
"""

import os
import time
import timm
from anomalib.data import MVTecAD, Visa
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AUPRO
from anomalib.metrics import Evaluator

# ============================================================
#  BACKBONE CONFIGURATION
# ============================================================
# MobileViT-S from timm (ICLR 2022, §4.1: "5.6M params, 78.4% top-1")
BACKBONE = "mobilevit_s"

# --- Discover available feature extraction layers ---
# (run once to find the right layer names for your timm version)
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

results = {ds: {} for ds, *_ in DATASETS}

for ds_name, DataModule, categories, root in DATASETS:
    for category in categories:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name.upper()}  |  Category: {category}")
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
            # Mid-level feature layers (discovered above)
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
out_path = "results/mobilevit_combined.txt"

W = 78
HDR = f"{'Category':<15} {'Img AUROC':>10} {'Pxl AUROC':>10} {'PRO':>10} {'Infer(s)':>10}"

lines = []
lines.append("=" * W)
lines.append(f"  PatchCore + MobileViT-S Results")
lines.append(f"  Backbone: {BACKBONE}  |  Layers: {LAYERS}")
lines.append(f"  Coreset: 25%  |  Neighbors: 9")
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
