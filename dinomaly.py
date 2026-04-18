"""
Dinomaly

NOTE: Paper trains ONE model for all 15 categories jointly (MUAD).
      This script trains per-category for consistency with other scripts.

WARNING: The default ViT-Base/14 + 392×392 input requires ~6+ GB VRAM.
         If you OOM on a small GPU, change ENCODER below to
         "dinov2reg_vit_small_14" (see comment).

"""

import os
import time
from anomalib.data import MVTecAD, Visa
from anomalib.models import Dinomaly
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AUPRO
from anomalib.metrics import Evaluator

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

results = {ds: {} for ds, *_ in DATASETS}

for ds_name, DataModule, categories, root in DATASETS:
    for category in categories:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name.upper()}  |  Category: {category}")
        print(f"{'='*50}")

        datamodule = DataModule(
            root=root,
            category=category,
            # (§4.1): "batch size of 16" — reduced for small GPUs
            train_batch_size=2,
            eval_batch_size=2,
        )

        # Image-level AUROC (Table 1)
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        # Pixel-level AUROC (Table 1)
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        # AUPRO (Table 1) — fpr_limit=0.3 default
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
            # anomalib uses 0-indexed: [2,3,4,5,6,7,8,9] — this is the default
            target_layers=None,
            # (§3.3): "Loose constraint with 2 groups" — [[0,1,2,3],[4,5,6,7]]
            fuse_layer_encoder=None,
            fuse_layer_decoder=None,
            # (§3.1): keep class token
            remove_class_token=False,
            # (§4.1): "input image is first resized to 448² then center-cropped
            #  to 392²" — this is the default pre_processor for Dinomaly
            evaluator=evaluator,
            visualizer=False,
        )

        # (§4.1): "trained for 10,000 iterations (steps) on MVTec-AD"
        # (§4.1): "StableAdamW optimizer with lr=2e-3, β=(0.9,0.999), wd=1e-4"
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
out_path = "results/dinomaly_combined.txt"

W = 78
HDR = f"{'Category':<15} {'Img AUROC':>10} {'Pxl AUROC':>10} {'AUPRO':>10} {'Infer(s)':>10}"

lines = []
lines.append("=" * W)
lines.append(f"  Dinomaly Results (per-category)")
lines.append(f"  Encoder: {ENCODER}  |  dropout=0.2  |  2 groups")
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
lines.append(f"Img AUROC: {oi:.1f}  |  Pxl AUROC: {op:.1f}  |  AUPRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
