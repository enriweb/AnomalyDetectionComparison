"""
EfficientAD
"""


import os
import time
from anomalib.data import MVTecAD, Visa
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AUPRO
from anomalib.metrics import Evaluator

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

results = {ds: {} for ds, *_ in DATASETS}

for ds_name, DataModule, categories, root in DATASETS:
    for category in categories:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name.upper()}  |  Category: {category}  |  Variant: EfficientAD-{VARIANT.upper()}")
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
            # (§3.2): "we sample a random image P from the pretraining dataset,
            #  in our case ImageNet" — anomalib uses Imagenette (subset)
            imagenet_dir="./datasets/imagenette",
            # (Fig.2): PDN output channels = 384
            teacher_out_channels=384,
            # (§4, Fig.2): "s" = base PDN; "m" = doubled hidden kernels + 1×1 convs
            # model_size=VARIANT,
            # (supplementary): learning rate = 1e-4
            lr=1e-4,
            # (supplementary): weight decay = 1e-5
            weight_decay=1e-5,
            # (§3.1): PDN has no padding by design — receptive field is exactly 33×33
            padding=False,
            # Pad output anomaly maps to match input size when padding=False
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
        results[ds_name][category] = {
            "image_AUROC": metrics.get("image_AUROC", 0) * 100,
            "pixel_AUPRO": metrics.get("pixel_AUPRO", 0) * 100,
            "inference_s": elapsed,
        }

# ============================================================
# Write results to file
# ============================================================
os.makedirs("results", exist_ok=True)
variant_label = VARIANT.upper()
out_path = f"results/efficientad_{VARIANT}_combined.txt"

W = 60
HDR = f"{'Category':<15} {'Img AUROC':>10} {'AU-PRO':>10} {'Infer(s)':>10}"

lines = []
lines.append("=" * W)
lines.append(f"  EfficientAD-{variant_label} Results")
lines.append(f"  PDN teacher_out=384  |  image_size=256×256")
lines.append("=" * W)

for ds_name, _, categories, _ in DATASETS:
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    lines.append(f"\n--- {label} ({len(categories)} categories) ---")
    lines.append(HDR)
    lines.append("-" * W)
    sum_img = sum_pro = sum_t = 0.0
    for cat in categories:
        r = results[ds_name][cat]
        img, pro, t = r["image_AUROC"], r["pixel_AUPRO"], r["inference_s"]
        sum_img += img; sum_pro += pro; sum_t += t
        lines.append(f"{cat:<15} {img:>10.1f} {pro:>10.1f} {t:>10.2f}")
    n = len(categories)
    lines.append("-" * W)
    lines.append(f"{'MEAN':<15} {sum_img/n:>10.1f} {sum_pro/n:>10.1f} {sum_t/n:>10.2f}")

all_cats = [(ds, cat) for ds, _, cats, _ in DATASETS for cat in cats]
n_all = len(all_cats)
oi = sum(results[ds][c]["image_AUROC"] for ds, c in all_cats) / n_all
or_ = sum(results[ds][c]["pixel_AUPRO"] for ds, c in all_cats) / n_all
ot = sum(results[ds][c]["inference_s"] for ds, c in all_cats) / n_all
lines.append(f"\n--- Overall Mean ({n_all} categories) ---")
lines.append(f"Img AUROC: {oi:.1f}  |  AU-PRO: {or_:.1f}  |  Infer(s): {ot:.2f}")
lines.append("=" * W)

output = "\n".join(lines)
print("\n" + output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {out_path}")
