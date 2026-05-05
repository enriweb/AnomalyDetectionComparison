"""
HybridCore: Dual-Encoder Coreset Anomaly Detection

Novel architecture combining CNN pyramid features (WideResNet-50) with ViT patch
tokens (DeiT-Tiny) in a single unified coreset memory bank. No gradient training —
both encoders are frozen pretrained models.

Novelty: First gradient-free dual-encoder coreset model fusing independent CNN
         pyramid features and ViT patch tokens in a joint memory bank. Distinct
         from MobileViT (single fused backbone) and PatchCore (CNN-only).

Usage:
    python hybridcore.py          # 1 run (default)
    python hybridcore.py 3        # 3 runs, results averaged
"""

import gc
import json
import os
import statistics
import sys
import time

import timm
import torch
import torch.nn.functional as F
from anomalib.data import InferenceBatch, MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

# ── How many runs to average ──────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1

CNN_BACKBONE = "wide_resnet50_2"
CNN_LAYERS   = ["layer2", "layer3"]
VIT_NAME     = "deit_tiny_patch16_224"
CORESET_RATIO = 0.25
NUM_NEIGHBORS = 9

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
PROGRESS_FILE = "results/hybridcore_progress.json"

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"Resuming from {PROGRESS_FILE}")
else:
    results = {ds: {} for ds, *_ in DATASETS}


# ── Model ─────────────────────────────────────────────────────

class HybridCoreModel(PatchcoreModel):
    """PatchcoreModel extended with a frozen DeiT-Tiny ViT branch.

    CNN features (WRN50 layer2+layer3) and ViT patch tokens (DeiT-Tiny) are
    spatially aligned and concatenated before the coreset memory bank.
    Embedding dim: 1536 (CNN) + 192 (ViT) = 1728.
    """

    def __init__(self, vit_name: str = VIT_NAME, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vit = timm.create_model(vit_name, pretrained=True).eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        output_size = input_tensor.shape[-2:]

        # CNN branch — inherited feature extraction + pooling + multi-scale concat
        with torch.no_grad():
            cnn_features = self.feature_extractor(input_tensor)
        cnn_features = {layer: self.feature_pooler(feat) for layer, feat in cnn_features.items()}
        cnn_embedding = self.generate_embedding(cnn_features)  # (B, 1536, H, W)

        # ViT branch — frozen DeiT-Tiny patch tokens upsampled to CNN spatial dims
        with torch.no_grad():
            vit_out = self.vit.forward_features(input_tensor)      # (B, N_tokens, D)
        n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        patch_tokens = vit_out[:, n_prefix:, :]                    # (B, N_patches, 192)
        n_side = int(patch_tokens.shape[1] ** 0.5)
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], n_side, n_side, -1)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)            # (B, 192, n, n)
        patch_tokens = F.interpolate(
            patch_tokens, size=cnn_embedding.shape[-2:], mode="bilinear", align_corners=False
        )                                                           # (B, 192, H, W)

        # Fuse: (B, 1728, H, W)
        embedding = torch.cat([cnn_embedding, patch_tokens], dim=1)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)              # (B*H*W, 1728)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores."
            raise ValueError(msg)

        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


class HybridCore(Patchcore):
    """Patchcore Lightning module using dual-encoder HybridCoreModel."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = HybridCoreModel(
            backbone=CNN_BACKBONE,
            pre_trained=True,
            layers=CNN_LAYERS,
            num_neighbors=NUM_NEIGHBORS,
        )


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
                train_batch_size=32,
                eval_batch_size=32,
            )

            image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
            pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            pixel_pro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
            evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro])

            model = HybridCore(
                backbone=CNN_BACKBONE,
                layers=CNN_LAYERS,
                pre_trained=True,
                coreset_sampling_ratio=CORESET_RATIO,
                num_neighbors=NUM_NEIGHBORS,
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
out_path = "results/hybridcore_combined.txt"

COL = 12 if N_RUNS > 1 else 10
W = 15 + COL * 4 + 3
HDR = f"{'Category':<15} {'Img AUROC':>{COL}} {'Pxl AUROC':>{COL}} {'PRO':>{COL}} {'Infer(s)':>{COL}}"

lines = []
lines.append("=" * W)
lines.append(f"  HybridCore Results  (N={N_RUNS} run{'s' if N_RUNS > 1 else ''})")
lines.append(f"  CNN: {CNN_BACKBONE} {CNN_LAYERS}  |  ViT: {VIT_NAME}  |  Coreset: {CORESET_RATIO*100:.0f}%")
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
