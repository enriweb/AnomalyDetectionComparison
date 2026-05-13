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

Metrics recorded per run:
    Accuracy  — image AUROC, pixel AUROC, pixel AUPRO,
                image F1Max (optimal threshold), pixel F1Max
    Efficiency — n_params (CNN + ViT combined), FLOPs (CNN branch),
                 inference time GPU (backbone),
                 peak GPU MB, GPU throughput, CPU latency

Progress is checkpointed to results/hybridcore_progress.json after every
completed run. On normal exit the checkpoint is deleted. On crash / OOM
the checkpoint survives and the next invocation auto-resumes.
"""

import copy
import gc
import json
import logging
import os
import sys
import time
import warnings

# ── Silence warnings and noisy loggers ───────────────────────
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries...")
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from anomalib.data import InferenceBatch, MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, F1Max, Evaluator
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.flop_counter import FlopCounterMode
import pandas as pd

# ── Config ────────────────────────────────────────────────────
N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1

torch.set_float32_matmul_precision('high')

# ── GPU cleanup helper ────────────────────────────────────────
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Inline efficiency helpers ─────────────────────────────────
@torch.no_grad()
def _gpu_throughput(model: nn.Module,
                    input_shape: tuple = (1, 3, 224, 224),
                    batch: int = 8,
                    warmup: int = 10,
                    iters: int = 50) -> float | None:
    """Images per second on CUDA. Returns None if no GPU."""
    if not torch.cuda.is_available():
        return None
    _model = model.cuda().eval()
    _, c, h, w = input_shape
    x = torch.randn(batch, c, h, w, device="cuda")
    for _ in range(warmup):
        _model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _model(x)
    torch.cuda.synchronize()
    return (batch * iters) / (time.perf_counter() - t0)


@torch.no_grad()
def _cpu_latency(model: nn.Module,
                 input_shape: tuple = (1, 3, 224, 224),
                 warmup: int = 5,
                 iters: int = 20) -> float:
    """Mean ms per single-image forward on CPU."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*input_shape)
    try:
        for _ in range(warmup):
            cpu_model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_model(x)
        return (time.perf_counter() - t0) / iters * 1000.0
    finally:
        torch.set_num_threads(prev_threads)
        del cpu_model


@torch.no_grad()
def _peak_gpu_mem(model: nn.Module,
                  input_shape: tuple = (1, 3, 224, 224)) -> int | None:
    """Peak GPU allocated bytes for a single forward pass. None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    _model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(*input_shape, device="cuda")
    _model(x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


# ── Single-line progress bar ──────────────────────────────────
class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm(); bar.leave = False; return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm(); bar.leave = False; return bar
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm(); bar.leave = False; return bar


# ── Model ─────────────────────────────────────────────────────

class HybridCoreModel(PatchcoreModel):
    """PatchcoreModel extended with a frozen DeiT-Tiny ViT branch.

    CNN features (WRN50 layer2+layer3) and ViT patch tokens (DeiT-Tiny) are
    spatially aligned and concatenated before the coreset memory bank.
    Embedding dim: 1536 (CNN) + 192 (ViT) = 1728.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vit = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
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
            backbone="wide_resnet50_2",
            pre_trained=True,
            layers=["layer2", "layer3"],
            num_neighbors=9,
        )


# ── Dataset config ────────────────────────────────────────────
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

total_categories = sum(len(c) for _, _, c, _ in DATASETS)

os.makedirs("results", exist_ok=True)
PROGRESS_FILE = "results/hybridcore_progress.json"

# ── Startup banner ────────────────────────────────────────────
print("=" * 60)
print("  HybridCore  |  CNN: wide_resnet50_2 [layer2, layer3]  |  ViT: deit_tiny_patch16_224")
print("  Coreset: 25%  |  Neighbors: 9  |  Emb: 1536+192=1728")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device            : {gpu_info}")
print("=" * 60)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        results = json.load(f)
    print(f"\n[resume] Found checkpoint — resuming from {PROGRESS_FILE}")
else:
    results = {ds: {} for ds, *_ in DATASETS}
    print("\n[start] No checkpoint found — starting fresh.")

# ── OOM skip tracking ─────────────────────────────────────────
oom_skips: list[dict] = []   # {"ds": ..., "category": ..., "run": ...}

# ── Experiment loop (run-outer) ───────────────────────────────
for run in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"  RUN {run+1}/{N_RUNS}")
    print(f"{'='*60}")

    for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
        print(f"\n{'─'*60}")
        print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
        print(f"{'─'*60}")

        for cat_idx, category in enumerate(categories, 1):
            # Skip if this run index already completed for this category
            if len(results.get(ds_name, {}).get(category, [])) > run:
                print(f"  [skip] {category} ({cat_idx}/{len(categories)}) — run {run+1} already done")
                continue

            if ds_name not in results:
                results[ds_name] = {}
            if category not in results[ds_name]:
                results[ds_name][category] = []

            print(f"\n  [{cat_idx}/{len(categories)}] {category}  —  Run {run+1}/{N_RUNS}")

            model = engine = datamodule = evaluator = None
            image_auroc = pixel_auroc = pixel_pro = None
            image_f1max = pixel_f1max = test_results = metrics = None
            try:
                print(f"  → Loading datamodule...")
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=32,
                    eval_batch_size=32,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                pixel_pro   = AUPRO(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro,
                                                    image_f1max, pixel_f1max])

                print(f"  → Building model (WRN50 + DeiT-Tiny)...")
                model = HybridCore(
                    backbone="wide_resnet50_2",
                    layers=["layer2", "layer3"],
                    pre_trained=True,
                    coreset_sampling_ratio=0.25,
                    num_neighbors=9,
                    pre_processor=Patchcore.configure_pre_processor(
                        image_size=(256, 256),
                        center_crop_size=(224, 224),
                    ),
                    evaluator=evaluator,
                    visualizer=False,
                )

                # n_params: CNN backbone + ViT branch combined
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  → Parameters (CNN+ViT): {n_params:,}")

                print(f"  → Training (building coreset)...")
                engine = Engine(max_epochs=1, devices="auto", strategy="auto", logger=False, callbacks=[CompactBar()])
                engine.fit(model=model, datamodule=datamodule)
                print(f"  → Coreset built.")

                # ── FLOPs: CNN backbone forward pass on one image ──────────────
                flops_M = None
                try:
                    device = next(model.parameters()).device
                    _dummy = torch.randn(1, 3, 224, 224, device=device)
                    with FlopCounterMode(display=False) as fcm:
                        _ = model.model.feature_extractor(_dummy)
                    flops_M = round(fcm.get_total_flops() / 1e6, 1)
                    print(f"  → CNN Backbone FLOPs: {flops_M:.1f} M")
                    del _dummy
                except Exception:
                    pass

                # ── Throughput, CPU latency, single-forward peak mem ───────────
                extractor = model.model.feature_extractor
                input_shape = (1, 3, 224, 224)

                try:
                    gpu_tput = _gpu_throughput(extractor, input_shape=input_shape, batch=8)
                    print(f"  → GPU throughput: {gpu_tput:.1f} img/s" if gpu_tput else "")
                except Exception:
                    gpu_tput = None

                try:
                    cpu_lat = _cpu_latency(extractor, input_shape=input_shape)
                    print(f"  → CPU latency: {cpu_lat:.1f} ms" if cpu_lat else "")
                except Exception:
                    cpu_lat = None

                try:
                    peak_mem_1fwd = _peak_gpu_mem(extractor, input_shape=input_shape)
                    print(f"  → Peak mem (1 fwd): {peak_mem_1fwd/1e6:.1f} MB" if peak_mem_1fwd else "")
                except Exception:
                    peak_mem_1fwd = None

                # ── GPU inference on test set ──────────────────────────────────
                print(f"  → Running inference on test set (GPU)...")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                test_results = engine.test(model=model, datamodule=datamodule)
                elapsed_gpu = time.time() - t0

                peak_gpu_mb = 0.0
                if torch.cuda.is_available():
                    peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

                metrics = test_results[0]
                img_auc  = metrics.get("image_AUROC", 0) * 100
                pxl_auc  = metrics.get("pixel_AUROC", 0) * 100
                pxl_pro  = metrics.get("pixel_AUPRO", 0) * 100
                img_f1   = (metrics["image_F1Max"] * 100) if metrics.get("image_F1Max") is not None else None
                pxl_f1   = (metrics["pixel_F1Max"] * 100) if metrics.get("pixel_F1Max") is not None else None

                print(f"  → Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                      f" | AUPRO {pxl_pro:.1f}%"
                      + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else ""))
                print(f"  → GPU inf: {elapsed_gpu:.3f}s  |  Peak GPU: {peak_gpu_mb:.1f} MB")

                run_record = {
                    "image_AUROC":     img_auc,
                    "pixel_AUROC":     pxl_auc,
                    "pixel_AUPRO":     pxl_pro,
                    "image_F1Max":     img_f1,
                    "pixel_F1Max":     pxl_f1,
                    "n_params":        n_params,
                    "flops_M":         flops_M,
                    "inference_gpu_s": elapsed_gpu,
                    "peak_gpu_mb":     peak_gpu_mb,
                    "gpu_throughput":  gpu_tput,
                    "cpu_latency_ms":  cpu_lat,
                    "peak_gpu_mem_b":  peak_mem_1fwd,
                }
                results[ds_name][category].append(run_record)

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → Checkpoint saved.")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_skips.append({"ds": ds_name, "category": category, "run": run + 1})
                    print(f"\n  [OOM] CUDA out of memory — {ds_name}/{category} run {run+1}. "
                          f"Skipping. (total OOM skips so far: {len(oom_skips)})")
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max
                del test_results, metrics
                free_gpu()

# ── Post-loop: clean up checkpoint ───────────────────────────
print(f"\n{'='*60}")
print("  All runs complete.")
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"  Checkpoint deleted ({PROGRESS_FILE}).")
    print("  Next invocation will start fresh.")
print(f"{'='*60}\n")

# ── Build pandas DataFrame & unified summary ──────────────────
rows = []
for ds_name, _, categories, _ in DATASETS:
    for cat in categories:
        runs_list = results.get(ds_name, {}).get(cat, [])
        for run_idx, rec in enumerate(runs_list):
            rows.append({
                "dataset": ds_name, "category": cat, "run": run_idx + 1,
                "img_auroc": rec.get("image_AUROC"), "pxl_auroc": rec.get("pixel_AUROC"),
                "aupro": rec.get("pixel_AUPRO"), "img_f1max": rec.get("image_F1Max"),
                "pxl_f1max": rec.get("pixel_F1Max"),
                "params": rec.get("n_params"), "flops_M": rec.get("flops_M"),
                "inf_s": rec.get("inference_gpu_s"), "peak_gpu_mb": rec.get("peak_gpu_mb"),
                "gpu_tput": rec.get("gpu_throughput"), "cpu_lat_ms": rec.get("cpu_latency_ms"),
                "peak_mem_b": rec.get("peak_gpu_mem_b"),
            })

if not rows:
    print("No results to display.")
    sys.exit(0)

df = pd.DataFrame(rows)
csv_path = "results/hybridcore_results.csv"
df.to_csv(csv_path, index=False)

metric_cols = ["img_auroc", "pxl_auroc", "aupro", "img_f1max", "pxl_f1max",
               "params", "flops_M", "inf_s", "peak_gpu_mb",
               "gpu_tput", "cpu_lat_ms", "peak_mem_b"]

if N_RUNS > 1:
    cat_agg = df.groupby(["dataset", "category"])[metric_cols].agg(["mean", "std"])
    cat_agg.columns = [f"{c[0]}" if c[1] == "mean" else f"{c[0]}_std" for c in cat_agg.columns]
    std_cols = [f"{c}_std" for c in metric_cols]
else:
    cat_agg = df.groupby(["dataset", "category"])[metric_cols].mean()
    std_cols = []
display_cols = metric_cols + std_cols

ds_rows = []
for ds_name, _, _, _ in DATASETS:
    ds_df = df[df["dataset"] == ds_name]
    if ds_df.empty:
        continue
    label = "MVTecAD" if ds_name == "mvtec" else "VisA"
    row = {"dataset": ds_name, "category": f"~ {label}"}
    row.update(ds_df[metric_cols].mean().to_dict())
    ds_rows.append(row)
ds_summary = pd.DataFrame(ds_rows).set_index(["dataset", "category"]) if ds_rows else None

ov = df[metric_cols].mean().to_dict()
ov.update({"dataset": "", "category": "~ OVERALL"})
overall_df = pd.DataFrame([ov]).set_index(["dataset", "category"])

def _align(frame, cols):
    if frame is None: return None
    frame = frame.copy()
    for c in cols:
        if c not in frame.columns: frame[c] = float("nan")
    return frame[cols]

pieces = [_align(cat_agg, display_cols)]
if ds_summary is not None: pieces.append(_align(ds_summary, display_cols))
pieces.append(_align(overall_df, display_cols))
summary = pd.concat(pieces)

print(f"\n{'=' * 140}")
print(f"  HybridCore  (N={N_RUNS}, CNN: wide_resnet50_2 [layer2, layer3], ViT: deit_tiny_patch16_224, Coreset: 25%)")
print(f"  Raw CSV: {csv_path}")
print(f"{'=' * 140}")
print(summary.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "—"))

if oom_skips:
    print(f"\n{'=' * 80}\n  OOM Skips ({len(oom_skips)} total)\n{'=' * 80}")
    for s in oom_skips: print(f"  • {s['ds']}/{s['category']}  run {s['run']}")
else:
    print("\n  No OOM skips recorded.")
print("=" * 80)
