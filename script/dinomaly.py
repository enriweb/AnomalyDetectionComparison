"""
Dinomaly (Guo et al., CVPR 2025 — "Dinomaly: The Less Is More Philosophy in
Multi-Class Unsupervised Anomaly Detection")

Usage:
    python dinomaly.py          # 1 run (default)
    python dinomaly.py 3        # 3 runs, results averaged

Paper config (Sec. 4.1, Tab. 1):
    Encoder      : DINOv2-R ViT-B/14 (frozen)
    Middle layers: 8 of 12 (indices 3..10 → 0-based [2..9])
    Bottleneck   : MLP, dropout 0.2 (noisy bottleneck)
    Decoder      : 8 ViT layers, linear attention
    Loose recon  : 2 groups [[0..3], [4..7]] (low-/high-semantic)
    Loss         : hard-mining global cosine, ratio 0..0.9 over 1000 steps
    Optimizer    : StableAdamW, lr=2e-3, wd=1e-4, warmup 100, cosine
    Resolution   : resize 448 → center-crop 392
    Batch / steps: 16 / 5000
    Eval         : Gaussian σ=4 kernel=5, max-ratio 0.01

NOTE: paper trains one model jointly for all categories (MUAD).
      This script trains per-category to match other thesis runners.

Metrics recorded per run:
    Accuracy  — image AUROC, pixel AUROC, pixel AUPRO,
                image F1Max (optimal threshold), pixel F1Max
    Efficiency — n_params, FLOPs (encoder), peak GPU MB,
                 deploy latency (raw frame -> anomaly decision, batch=1):
                 total mean/p95/p99 + per-stage pre/fwd/post breakdown

Progress is checkpointed to results/dinomaly_results.csv after every completed
run. On normal exit the checkpoint is deleted. On crash / OOM the checkpoint
survives and the next invocation auto-resumes.
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import time
import warnings

# ── Silence warnings and noisy loggers ───────────────────────
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries...")
import torch
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, F1Max, Evaluator
from anomalib.models import Dinomaly
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.flop_counter import FlopCounterMode
import pandas as pd

torch.set_float32_matmul_precision('high')

# ── Config ────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("n_runs", nargs="?", type=int, default=1)
_parser.add_argument("--instance", type=int, default=0, metavar="I",
                     help="0-based instance index (for parallel runs)")
_parser.add_argument("--total", type=int, default=1, metavar="M",
                     help="total number of parallel instances")
_args = _parser.parse_args()
N_RUNS:      int = _args.n_runs
INSTANCE:    int = _args.instance
N_INSTANCES: int = _args.total
MAX_STEPS:  int = 5000   # Guo et al. 2025, Tab. 1 (training iterations)
BATCH_SIZE: int = 16     # Guo et al. 2025, Sec. 4.1
IMAGE_SIZE: int = 448    # resize side
CROP_SIZE:  int = 392    # center crop (= 28*14, ViT-14 patch grid 28x28)

# ── GPU helpers ───────────────────────────────────────────────
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ── Inline efficiency helpers ─────────────────────────────────
@torch.no_grad()
def _deploy_latency(model, raw_image, device="cuda", warmup=20, iters=200):
    """Per-image deployment latency: raw camera frame -> anomaly decision.

    Times the conveyor-belt inference path at batch=1:
        preprocess  -- model.pre_processor: resize + center-crop + normalize (CPU)
        h2d         -- host -> GPU copy
        forward     -- model.model: encoder + bottleneck + decoder + anomaly map
        postprocess -- model.post_processor: normalize + threshold -> decision

    raw_image : single CHW float tensor on CPU (one real test image at native
                resolution -- simulates an arriving camera frame).
    Returns {stage: {mean,p50,p95,p99}} in ms for pre/h2d/fwd/post/total,
    or None if CUDA unavailable. perf_counter + cuda.synchronize because the
    pipeline mixes CPU and GPU work.
    """
    if not torch.cuda.is_available():
        return None
    inner = model.model.to(device).eval()
    pre   = model.pre_processor
    post  = model.post_processor.to(device)
    raw   = raw_image.unsqueeze(0)  # (1,C,H,W), CPU

    def _one():
        t0 = time.perf_counter()
        x = pre(raw)
        t1 = time.perf_counter()
        x = x.to(device); torch.cuda.synchronize()
        t2 = time.perf_counter()
        out = inner(x); torch.cuda.synchronize()
        t3 = time.perf_counter()
        post(out); torch.cuda.synchronize()
        t4 = time.perf_counter()
        return (t1-t0, t2-t1, t3-t2, t4-t3, t4-t0)

    for _ in range(warmup):
        _one()
    keys = ("pre", "h2d", "fwd", "post", "total")
    acc = {k: [] for k in keys}
    for _ in range(iters):
        for k, s in zip(keys, _one()):
            acc[k].append(s * 1e3)

    def _stat(v):
        v = sorted(v); n = len(v)
        return {"mean": sum(v)/n, "p50": v[n//2],
                "p95": v[min(n-1, int(n*0.95))],
                "p99": v[min(n-1, int(n*0.99))]}
    return {k: _stat(v) for k, v in acc.items()}


@torch.no_grad()
def _peak_gpu_mem(model,
                  input_shape: tuple = (1, 3, 392, 392)):
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

def _partition(cats: list, instance: int, total: int) -> list:
    if total <= 1:
        return cats
    return [c for i, c in enumerate(cats) if i % total == instance]

_mvtec = _partition(MVTEC_CATEGORIES, INSTANCE, N_INSTANCES)
_visa   = _partition(VISA_CATEGORIES,  INSTANCE, N_INSTANCES)

DATASETS = [
    ("mvtec", MVTecAD, _mvtec, "./datasets/MVTecAD"),
    ("visa",  Visa,    _visa,  "./datasets/VisA"),
]

total_categories = sum(len(c) for _, _, c, _ in DATASETS)

_inst_suffix = f"_inst{INSTANCE}" if N_INSTANCES > 1 else ""
os.makedirs("results", exist_ok=True)
csv_path     = f"results/dinomaly_results{_inst_suffix}.csv"
combined_txt = f"results/dinomaly_combined{_inst_suffix}.txt"
CKPT_DIR     = f"results/Dinomaly{_inst_suffix}"
WORK_DIR     = f"lightning_logs{_inst_suffix}"

# ── Startup banner ────────────────────────────────────────────
print("=" * 60)
print(f"  Dinomaly  |  encoder=dinov2reg_vit_base_14  |  decoder_depth=8")
print(f"  dropout=0.2  |  image={IMAGE_SIZE}→crop={CROP_SIZE}  |  batch={BATCH_SIZE}  |  max_steps={MAX_STEPS}")
print(f"  Runs per category : {N_RUNS}")
if N_INSTANCES > 1:
    print(f"  Instance          : {INSTANCE}/{N_INSTANCES}")
print(f"  Categories total  : {total_categories}  ({len(_mvtec)} MVTec + {len(_visa)} VisA)")
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device            : {gpu_info}")
print("=" * 60)

results = {ds: {} for ds, *_ in DATASETS}
if os.path.exists(csv_path):
    prior = pd.read_csv(csv_path)
    for _, r in prior.iterrows():
        ds, cat = r["dataset"], r["category"]
        results.setdefault(ds, {}).setdefault(cat, []).append(r.to_dict())
    print(f"\n[resume] Loaded {len(prior)} prior rows from {csv_path}")
else:
    print("\n[start] No CSV found — starting fresh.")

# ── OOM skip tracking ─────────────────────────────────────────
oom_skips: list[dict] = []
cycle_secs: list[float] = []
t_total_start = time.time()

# ── Experiment loop (run-outer) ───────────────────────────────
for run in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"  RUN {run+1}/{N_RUNS}")
    print(f"{'='*60}")
    t_cycle_start = time.time()

    for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
        print(f"\n{'-'*60}")
        print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
        print(f"{'-'*60}")

        for cat_idx, category in enumerate(categories, 1):
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
            t_run_start = time.time()
            try:
                print(f"  → Loading datamodule...")
                # (Sec. 4.1): batch size 16, no val split needed (fixed max_steps)
                datamodule = DataModule(
                    root=root,
                    category=category,
                    train_batch_size=BATCH_SIZE,
                    eval_batch_size=BATCH_SIZE,
                )

                image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                pixel_pro   = AUPRO(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                image_f1max = F1Max(fields=["pred_score", "gt_label"], prefix="image_")
                pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
                evaluator = Evaluator(test_metrics=[image_auroc, pixel_auroc, pixel_pro,
                                                    image_f1max, pixel_f1max])

                print(f"  → Building model (Dinomaly dinov2reg_vit_base_14)...")
                # NOTE: fuse_layer_encoder / fuse_layer_decoder NOT passed
                # explicitly because anomalib 2.4.2 torch_model.py has a bug
                # (missing else branch L133-136) that drops custom values.
                # Defaults DEFAULT_FUSE_LAYERS = [[0,1,2,3],[4,5,6,7]] already
                # match Dinomaly paper Sec. 3.5 "Loose Constraint" (2 groups).
                model = Dinomaly(
                    # (Sec. 3.1): DINOv2 with registers, ViT-B/14
                    encoder_name="dinov2reg_vit_small_14",
                    # (Sec. 3.3 "Noisy Bottleneck"): MLP dropout p=0.2
                    bottleneck_dropout=0.2,
                    # (Sec. 3.4 / Tab. 1): 8 ViT decoder layers
                    decoder_depth=8,
                    # (Sec. 3.2): middle 8 of 12 encoder blocks (1-based 3..10)
                    target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                    remove_class_token=False,
                    # (Sec. 4.1): resize 448 → center-crop 392 (= 28·14)
                    pre_processor=Dinomaly.configure_pre_processor(
                        image_size=(IMAGE_SIZE, IMAGE_SIZE),
                        crop_size=CROP_SIZE,
                    ),
                    evaluator=evaluator,
                    visualizer=False,
                )

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  → Parameters: {n_params:,}")

                print(f"  → Training (max_steps={MAX_STEPS})...")
                # Optimizer / schedule (StableAdamW, lr=2e-3, wd=1e-4,
                # warmup 100, cosine decay) handled in Dinomaly.configure_optimizers
                # using the trainer's max_steps. Hard-mining cosine loss
                # (ratio 0→0.9 over 1000 steps) handled inside model.forward.
                engine = Engine(
                    max_steps=MAX_STEPS,
                    devices=1,
                    logger=False,
                    callbacks=[CompactBar()],
                    default_root_dir=WORK_DIR,
                )
                engine.fit(model=model, datamodule=datamodule)
                print(f"  → Training complete.")

                # ── FLOPs: encoder forward on one image ────────────────────
                flops_M = None
                try:
                    device = next(model.parameters()).device
                    _dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)
                    with FlopCounterMode(display=False) as fcm:
                        _ = model.model.encoder(_dummy)
                    flops_M = round(fcm.get_total_flops() / 1e6, 1)
                    print(f"  → Encoder FLOPs: {flops_M:.1f} M")
                    del _dummy
                except Exception:
                    pass

                extractor = model.model.encoder
                input_shape = (1, 3, CROP_SIZE, CROP_SIZE)

                try:
                    peak_mem_1fwd = _peak_gpu_mem(extractor, input_shape=input_shape)
                    print(f"  → Peak mem (1 fwd): {peak_mem_1fwd/1e6:.1f} MB" if peak_mem_1fwd else "")
                except Exception:
                    peak_mem_1fwd = None

                # ── GPU inference on test set ─────────────────────────
                print(f"  → Running inference on test set (GPU)...")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                test_results = engine.test(model=model, datamodule=datamodule)

                peak_gpu_mb = 0.0
                if torch.cuda.is_available():
                    peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

                # ── Deployment latency: raw frame -> anomaly decision ──
                try:
                    raw_img = datamodule.test_data[0].image
                    deploy = _deploy_latency(model, raw_img)
                except Exception:
                    deploy = None
                if deploy:
                    t = deploy["total"]
                    print(f"  → Deploy latency: {t['mean']:.2f} ms  "
                          f"(p95 {t['p95']:.2f}, p99 {t['p99']:.2f})  "
                          f"[pre {deploy['pre']['mean']:.2f} | fwd {deploy['fwd']['mean']:.2f} "
                          f"| post {deploy['post']['mean']:.2f}]")

                metrics = test_results[0]
                img_auc = metrics.get("image_AUROC", 0) * 100
                pxl_auc = metrics.get("pixel_AUROC", 0) * 100
                pxl_pro = metrics.get("pixel_AUPRO", 0) * 100
                img_f1  = (metrics["image_F1Max"] * 100) if metrics.get("image_F1Max") is not None else None
                pxl_f1  = (metrics["pixel_F1Max"] * 100) if metrics.get("pixel_F1Max") is not None else None

                print(f"  → Results: Img AUROC {img_auc:.1f}% | Pxl AUROC {pxl_auc:.1f}%"
                      f" | AUPRO {pxl_pro:.1f}%"
                      + (f" | F1Max {img_f1:.1f}%" if img_f1 is not None else ""))
                print(f"  → Peak GPU: {peak_gpu_mb:.1f} MB")

                run_wall_s = time.time() - t_run_start
                run_record = {
                    "image_AUROC":     img_auc,
                    "pixel_AUROC":     pxl_auc,
                    "pixel_AUPRO":     pxl_pro,
                    "image_F1Max":     img_f1,
                    "pixel_F1Max":     pxl_f1,
                    "n_params":        n_params,
                    "flops_M":         flops_M,
                    "peak_gpu_mb":     peak_gpu_mb,
                    "peak_gpu_mem_b":  peak_mem_1fwd,
                    "deploy_total_ms":     deploy["total"]["mean"] if deploy else None,
                    "deploy_total_p95_ms": deploy["total"]["p95"]  if deploy else None,
                    "deploy_total_p99_ms": deploy["total"]["p99"]  if deploy else None,
                    "deploy_pre_ms":       deploy["pre"]["mean"]   if deploy else None,
                    "deploy_h2d_ms":       deploy["h2d"]["mean"]   if deploy else None,
                    "deploy_fwd_ms":       deploy["fwd"]["mean"]   if deploy else None,
                    "deploy_post_ms":      deploy["post"]["mean"]  if deploy else None,
                    "run_wall_s":      run_wall_s,
                }
                results[ds_name][category].append(run_record)

                csv_row = {"dataset": ds_name, "category": category, "run": run + 1, **run_record}
                pd.DataFrame([csv_row]).to_csv(
                    csv_path, mode="a",
                    header=not os.path.exists(csv_path),
                    index=False,
                )
                print(f"  → Row saved ({run_wall_s:.1f}s wall).")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_skips.append({"ds": ds_name, "category": category,
                                       "run": run + 1, "wall_s": time.time() - t_run_start})
                    print(f"\n  [OOM] CUDA out of memory — {ds_name}/{category} run {run+1}. "
                          f"Skipping. (total OOM skips so far: {len(oom_skips)})")
                else:
                    raise
            finally:
                del model, engine, datamodule, evaluator
                del image_auroc, pixel_auroc, pixel_pro, image_f1max, pixel_f1max
                del test_results, metrics
                free_gpu()
                shutil.rmtree(CKPT_DIR, ignore_errors=True)
                shutil.rmtree(WORK_DIR, ignore_errors=True)

    cycle_secs.append(time.time() - t_cycle_start)
    print(f"\n  Cycle {run+1}/{N_RUNS} complete in {cycle_secs[-1]:.1f}s ({cycle_secs[-1]/60:.1f} min)")

total_secs = time.time() - t_total_start

# ── Post-loop ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  All runs complete in {total_secs:.1f}s ({total_secs/60:.1f} min)")
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
                "peak_gpu_mb": rec.get("peak_gpu_mb"),
                "peak_mem_b": rec.get("peak_gpu_mem_b"),
                "deploy_total_ms": rec.get("deploy_total_ms"),
                "deploy_total_p95_ms": rec.get("deploy_total_p95_ms"),
                "deploy_fwd_ms": rec.get("deploy_fwd_ms"),
                "deploy_pre_ms": rec.get("deploy_pre_ms"),
                "deploy_post_ms": rec.get("deploy_post_ms"),
                "run_wall_s": rec.get("run_wall_s"),
            })

if not rows:
    print("No results to display.")
    sys.exit(0)

df = pd.DataFrame(rows)

metric_cols = ["img_auroc", "pxl_auroc", "aupro", "img_f1max", "pxl_f1max",
               "params", "flops_M", "peak_gpu_mb", "peak_mem_b",
               "deploy_total_ms", "deploy_total_p95_ms", "deploy_fwd_ms",
               "deploy_pre_ms", "deploy_post_ms"]

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

lines = []
lines.append("=" * 140)
lines.append(f"  Dinomaly  (N={N_RUNS}, dinov2reg_vit_base_14, dropout=0.2, decoder_depth=8, "
             f"image={IMAGE_SIZE}→{CROP_SIZE}, batch={BATCH_SIZE}, max_steps={MAX_STEPS})")
lines.append(f"  Raw CSV: {csv_path}")
lines.append("=" * 140)
lines.append(summary.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "—"))

lines.append("")
lines.append("=" * 80)
lines.append("  Timing")
lines.append("=" * 80)
lines.append(f"  Total wall:        {total_secs:.1f} s  ({total_secs/60:.1f} min)")
for i, c in enumerate(cycle_secs, 1):
    lines.append(f"  Cycle {i}/{N_RUNS} wall:    {c:.1f} s  ({c/60:.1f} min)")
if "run_wall_s" in df.columns:
    rw = df["run_wall_s"].dropna()
    if len(rw):
        lines.append(f"  Per-run wall:      mean {rw.mean():.1f} s  min {rw.min():.1f}  max {rw.max():.1f}")

lines.append("")
lines.append("=" * 80)
if oom_skips:
    lines.append(f"  OOM Skips ({len(oom_skips)} total)")
    lines.append("=" * 80)
    for s in oom_skips:
        wall = s.get("wall_s")
        wall_str = f"  ({wall:.1f}s before OOM)" if wall is not None else ""
        lines.append(f"  • {s['ds']}/{s['category']}  run {s['run']}{wall_str}")
else:
    lines.append("  No OOM skips recorded.")
lines.append("=" * 80)

text = "\n".join(lines)
print("\n" + text)
with open(combined_txt, "w") as f:
    f.write(text + "\n")
print(f"\nSummary written to {combined_txt}")
