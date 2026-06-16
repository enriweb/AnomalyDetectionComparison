"""
PatchCore-25% with ConvNeXt V2 Nano / Pico / Femto / Atto backbones
Two layer configs tested per variant (8 configs total):

  s12 — stages.1 + stages.2  (28×28 + 14×14 px, analogous to WRN-50 layer2+3)
  s23 — stages.2 + stages.3  (14×14 + 7×7 px, deeper/more semantic)

Usage:
    python convnextv2_small.py          # 1 run per config (default)
    python convnextv2_small.py 3        # 3 runs per config, results averaged

Variants (all FCMAE-pretrained, PatchCore-25%):
    nano   — convnextv2_nano.fcmae_ft_in22k_in1k   ~15.6 M params
    pico   — convnextv2_pico.fcmae_ft_in1k          ~ 9.1 M params
    femto  — convnextv2_femto.fcmae_ft_in1k         ~ 5.2 M params
    atto   — convnextv2_atto.fcmae_ft_in1k          ~ 3.7 M params

Metrics recorded per run:
    Accuracy  — image AUROC, pixel AUROC, pixel AUPRO, image F1Max, pixel F1Max
    Efficiency — n_params, FLOPs (backbone), peak GPU MB,
                 deploy latency: total mean/p95/p99 + pre/fwd/post breakdown

Progress checkpointed to results/convnextv2_<variant>_<layers>_results.csv.
Auto-resumes from CSV on restart (per config).
"""

import gc
import logging
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _log in ("lightning", "lightning.pytorch", "anomalib", "torchvision", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)
print("Importing torch-related libraries...")
import torch
import torch.nn as nn
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, F1Max, Evaluator
from anomalib.models import Patchcore
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.flop_counter import FlopCounterMode
import pandas as pd

torch.set_float32_matmul_precision('high')

N_RUNS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BACKBONE_FILTER: str | None = sys.argv[2].lower() if len(sys.argv) > 2 else None

CORESET = 0.25

# (config_name, backbone, layers)
# s12 = stages.1+2 (28×14 px) — analogous to WRN-50 layer2+3 used in original PatchCore
# s23 = stages.2+3 (14×7 px)  — deeper semantic features
CONFIGS = [
    ("nano_s012",  "convnextv2_nano.fcmae_ft_in22k_in1k",  ["stages.0", "stages.1", "stages.2"]),
    ("nano_s12",   "convnextv2_nano.fcmae_ft_in22k_in1k",  ["stages.1", "stages.2"]),
    ("nano_s23",   "convnextv2_nano.fcmae_ft_in22k_in1k",  ["stages.2", "stages.3"]),
    ("pico_s012",  "convnextv2_pico.fcmae_ft_in1k",        ["stages.0", "stages.1", "stages.2"]),
    ("pico_s12",   "convnextv2_pico.fcmae_ft_in1k",        ["stages.1", "stages.2"]),
    ("pico_s23",   "convnextv2_pico.fcmae_ft_in1k",        ["stages.2", "stages.3"]),
    ("femto_s012", "convnextv2_femto.fcmae_ft_in1k",       ["stages.0", "stages.1", "stages.2"]),
    ("femto_s12",  "convnextv2_femto.fcmae_ft_in1k",       ["stages.1", "stages.2"]),
    ("femto_s23",  "convnextv2_femto.fcmae_ft_in1k",       ["stages.2", "stages.3"]),
    ("atto_s012",  "convnextv2_atto.fcmae_ft_in1k",        ["stages.0", "stages.1", "stages.2"]),
    ("atto_s12",   "convnextv2_atto.fcmae_ft_in1k",        ["stages.1", "stages.2"]),
    ("atto_s23",   "convnextv2_atto.fcmae_ft_in1k",        ["stages.2", "stages.3"]),
]


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def _deploy_latency(model, raw_image, device="cuda", warmup=20, iters=200):
    """Per-image deployment latency: raw camera frame -> anomaly decision."""
    if not torch.cuda.is_available():
        return None
    inner = model.model.to(device).eval()
    pre   = model.pre_processor
    post  = model.post_processor.to(device)
    raw   = raw_image.unsqueeze(0)

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
def _peak_gpu_mem(model: nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> int | None:
    if not torch.cuda.is_available():
        return None
    _model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(*input_shape, device="cuda")
    _model(x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


class CompactBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm(); bar.leave = False; return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm(); bar.leave = False; return bar
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm(); bar.leave = False; return bar


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

if BACKBONE_FILTER is not None:
    CONFIGS = [(n, b, l) for n, b, l in CONFIGS if n.startswith(BACKBONE_FILTER)]
    if not CONFIGS:
        print(f"[error] No configs match backbone filter '{BACKBONE_FILTER}'. "
              f"Valid: nano, pico, femto, atto")
        sys.exit(1)

CKPT_DIR = f"results/Patchcore_{BACKBONE_FILTER or 'all'}"
LOG_DIR  = f"lightning_logs_{BACKBONE_FILTER or 'all'}"

os.makedirs("results", exist_ok=True)

gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print("=" * 60)
print(f"  ConvNeXt V2 Small Variants  |  PatchCore-25%")
print(f"  Configs           : {', '.join(c for c, _, _ in CONFIGS)}")
print(f"  Runs per category : {N_RUNS}")
print(f"  Categories total  : {total_categories}  ({len(MVTEC_CATEGORIES)} MVTec + {len(VISA_CATEGORIES)} VisA)")
print(f"  Device            : {gpu_info}")
print("=" * 60)

t_grand_start = time.time()
all_variant_summaries = []

for variant_name, backbone, layers in CONFIGS:
    csv_path     = f"results/convnextv2_{variant_name}_results.csv"
    combined_txt = f"results/convnextv2_{variant_name}_combined.txt"

    print(f"\n{'#'*60}")
    print(f"  CONFIG: {variant_name.upper()}  ({backbone}, layers={layers})")
    print(f"{'#'*60}")

    results = {ds: {} for ds, *_ in DATASETS}
    if os.path.exists(csv_path):
        prior = pd.read_csv(csv_path)
        for _, r in prior.iterrows():
            ds, cat = r["dataset"], r["category"]
            results.setdefault(ds, {}).setdefault(cat, []).append(r.to_dict())
        print(f"[resume] Loaded {len(prior)} prior rows from {csv_path}")
    else:
        print("[start] No CSV found — starting fresh.")

    oom_skips: list[dict] = []
    cycle_secs: list[float] = []
    t_variant_start = time.time()

    for run in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"  [{variant_name}]  RUN {run+1}/{N_RUNS}")
        print(f"{'='*60}")
        t_cycle_start = time.time()

        for ds_idx, (ds_name, DataModule, categories, root) in enumerate(DATASETS, 1):
            print(f"\n{'─'*60}")
            print(f"  Dataset {ds_idx}/{len(DATASETS)}: {ds_name.upper()}  ({len(categories)} categories)")
            print(f"{'─'*60}")

            for cat_idx, category in enumerate(categories, 1):
                if len(results.get(ds_name, {}).get(category, [])) > run:
                    print(f"  [skip] {category} ({cat_idx}/{len(categories)}) — run {run+1} already done")
                    continue

                results.setdefault(ds_name, {}).setdefault(category, [])

                print(f"\n  [{cat_idx}/{len(categories)}] {category}  —  Run {run+1}/{N_RUNS}")

                model = engine = datamodule = evaluator = None
                image_auroc = pixel_auroc = pixel_pro = None
                image_f1max = pixel_f1max = test_results = metrics = None
                t_run_start = time.time()
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

                    print(f"  → Building model (ConvNeXt V2-{variant_name} backbone)...")
                    model = Patchcore(
                        backbone=backbone,
                        layers=layers,
                        pre_trained=True,
                        coreset_sampling_ratio=CORESET,
                        num_neighbors=9,
                        pre_processor=Patchcore.configure_pre_processor(
                            image_size=(256, 256),
                            center_crop_size=(224, 224),
                        ),
                        evaluator=evaluator,
                        visualizer=False,
                    )

                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  → Parameters: {n_params:,}")

                    print(f"  → Training (building coreset)...")
                    engine = Engine(max_epochs=1, devices="auto", strategy="auto",
                                    logger=False, callbacks=[CompactBar()],
                                    default_root_dir=LOG_DIR)
                    engine.fit(model=model, datamodule=datamodule)
                    print(f"  → Coreset built.")

                    flops_M = None
                    try:
                        device = next(model.parameters()).device
                        _dummy = torch.randn(1, 3, 224, 224, device=device)
                        with FlopCounterMode(display=False) as fcm:
                            _ = model.model.feature_extractor(_dummy)
                        flops_M = round(fcm.get_total_flops() / 1e6, 1)
                        print(f"  → Backbone FLOPs: {flops_M:.1f} M")
                        del _dummy
                    except Exception:
                        pass

                    extractor = model.model.feature_extractor
                    input_shape = (1, 3, 224, 224)

                    try:
                        peak_mem_1fwd = _peak_gpu_mem(extractor, input_shape=input_shape)
                        print(f"  → Peak mem (1 fwd): {peak_mem_1fwd/1e6:.1f} MB" if peak_mem_1fwd else "")
                    except Exception:
                        peak_mem_1fwd = None

                    print(f"  → Running inference on test set (GPU)...")
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    test_results = engine.test(model=model, datamodule=datamodule)

                    peak_gpu_mb = 0.0
                    if torch.cuda.is_available():
                        peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

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
                        "image_AUROC":         img_auc,
                        "pixel_AUROC":         pxl_auc,
                        "pixel_AUPRO":         pxl_pro,
                        "image_F1Max":         img_f1,
                        "pixel_F1Max":         pxl_f1,
                        "n_params":            n_params,
                        "flops_M":             flops_M,
                        "peak_gpu_mb":         peak_gpu_mb,
                        "peak_gpu_mem_b":      peak_mem_1fwd,
                        "deploy_total_ms":     deploy["total"]["mean"] if deploy else None,
                        "deploy_total_p95_ms": deploy["total"]["p95"]  if deploy else None,
                        "deploy_total_p99_ms": deploy["total"]["p99"]  if deploy else None,
                        "deploy_pre_ms":       deploy["pre"]["mean"]   if deploy else None,
                        "deploy_h2d_ms":       deploy["h2d"]["mean"]   if deploy else None,
                        "deploy_fwd_ms":       deploy["fwd"]["mean"]   if deploy else None,
                        "deploy_post_ms":      deploy["post"]["mean"]  if deploy else None,
                        "run_wall_s":          run_wall_s,
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
                    shutil.rmtree(LOG_DIR, ignore_errors=True)

        cycle_secs.append(time.time() - t_cycle_start)
        print(f"\n  [{variant_name}] Cycle {run+1}/{N_RUNS} complete in "
              f"{cycle_secs[-1]:.1f}s ({cycle_secs[-1]/60:.1f} min)")

    variant_secs = time.time() - t_variant_start

    # ── per-variant summary ──────────────────────────────────────────────────
    rows = []
    for ds_name, _, categories, _ in DATASETS:
        for cat in categories:
            runs_list = results.get(ds_name, {}).get(cat, [])
            for run_idx, rec in enumerate(runs_list):
                rows.append({
                    "variant": variant_name,
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
        print(f"\n[{variant_name}] No results to display.")
        continue

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

    def _align(frame, cols):
        if frame is None:
            return None
        frame = frame.copy()
        for c in cols:
            if c not in frame.columns:
                frame[c] = float("nan")
        return frame[cols]

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

    pieces = [_align(cat_agg, display_cols)]
    if ds_summary is not None:
        pieces.append(_align(ds_summary, display_cols))
    pieces.append(_align(overall_df, display_cols))
    summary = pd.concat(pieces)

    lines = []
    lines.append("=" * 140)
    lines.append(f"  ConvNeXt V2-{variant_name} + PatchCore-25%  "
                 f"(N={N_RUNS}, backbone={backbone}, layers={layers}, coreset={CORESET})")
    lines.append(f"  Raw CSV: {csv_path}")
    lines.append("=" * 140)
    lines.append(summary.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "—"))
    lines.append("")
    lines.append("=" * 80)
    lines.append("  Timing")
    lines.append("=" * 80)
    lines.append(f"  Variant wall:      {variant_secs:.1f} s  ({variant_secs/60:.1f} min)")
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

    all_variant_summaries.append((variant_name, layers, df[metric_cols].mean()))

grand_secs = time.time() - t_grand_start

print(f"\n{'#'*60}")
print(f"  ALL VARIANTS COMPLETE  ({grand_secs:.1f}s / {grand_secs/60:.1f} min)")
print(f"{'#'*60}\n")

if all_variant_summaries:
    print(f"{'Config':<12}  {'Layers':<18}  {'ImgAUROC':>9}  {'PxlAUROC':>9}  {'AUPRO':>7}  {'FLOPs_M':>8}  {'Params':>10}  {'DeployMs':>9}")
    print("-" * 110)
    for vname, vlayers, means in all_variant_summaries:
        layer_str = "+".join(l.split(".")[-1] for l in vlayers)
        print(f"{vname:<12}  {layer_str:<18}  "
              f"{means.get('img_auroc', float('nan')):>9.1f}  "
              f"{means.get('pxl_auroc', float('nan')):>9.1f}  "
              f"{means.get('aupro', float('nan')):>7.1f}  "
              f"{means.get('flops_M', float('nan')):>8.1f}  "
              f"{means.get('params', float('nan')):>10.0f}  "
              f"{means.get('deploy_total_ms', float('nan')):>9.2f}")
