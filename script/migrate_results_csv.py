"""
One-off migration: update results/*_results.csv to the new metric schema.

Drops the removed timing columns, adds the new deploy-latency columns
(filled with a placeholder so old rows don't break averaging/resume), and
reorders columns to match what the experiment scripts now append.

Old rows keep their accuracy/memory values; their deploy_* cells are empty
(NaN) -- mean/std aggregation in the scripts skips NaN, so this is safe.

Usage:
    python migrate_results_csv.py            # migrate results/*_results.csv
    python migrate_results_csv.py --dry-run  # show what would change, write nothing

A backup <file>.bak is written before each file is overwritten.
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Columns removed in the deploy-latency rework.
OLD_COLS = ["inference_gpu_s", "gpu_throughput", "cpu_latency_ms", "single_img_lat_ms"]

# New columns, with the placeholder used for pre-existing rows.
NEW_COLS = [
    "deploy_total_ms", "deploy_total_p95_ms", "deploy_total_p99_ms",
    "deploy_pre_ms", "deploy_h2d_ms", "deploy_fwd_ms", "deploy_post_ms",
]
PLACEHOLDER = float("nan")

# Canonical column order -- must match {"dataset","category","run", **run_record}
# as the experiment scripts now write it, so appended rows stay aligned.
CANONICAL = [
    "dataset", "category", "run",
    "image_AUROC", "pixel_AUROC", "pixel_AUPRO", "image_F1Max", "pixel_F1Max",
    "n_params", "flops_M", "peak_gpu_mb", "peak_gpu_mem_b",
    "deploy_total_ms", "deploy_total_p95_ms", "deploy_total_p99_ms",
    "deploy_pre_ms", "deploy_h2d_ms", "deploy_fwd_ms", "deploy_post_ms",
    "run_wall_s",
]


def migrate(path: str, dry_run: bool) -> None:
    df = pd.read_csv(path)
    before = list(df.columns)

    dropped = [c for c in OLD_COLS if c in df.columns]
    df = df.drop(columns=dropped)

    added = [c for c in NEW_COLS if c not in df.columns]
    for c in added:
        df[c] = PLACEHOLDER

    # Reorder: canonical columns first (in canonical order), then any extras
    # present in the file but not in the schema -- never silently lose data.
    ordered = [c for c in CANONICAL if c in df.columns]
    extras = [c for c in df.columns if c not in CANONICAL]
    df = df[ordered + extras]

    print(f"\n{path}  ({len(df)} rows)")
    print(f"  dropped : {dropped or '-'}")
    print(f"  added   : {added or '-'}")
    if extras:
        print(f"  kept extra (not in schema): {extras}")
    if before == list(df.columns):
        print("  no change.")
        return

    if dry_run:
        print("  [dry-run] not written.")
        return

    bak = path + ".bak"
    os.replace(path, bak)
    df.to_csv(path, index=False)
    print(f"  written. backup: {bak}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    ap.add_argument("--results-dir", default="results", help="folder to scan")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*_results.csv")))
    if not files:
        print(f"No *_results.csv found in {args.results_dir}/")
        sys.exit(0)

    print(f"Found {len(files)} CSV file(s) in {args.results_dir}/")
    for f in files:
        migrate(f, args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
