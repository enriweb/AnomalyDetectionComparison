"""
run_all.py — Orchestrate all anomaly detection experiment scripts

Usage:
    python run_all.py                              # all scripts, 1 run, sequential
    python run_all.py --runs 3                     # 3 runs each, sequential
    python run_all.py --parallel                   # all scripts concurrently
    python run_all.py --parallel --jobs 2          # max 2 concurrent (default: all)
    python run_all.py --scripts patchcore,cfa      # subset only
    python run_all.py --list                       # list available scripts

NOTE: --parallel on a single GPU will likely cause OOM between scripts.
      Use sequential (default) unless you have multiple GPUs or are testing.
      Each script's MAX_VRAM_GB threshold will help manage VRAM within a
      single script, but not across concurrent processes.

Output:
    Live: per-script status line (started / done / FAILED)
    Logs: results/run_all_<script>.log  (full stdout+stderr per script)
    Summary: results/run_all_summary.txt
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Available scripts (order = default sequential order) ─────
SCRIPTS = {
    "patchcore":  "patchcore.py",
    "efficientad": "efficientAD.py",
    "cfa":        "cfa.py",
    "uninet":     "uninet.py",
    "dinomaly":   "dinomaly.py",
    "mobilevit":  "mobilevit.py",
    "hybridcore": "hybridcore.py",
}

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run all AD experiment scripts.")
parser.add_argument("--runs",     type=int,   default=1,
                    help="Number of runs per category (passed to each script).")
parser.add_argument("--parallel", action="store_true",
                    help="Run all scripts concurrently instead of sequentially.")
parser.add_argument("--jobs",     type=int,   default=0,
                    help="Max concurrent scripts in parallel mode (0 = unlimited).")
parser.add_argument("--scripts",  type=str,   default="",
                    help="Comma-separated list of scripts to run (e.g. patchcore,cfa).")
parser.add_argument("--list",     action="store_true",
                    help="Print available script names and exit.")
args = parser.parse_args()

if args.list:
    print("Available scripts:")
    for name, path in SCRIPTS.items():
        print(f"  {name:<15} → {path}")
    sys.exit(0)

# ── Select scripts ────────────────────────────────────────────
if args.scripts:
    selected = [s.strip() for s in args.scripts.split(",")]
    unknown = [s for s in selected if s not in SCRIPTS]
    if unknown:
        print(f"[ERROR] Unknown script(s): {', '.join(unknown)}")
        print(f"        Available: {', '.join(SCRIPTS)}")
        sys.exit(1)
    run_list = [(name, SCRIPTS[name]) for name in selected]
else:
    run_list = list(SCRIPTS.items())

os.makedirs("results", exist_ok=True)

# ── Banner ────────────────────────────────────────────────────
mode_str = f"parallel (max_jobs={'unlimited' if args.jobs <= 0 else args.jobs})" if args.parallel else "sequential"
print("=" * 65)
print(f"  run_all.py")
print(f"  Mode     : {mode_str}")
print(f"  Runs     : {args.runs} per category")
print(f"  Scripts  : {len(run_list)}  ({', '.join(n for n, _ in run_list)})")
print(f"  Logs     : results/run_all_<name>.log")
print("=" * 65)

# ── Runner ────────────────────────────────────────────────────
results_summary: list[dict] = []

def run_script(name: str, path: str) -> dict:
    log_path = f"results/run_all_{name}.log"
    cmd = [sys.executable, path, str(args.runs)]
    t_start = time.time()
    start_str = datetime.now().strftime("%H:%M:%S")
    print(f"  [START {start_str}]  {name}  →  {path}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        proc.wait()

    elapsed = time.time() - t_start
    done_str = datetime.now().strftime("%H:%M:%S")
    status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins  = divmod(mins, 60)
    elapsed_fmt = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    print(f"  [{status.upper() if proc.returncode != 0 else 'DONE ' + done_str}]  "
          f"{name}  elapsed: {elapsed_fmt}  log: {log_path}")

    return {
        "name":       name,
        "path":       path,
        "returncode": proc.returncode,
        "elapsed_s":  round(elapsed, 1),
        "log":        log_path,
    }


wall_start = time.time()

if args.parallel:
    max_workers = args.jobs if args.jobs > 0 else len(run_list)
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for name, path in run_list:
            fut = pool.submit(run_script, name, path)
            futures[fut] = name
        for fut in as_completed(futures):
            results_summary.append(fut.result())
else:
    for name, path in run_list:
        results_summary.append(run_script(name, path))

wall_elapsed = time.time() - wall_start
wm, ws = divmod(int(wall_elapsed), 60)
wh, wm = divmod(wm, 60)

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SUMMARY  —  total wall time: {wh:02d}:{wm:02d}:{ws:02d}")
print(f"{'='*65}")
ok  = [r for r in results_summary if r["returncode"] == 0]
bad = [r for r in results_summary if r["returncode"] != 0]

for r in sorted(results_summary, key=lambda x: x["elapsed_s"], reverse=True):
    sym = "✓" if r["returncode"] == 0 else "✗"
    m, s = divmod(int(r["elapsed_s"]), 60)
    h, m = divmod(m, 60)
    print(f"  {sym}  {r['name']:<15}  {h:02d}:{m:02d}:{s:02d}  log: {r['log']}")

print(f"\n  Passed: {len(ok)}/{len(results_summary)}"
      + (f"   Failed: {[r['name'] for r in bad]}" if bad else ""))

# ── Write summary file ────────────────────────────────────────
summary_path = "results/run_all_summary.txt"
lines = [
    f"run_all.py  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Mode: {mode_str}   Runs: {args.runs}   Wall: {wh:02d}:{wm:02d}:{ws:02d}",
    "=" * 65,
]
for r in sorted(results_summary, key=lambda x: x["elapsed_s"], reverse=True):
    sym = "OK  " if r["returncode"] == 0 else "FAIL"
    m, s = divmod(int(r["elapsed_s"]), 60)
    h, m = divmod(m, 60)
    lines.append(f"  [{sym}]  {r['name']:<15}  {h:02d}:{m:02d}:{s:02d}  {r['log']}")
lines.append("=" * 65)

with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n  Summary saved to {summary_path}")
