#!/usr/bin/env bash
# Submit CNN array job. Optional env overrides:
#   N_RUNS=100 CONDA_ENV=thesis VENV_PATH=/path/to/venv ./scripts/submit_cnns.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p .logs

sbatch \
    --export=ALL,N_RUNS="${N_RUNS:-100}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}" \
    scripts/run_cnns.sbatch
