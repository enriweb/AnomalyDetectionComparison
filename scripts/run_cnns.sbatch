#!/bin/bash
#SBATCH --job-name=thesis_cnns
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=.logs/cnn_%A_%a_%x.out
#SBATCH --error=.logs/cnn_%A_%a_%x.err

set -euo pipefail

MODELS=(patchcore efficientAD cfa)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

N_RUNS=${N_RUNS:-100}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p .logs results

echo "[$(date -Is)] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES model=$MODEL n_runs=$N_RUNS"
nvidia-smi || true

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
fi
if [ -n "${VENV_PATH:-}" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

python -u "${MODEL}.py" "$N_RUNS"

echo "[$(date -Is)] done model=$MODEL"
