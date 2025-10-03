#!/bin/bash
#SBATCH --job-name=mg_grid_pca
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load python/3.12 || true
module load slurm || module load slurm/slurm || true

VENV_DIR=${VENV_DIR:-$PWD/venv}
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/runs/matplotlib_cache}
mkdir -p "$MPLCONFIGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MAX_COMBINATIONS=${MAX_COMBINATIONS:-200000}
INTERPOLATION_POINTS=${INTERPOLATION_POINTS:-5}
MAX_WORKERS=${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}

ensure_runs_dir() {
  local target_dir
  target_dir=$(dirname "$1")
  mkdir -p "$target_dir"
}

PLOT_PATH=${PLOT_PATH:-$PWD/runs/local_mackey_grid_pca/pca_landscape.png}
CSV_PATH=${CSV_PATH:-$PWD/runs/local_mackey_grid_pca/results.csv}
ensure_runs_dir "$PLOT_PATH"
ensure_runs_dir "$CSV_PATH"

set -x
python -u local_mackey_grid_pca.py \
  --interpolation-points "$INTERPOLATION_POINTS" \
  --max-combinations "$MAX_COMBINATIONS" \
  --max-workers "$MAX_WORKERS" \
  --plot-path "$PLOT_PATH" \
  --csv-path "$CSV_PATH"
set +x

scontrol show job "$SLURM_JOB_ID" | egrep "TRES|MinMemoryCPU|NumCPUs|Nodes|Partition|Account" || true
