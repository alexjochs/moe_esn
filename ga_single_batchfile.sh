#!/bin/bash
#SBATCH --job-name=ga_cpu
#SBATCH --account=eecs  
#SBATCH --partition=share           # high core-count partition for worker fan-out
#SBATCH --time=4:00:00              # walltime HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # head node just orchestrates Dask
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL

set -euo pipefail
# -------- Parameters via environment or defaults --------
OUTDIR=${OUTDIR:-$PWD/runs}
TAG=${TAG:-ga-$(date +%Y%m%d-%H%M%S)}
POP=${POP:-60}
GENS=${GENS:-120}
SEEDS_ENV=${SEEDS:-"42 1337 2025 1962"}


# Ensure output directory exists per-node
mkdir -p "$OUTDIR"

# ---------- Environment ----------
module purge
module load python/3.12 || true
module load slurm || module load slurm/slurm || true

VENV_DIR="$PWD/venv"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Make multiprocessing predictable and avoid over-threading by math libs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg  # no GUI

DASK_JOBS=${DASK_JOBS:-40}
DASK_WORKER_CORES=${DASK_WORKER_CORES:-10}
DASK_WORKER_MEM=${DASK_WORKER_MEM:-16GB}
DASK_WORKER_WALLTIME=${DASK_WORKER_WALLTIME:-01:00:00}
DASK_PARTITION=${DASK_PARTITION:-share}
DASK_ACCOUNT=${DASK_ACCOUNT:-eecs}
DASK_TIMEOUT=${DASK_TIMEOUT:-600}
DASK_PROCESSES_PER_WORKER=${DASK_PROCESSES_PER_WORKER:-1}

# Convert SEEDS string into array for nice expansion below
read -r -a SEED_ARR <<< "$SEEDS_ENV"

# ---------- Run ----------
set -x
python -u "single_reservoir_baseline.py" \
  --ga \
  --dask \
  --jobs "$DASK_JOBS" \
  --dask-worker-cores "$DASK_WORKER_CORES" \
  --dask-worker-mem "$DASK_WORKER_MEM" \
  --dask-worker-walltime "$DASK_WORKER_WALLTIME" \
  --dask-account "$DASK_ACCOUNT" \
  --dask-partition "$DASK_PARTITION" \
  --dask-processes-per-worker "$DASK_PROCESSES_PER_WORKER" \
  --dask-timeout "$DASK_TIMEOUT" \
  --pop "$POP" \
  --gens "$GENS" \
  --seeds "${SEED_ARR[@]}" \
  --no-plots \
  --outdir "$OUTDIR" \
  --tag "$TAG"
set +x

# ---------- Post-run accounting (optional) ----------
echo "\n=== Job resources ==="
scontrol show job "$SLURM_JOB_ID" | egrep "TRES|MinMemoryCPU|NumCPUs|Nodes|Partition|Account" || true
