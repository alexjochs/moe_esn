#!/bin/bash
#SBATCH --job-name=dask_smoke
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

module purge
module load python/3.12 || true

VENV_DIR=${VENV_DIR:-"$SCRIPT_DIR/.venv"}
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv not found at $VENV_DIR" >&2
  echo "Run $SCRIPT_DIR/setup_venv.sh before submitting this job." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

export DASK_VENV_PATH="$VENV_DIR"
export DASK_ACCOUNT=${DASK_ACCOUNT:-eecs}
export DASK_QUEUE=${DASK_QUEUE:-share}
export DASK_JOBS=${DASK_JOBS:-2}
export DASK_WORKER_CORES=${DASK_WORKER_CORES:-2}
export DASK_WORKER_MEM=${DASK_WORKER_MEM:-4GB}
export DASK_WORKER_WALLTIME=${DASK_WORKER_WALLTIME:-00:15:00}
export DASK_PROCESSES_PER_JOB=${DASK_PROCESSES_PER_JOB:-1}
export DASK_WAIT_TIMEOUT=${DASK_WAIT_TIMEOUT:-300}
export DASK_EXPECTED_WORKERS=${DASK_EXPECTED_WORKERS:-$((DASK_JOBS * DASK_PROCESSES_PER_JOB))}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python ./run_dask_test.py
