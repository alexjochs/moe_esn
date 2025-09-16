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

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$SUBMIT_DIR"

module purge
module load python/3.12 || true

VENV_DIR=${VENV_DIR:-"$SUBMIT_DIR/.venv"}
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv not found at $VENV_DIR" >&2
  echo "Run $SUBMIT_DIR/setup_venv.sh before submitting this job." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python ./run_dask_test.py
