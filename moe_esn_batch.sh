#!/bin/bash
#SBATCH --job-name=moe_esn
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

TAG=${TAG:-moe-$(date +%Y%m%d-%H%M%S)}
OUTDIR=${OUTDIR:-$PWD/runs}
ITERATIONS=${ITERATIONS:-30}

mkdir -p "$OUTDIR"

module purge
module load python/3.12 || true
module load slurm || module load slurm/slurm || true

VENV_DIR="$PWD/venv"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export ORCHESTRATOR_THREADS=${ORCHESTRATOR_THREADS:-${SLURM_CPUS_PER_TASK:-10}}
export OMP_NUM_THREADS=$ORCHESTRATOR_THREADS
export OPENBLAS_NUM_THREADS=$ORCHESTRATOR_THREADS
export MKL_NUM_THREADS=$ORCHESTRATOR_THREADS
export VECLIB_MAXIMUM_THREADS=$ORCHESTRATOR_THREADS
export NUMEXPR_NUM_THREADS=$ORCHESTRATOR_THREADS
export MPLBACKEND=Agg

export ESN_DASK_JOBS=${ESN_DASK_JOBS:-20}
export ESN_DASK_WORKER_CORES=${ESN_DASK_WORKER_CORES:-10}
export ESN_DASK_WORKER_MEM=${ESN_DASK_WORKER_MEM:-32GB}
export ESN_DASK_WORKER_WALLTIME=${ESN_DASK_WORKER_WALLTIME:-12:00:00}
export ESN_DASK_PARTITION=${ESN_DASK_PARTITION:-preempt}
export ESN_DASK_ACCOUNT=${ESN_DASK_ACCOUNT:-eecs}
# Force worker jobs to be requeued after preemption so Dask maintains the pool.
export ESN_DASK_REQUEUE=1
export ESN_DASK_PROCESSES_PER_JOB=${ESN_DASK_PROCESSES_PER_JOB:-2}

set -x
python -u mixture_of_reservoirs_annotated.py \
  --outdir "$OUTDIR" \
  --tag "$TAG" \
  --iterations "$ITERATIONS"
set +x

scontrol show job "$SLURM_JOB_ID" | egrep "TRES|MinMemoryCPU|NumCPUs|Nodes|Partition|Account" || true
