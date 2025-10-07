#!/bin/bash
#SBATCH --job-name=moe_esn_oracle
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=%x-n100-%j.out
#SBATCH --error=%x-n100-%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

TAG=${TAG:-moe-oracle-n100-$(date +%Y%m%d-%H%M%S)}
OUTDIR=${OUTDIR:-$PWD/runs}
ITERATIONS=${ITERATIONS:-30}
GA_POPULATION=${GA_POPULATION:-5000}
GA_EXPLORE_GENERATIONS=${GA_EXPLORE_GENERATIONS:-8}
GA_ELITE_FRACTION=${GA_ELITE_FRACTION:-0.1}
GA_RANDOM_FRAC_START=${GA_RANDOM_FRAC_START:-0.4}
GA_RANDOM_FRAC_END=${GA_RANDOM_FRAC_END:-0.05}
GA_MUTATION_SCALE_START=${GA_MUTATION_SCALE_START:-1.0}
GA_MUTATION_SCALE_END=${GA_MUTATION_SCALE_END:-0.25}
GA_TOURNAMENT_SIZE=${GA_TOURNAMENT_SIZE:-3}

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

export ESN_DASK_JOBS=${ESN_DASK_JOBS:-40}
export ESN_DASK_WORKER_CORES=${ESN_DASK_WORKER_CORES:-10}
export ESN_DASK_WORKER_MEM=${ESN_DASK_WORKER_MEM:-32GB}
export ESN_DASK_WORKER_WALLTIME=${ESN_DASK_WORKER_WALLTIME:-12:00:00}
export ESN_DASK_PARTITION=${ESN_DASK_PARTITION:-preempt}
export ESN_DASK_ACCOUNT=${ESN_DASK_ACCOUNT:-eecs}
# Force worker jobs to be requeued after preemption so Dask maintains the pool.
export ESN_DASK_REQUEUE=1
export ESN_DASK_PROCESSES_PER_JOB=${ESN_DASK_PROCESSES_PER_JOB:-2}
export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=${DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES:-100}

set -x
python -u mixture_of_reservoirs_oracle.py \
  --outdir "$OUTDIR" \
  --tag "$TAG" \
  --iterations "$ITERATIONS" \
  --ga-population "$GA_POPULATION" \
  --ga-explore-generations "$GA_EXPLORE_GENERATIONS" \
  --ga-elite-fraction "$GA_ELITE_FRACTION" \
  --ga-random-frac-start "$GA_RANDOM_FRAC_START" \
  --ga-random-frac-end "$GA_RANDOM_FRAC_END" \
  --ga-mutation-scale-start "$GA_MUTATION_SCALE_START" \
  --ga-mutation-scale-end "$GA_MUTATION_SCALE_END" \
  --ga-tournament-size "$GA_TOURNAMENT_SIZE"
set +x

scontrol show job "$SLURM_JOB_ID" | egrep "TRES|MinMemoryCPU|NumCPUs|Nodes|Partition|Account" || true
