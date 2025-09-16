#!/bin/bash
#SBATCH --job-name=ga_cpu
#SBATCH --account=sy-grp  
#SBATCH --partition=eecs             # CPU partition; change to a CPU partition you can use
#SBATCH --time=4:00:00               # walltime HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40            # use all these threads inside the GA via --jobs
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL

# --------------------------
# Usage notes (edit these):
#   1) Set PYFILE to your entrypoint script (the one with argparse in __main__).
#   2) Replace <YOUR_PI_ACCOUNT> above with your Slurm account for the research group.
#   3) Optionally adjust partition, time, cpus, and mem based on sinfo and availability.
#   4) Submit with:  sbatch ga_single_batchfile.sh PYFILE=path/to/your_script.py TAG=myrun POP=50 GENS=20
# --------------------------

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

VENV_DIR="$OUTDIR/venv"
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

# Slurm gives you this many logical CPUs; pass it to --jobs
JOBS=$SLURM_CPUS_PER_TASK

# Convert SEEDS string into array for nice expansion below
read -r -a SEED_ARR <<< "$SEEDS_ENV"

# ---------- Run ----------
set -x
srun python -u "single_reservoir_baseline.py" \
  --ga \
  --jobs "$JOBS" \
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