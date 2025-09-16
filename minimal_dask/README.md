# Minimal Dask on Slurm smoke test

This directory holds a tiny harness to verify that Dask + `dask-jobqueue`
cluster submission works on the HPC system.

## Usage

1. Create or update the virtual environment (once per filesystem)
   ```bash
   ./setup_venv.sh
   ```
2. Submit the smoke test job
   ```bash
   sbatch submit_dask_smoke.sh
   ```

The job will launch a `SLURMCluster`, request a couple of worker jobs on the
`share` partition (defaults can be overridden via environment variables), and
print scheduler/worker information to the job's stdout file.  If recursive
`sbatch` calls are disabled you should see an informative failure in that log.
