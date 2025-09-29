# Repository Guidelines

## Project Scope & Layout
- This repository houses experiments on echo state networks (ESNs) and their architectural variants (`single_reservoir_baseline.py`, `mixture_of_reservoirs.py`, etc.).
- Only `ga_single_reservoir.py` relies on a genetic algorithm; the other experiments, including `mixture_of_reservoirs.py`, do not use GA-based optimization.
- Shared data-generation and reservoir utilities sit at the repo root (`dataset.py`, `reservoir.py`); changes here ripple through multiple experiments.
- HPC orchestration scripts such as `ga_single_batchfile.sh` mirror the Python CLI interface and should be updated in lockstep with code changes.
- Generated artifacts default to `runs/<tag>/`; keep intermediate logs and genomes inside those folders for reproducibility.

## Development Workflow
- `python -m venv venv && source venv/bin/activate`: bootstrap the isolated environment recommended for all work.
- Always activate the repo virtualenv (`source venv/bin/activate`) before running Python tooling; automation here runs on macOS, while HPC GA jobs execute on Linux nodes.
- `python -m pip install -r requirements.txt`: install runtime and Dask dependencies.
- For the GA-only entry point `ga_single_reservoir.py`, run `python ga_single_reservoir.py --jobs ...` with every required flag (jobs, Dask resources, seeds, etc.) matching `ga_single_batchfile.sh`.
- `python -m compileall single_reservoir_baseline.py`: quick syntax check before launching long HPC jobs.
- When sanity-checking the GA workflow, run a tiny job (`--pop 5 --gens 1 --seeds 42`) and inspect the emitted metrics.

## Coding Style & Readability
- Python follows 4-space indentation, snake_case identifiers, and CapWords for classes; match surrounding style.
- Prioritize readability over cleverness—document rationale for experimental knobs, cite papers when relevant, and note assumptions about data scales or ESN hyperparameters.
- Keep comments focused on non-obvious math, HPC coordination steps, or experimental protocol; avoid restating simple code.
- Do NOT use single letter variable names unless the variable is explicitly representing a core Echo State Network math equation. Do NOT use "e" when you
could use "segment_end" instead. It is fine for variables that follow the representations of Echo State Network math in the literature to have names like W, or X_n. 

## Experimentation Practices
- Record significant runs by saving CLI invocations (e.g., append to `runs/<tag>/command.txt`); for GA runs, include the resulting `best_genome*.json` artifacts.
- When adjusting GA genome parameter ranges or dataset regimes, describe the intent and expected behavior shift in the commit body.
- Maintain deterministic seeds for comparative studies; introduce randomness intentionally and call it out in notes.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood and scope them around a single experimental or infrastructure change (e.g., "Tune GA chunk sizing for HPC").
- Include concise summaries of observed outcomes or validation runs in commit messages or PR descriptions.
- PRs should highlight any required batch-script adjustments, resource implications, or new configuration flags.

## HPC & Distributed Runs
- For GA population sweeps, use Dask worker processes (not nested thread pools); align `--jobs` and `--dask-worker-cores` with the current population size to avoid idle allocations.
- Leave `dask_chunk_size` unset; the GA launcher now derives chunk sizing internally from the active worker count.
- Document any cluster-specific environment modules or scheduler quirks directly in the relevant batch scripts.

## Dask Orchestration Notes
- `moe_esn_batch.sh` submits a single orchestrator job on `share` with the SBATCH resources; it bootstraps a fresh venv, installs requirements, and exports `ESN_DASK_*` before launching `mixture_of_reservoirs.py`.
- Inside `start_dask_client`, the Python script detects `SLURM_JOB_ID` and constructs a `dask_jobqueue.SLURMCluster` so every worker is a separate SLURM job submitted against `ESN_DASK_PARTITION` (defaults to `preempt`) with its own `--mem` derived from `ESN_DASK_WORKER_MEM`.
- `cluster.scale(jobs=DASK_JOBS)` requests that many worker jobs; effective concurrency is `jobs * ESN_DASK_PROCESSES_PER_JOB`, so each process shares the worker memory reservation.
- `ESN_DASK_REQUEUE` is forced to `1` in `moe_esn_batch.sh`, so every worker submission carries `--requeue`; preempted workers are re-queued automatically, but watch for fast crash loops when a worker fails deterministically.
- The log and scratch paths for workers live under `runs/<tag>/dask_logs` and `runs/<tag>/dask_tmp`, making it easy to inspect worker crashes (OOM, etc.).
- If worker processes are OOM-killed, increase `ESN_DASK_WORKER_MEM` or reduce `ESN_DASK_PROCESSES_PER_JOB`; bumping SBATCH `--mem` only affects the orchestrator node and does not trickle down to worker submissions.
- For local dev (no `SLURM_JOB_ID`), the script instead spins up a `LocalCluster` sized by `DASK_JOBS * DASK_PROCESSES_PER_JOB`, each with one thread per worker.
