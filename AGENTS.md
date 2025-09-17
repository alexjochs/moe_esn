# Repository Guidelines

## Project Scope & Layout
- This repository houses experiments on echo state networks (ESNs) and their architectural variants (`single_reservoir_baseline.py`, `mixture_of_reservoirs.py`, etc.).
- Shared data-generation and reservoir utilities sit at the repo root (`dataset.py`, `reservoir.py`); changes here ripple through multiple experiments.
- HPC orchestration scripts such as `ga_single_batchfile.sh` mirror the Python CLI interface and should be updated in lockstep with code changes.
- Generated artifacts default to `runs/<tag>/`; keep intermediate logs and genomes inside those folders for reproducibility.

## Development Workflow
- `python -m venv venv && source venv/bin/activate`: bootstrap the isolated environment recommended for all work.
- `python -m pip install -r requirements.txt`: install runtime and Dask dependencies.
- `python single_reservoir_baseline.py --ga --no-plots ...`: run local GA experiments; align flags with the corresponding batch script when sanity-checking a change.
- `python -m compileall single_reservoir_baseline.py`: quick syntax check before launching long HPC jobs.
- For smoke validation, run a tiny GA (`--pop 5 --gens 1 --seeds 42`) and inspect the emitted metrics.

## Coding Style & Readability
- Python follows 4-space indentation, snake_case identifiers, and CapWords for classes; match surrounding style.
- Prioritize readability over cleverness—document rationale for experimental knobs, cite papers when relevant, and note assumptions about data scales or ESN hyperparameters.
- Keep comments focused on non-obvious math, HPC coordination steps, or experimental protocol; avoid restating simple code.
- Do NOT use single letter variable names unless the variable is explicitly representing a core Echo State Network math equation. Do NOT use "e" when you
could use "segment_end" instead. It is fine for variables that follow the representations of Echo State Network math in the literature to have names like W, or X_n. 

## Experimentation Practices
- Record significant runs by saving CLI invocations (e.g., append to `runs/<tag>/command.txt`) and attaching the resulting `best_genome*.json` files.
- When updating genome parameter ranges or dataset regimes, describe the intent and expected behavior shift in the commit body.
- Maintain deterministic seeds for comparative studies; introduce randomness intentionally and call it out in notes.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood and scope them around a single experimental or infrastructure change (e.g., "Tune GA chunk sizing for HPC").
- Include concise summaries of observed outcomes or validation runs in commit messages or PR descriptions.
- PRs should highlight any required batch-script adjustments, resource implications, or new configuration flags.

## HPC & Distributed Runs
- Use Dask worker processes (not nested thread pools) for GA parallelism; align `--jobs` and `--dask-worker-cores` with the current population size to avoid idle allocations.
- Document any cluster-specific environment modules or scheduler quirks directly in the relevant batch scripts.
