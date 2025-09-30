# Mixture-of-Reservoirs Grid Search

This repository runs echo state network experiments. The mixture-of-reservoirs entry point (`mixture_of_reservoirs.py`) trains a mixture of three ESN experts using an EM loop paired with a Dask-driven hyperparameter grid search. The helper modules live under `moe/`.


## Mixture-of-Reservoirs Workflow
- `mixture_of_reservoirs.py` orchestrates the run: parse CLI (`--tag`, `--outdir`, `--iterations`), stamp the run directory, snapshot config, then loop over EM updates and Dask-backed grid searches for `iterations` passes.
- Window generation and regime slicing live in `moe/data_windows.py`; the training loop imports `_build_fixed_windows` so one change updates every iteration.
- Reservoir defaults and deterministic seeding are defined in `moe/reservoir_bank.py`. The main script clones these configs each iteration before replacements.
- Readout EM helpers in `moe/readout_em.py` handle teacher-forced state capture, unweighted initialization, weighted refits, and the responsibility update via `run_one_em_round`.
- Gating math (NRMSE, temperature softmax, load balancing, epsilon smoothing, EMA) resides in `moe/gating.py`, keeping annealing logic independent of the trainer.
- Hyperparameter sweeps run through `moe/hparam_search.py`. Candidates are sampled from interpolation grids, evaluated on Dask (`reservoir_eval` resource), and the best per expert is returned for the next iteration.
- Dask orchestration is centralized in `moe/dask_utils.py`; it constructs a `SLURMCluster` when `SLURM_JOB_ID` is present and mirrors the `ESN_DASK_*` env settings expected by `moe_esn_batch.sh`.
- Logging helpers in `moe/logging.py` provide `timestamp` and `write_json` so the trainer writes `config.json` and `iteration_log.jsonl` inside each `runs/<tag>/` directory.
