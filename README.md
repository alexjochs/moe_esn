# Mixture-of-Reservoirs Genetic Search

This repository runs echo state network experiments. The mixture-of-reservoirs entry point (`mixture_of_reservoirs.py`) trains a mixture of three ESN experts using an EM loop paired with a Dask-driven genetic hyperparameter search. The helper modules live under `moe/`.


## Mixture-of-Reservoirs Workflow
- `mixture_of_reservoirs.py` orchestrates the run: parse CLI (`--tag`, `--outdir`, `--iterations`), stamp the run directory, snapshot config, then loop over EM updates and Dask-backed GA sweeps for `iterations` passes.
- Window generation and regime slicing live in `moe/data_windows.py`; the training loop imports `_build_fixed_windows` so one change updates every iteration.
- Reservoir defaults and deterministic seeding are defined in `moe/reservoir_bank.py`. The main script clones these configs each iteration before replacements.
- Readout EM helpers in `moe/readout_em.py` handle teacher-forced state capture, unweighted initialization, weighted refits, and the responsibility update via `run_one_em_round`.
- Gating math (NRMSE, temperature softmax, load balancing, epsilon smoothing, EMA) resides in `moe/gating.py`, keeping annealing logic independent of the trainer.
- Genetic hyperparameter sweeps run through `moe/genetic_search.py`. Populations start with uniform exploration, then anneal toward crossover plus per-parameter Gaussian mutations. Evaluations are dispatched on Dask (`reservoir_eval` resource), and the champion per expert seeds the next EM round.
- Dask orchestration is centralized in `moe/dask_utils.py`; it constructs a `SLURMCluster` when `SLURM_JOB_ID` is present and mirrors the `ESN_DASK_*` env settings expected by `moe_esn_batch.sh`.
- Logging helpers in `moe/logging.py` provide `timestamp` and `write_json` so the trainer writes `config.json` and `iteration_log.jsonl` inside each `runs/<tag>/` directory.


• How The GA Runs

  - Every EM iteration calls GeneticReservoirOptimizer.iterate, which evaluates the full per‑expert population on Dask, scores candidates with responsibility‑weighted NRMSE, and returns the best hyperparameters for the next round (mixture_of_reservoirs.py:240-253, moe/
    genetic_search.py:187-217).
  - Populations start as uniform samples over each parameter’s bounds; while the current generation index is inside the declared exploration window, the next generation is re-drawn uniformly (moe/genetic_search.py:159-201, moe/genetic_search.py:238-248).
  - After exploration, the optimizer keeps elites, adds crossover children, applies Gaussian mutations with per-parameter sigmas, and injects a configurable fraction of fresh random genomes (moe/genetic_search.py:250-287).
  - Mutation widths track the parameter ranges—fractions are converted into absolute sigmas and annealed toward smaller steps across generations (moe/genetic_search.py:16-52, moe/genetic_search.py:161-185, moe/genetic_search.py:265-283).
  - Logged metrics (best/median error, current random-injection fraction, mean mutation sigma) land in each iteration record and console print so you can monitor the GA’s phase and progress (mixture_of_reservoirs.py:255-268, moe/genetic_search.py:216-224).

  Key Knobs & Tuning Tips

  - Population size (--ga-population) controls how many candidates you evaluate per expert each iteration; larger values boost coverage but increase Dask load (mixture_of_reservoirs.py:115-129, moe/genetic_search.py:156-159).
  - Exploration generations (--ga-explore-generations) set how many rounds stay in pure uniform sampling; extend if early results look brittle or error histograms remain high variance (mixture_of_reservoirs.py:115-129, moe/genetic_search.py:159-166, moe/genetic_search.py:238-248).
  - Elite fraction (--ga-elite-fraction) determines how much of the best population survives untouched; higher keeps stability but can slow exploration (mixture_of_reservoirs.py:115-129, moe/genetic_search.py:160-167, moe/genetic_search.py:265-275).
  - Random injection start/end (--ga-random-frac-start, --ga-random-frac-end) taper the number of fresh uniform draws each generation; watch the logged random_frac to ensure it decays but never strands the search (mixture_of_reservoirs.py:120-128, moe/genetic_search.py:161-167,
    moe/genetic_search.py:265-282).
  - Mutation scale start/end (--ga-mutation-scale-start, --ga-mutation-scale-end) globally scale the per-parameter sigmas; if the GA stalls, raise the start or the end floor so mutations stay lively (mixture_of_reservoirs.py:125-128, moe/genetic_search.py:163-185, moe/
    genetic_search.py:265-283).
  - Tournament size (--ga-tournament-size) adjusts selection pressure; larger tournaments favor the best genomes more aggressively (mixture_of_reservoirs.py:129-129, moe/genetic_search.py:167-168, moe/genetic_search.py:284-290).
  - RNG seed is fixed in GASettings so runs stay reproducible unless you override it (mixture_of_reservoirs.py:209-224, moe/genetic_search.py:170-176).

  What To Watch

  - Console output prints per-expert best/median errors and the current random fraction; track these to confirm the annealing trend and that error medians drop generation to generation (mixture_of_reservoirs.py:255-263).
  - iteration_log.jsonl now contains ga_metrics; plotting best_error vs iteration or random_fraction vs iteration is an easy diagnostics pass (mixture_of_reservoirs.py:262-268).
  - If mutation sigmas shrink too quickly (mean approaches zero early), either increase --ga-mutation-scale-end or lengthen exploration so the GA enters exploitation with stronger candidates.
  - If elites dominate and diversity collapses (high best_error but zero improvement), bump --ga-random-frac-start or reduce elite fraction to keep turnover up.

  With these knobs you can lengthen the explore phase, dial how sharply the algorithm anneals, and keep an eye on the logged metrics to steer future tuning.