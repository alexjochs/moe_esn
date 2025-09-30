import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import teacher_forced_states, fit_linear_readout
from moe.gating import compute_nrmse


def _candidate_param_values(base: float, span: Tuple[float, float], interpolation_points: int, clip_to_unit: bool = False) -> np.ndarray:
    lo, hi = span
    values = np.linspace(lo, hi, interpolation_points, dtype=np.float64)
    if clip_to_unit:
        values = np.clip(values, 0.0, 1.0)
    return values


def _generate_param_candidates(base_params: ReservoirParams,
                               interpolation_points: int,
                               sample_budget: int,
                               rng_seed: Optional[int] = None) -> List[ReservoirParams]:
    """Build candidate list by sampling from the per-parameter interpolation grids."""
    rng_local = np.random.default_rng(rng_seed)

    grid_axes = [
        _candidate_param_values(base_params.spectral_radius, (0.5, 1.2), interpolation_points),
        _candidate_param_values(base_params.C, (0.4, 1.0), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.decay_rate, (0.6, 0.99), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.w_scale, (0.15, 0.5), interpolation_points),
        _candidate_param_values(base_params.w_sparsity, (0.94, 0.995), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.w_back_scale, (0.2, 0.8), interpolation_points),
        _candidate_param_values(base_params.w_in_scale, (0.02, 0.12), interpolation_points),
        _candidate_param_values(base_params.w_in_sparsity, (0.3, 0.9), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.bias_value, (-0.2, 0.2), interpolation_points),
    ]

    total_combinations = int(np.prod([axis.size for axis in grid_axes], dtype=np.int64))

    def candidate_from_tuple(values: Tuple[float, ...]) -> ReservoirParams:
        return ReservoirParams(
            spectral_radius=float(values[0]),
            C=float(values[1]),
            decay_rate=float(values[2]),
            w_scale=float(values[3]),
            w_sparsity=float(values[4]),
            w_back_scale=float(values[5]),
            w_in_scale=float(values[6]),
            w_in_sparsity=float(values[7]),
            bias_value=float(values[8]),
        )

    samples: Dict[Tuple[float, ...], ReservoirParams] = {}

    if total_combinations <= sample_budget:
        for combo in itertools.product(*grid_axes):
            samples.setdefault(tuple(round(value, 6) for value in combo), candidate_from_tuple(combo))
    else:
        while len(samples) < sample_budget:
            combo = tuple(axis[rng_local.integers(axis.size)] for axis in grid_axes)
            rounded = tuple(round(value, 6) for value in combo)
            if rounded not in samples:
                samples[rounded] = candidate_from_tuple(combo)

    baseline_tuple = (
        round(base_params.spectral_radius, 6),
        round(base_params.C, 6),
        round(base_params.decay_rate, 6),
        round(base_params.w_scale, 6),
        round(base_params.w_sparsity, 6),
        round(base_params.w_back_scale, 6),
        round(base_params.w_in_scale, 6),
        round(base_params.w_in_sparsity, 6),
        round(base_params.bias_value, 6),
    )
    samples.setdefault(baseline_tuple, ReservoirParams(**vars(base_params)))

    return list(samples.values())


def _evaluate_reservoir_candidate(reservoir_index: int,
                                  params: ReservoirParams,
                                  windows: List[Dict],
                                  responsibility_column: np.ndarray,
                                  lam: float,
                                  horizon: int,
                                  rng_seed: int,
                                  N: int,
                                  K: int,
                                  L: int) -> float:
    """Evaluate a single reservoir candidate and return responsibility-weighted NRMSE."""
    candidate_rng = np.random.default_rng(rng_seed)
    candidate_reservoir = Reservoir(N, K, L, params, candidate_rng)

    active_indices = np.where(responsibility_column > 1e-6)[0]

    design_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []
    sample_weight_blocks: List[np.ndarray] = []

    for window_index in active_indices:
        weight = float(responsibility_column[window_index])
        window = windows[window_index]
        targets = window['y'].astype(np.float32)
        states = teacher_forced_states(candidate_reservoir, targets.T)
        idx_warmup_end = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']
        X_fit = states[:, idx_warmup_end + 1:idx_fit_end + 1].T.astype(np.float32)
        Y_fit = targets[idx_warmup_end:idx_fit_end, :].astype(np.float32)
        design_blocks.append(X_fit)
        target_blocks.append(Y_fit)
        sample_weight_blocks.append(np.full(X_fit.shape[0], weight, dtype=np.float32))

    design_matrix = np.vstack(design_blocks)
    target_matrix = np.vstack(target_blocks)
    sample_weights = np.concatenate(sample_weight_blocks)
    W_out = fit_linear_readout(design_matrix, target_matrix, alpha=lam, sample_weight=sample_weights)

    errors: List[float] = []
    error_weights: List[float] = []
    for window_index in active_indices:
        weight = float(responsibility_column[window_index])
        window = windows[window_index]
        y_hat, y_true = candidate_reservoir.free_run(window, W_out, horizon=horizon)
        error_value = compute_nrmse(y_hat, y_true)
        errors.append(error_value)
        error_weights.append(weight)

    weighted_error = float(np.average(errors, weights=error_weights))
    return weighted_error


def em_round_hyperparam_tuning(reservoirs: List[Reservoir],
                               windows: List[Dict],
                               responsibilities: np.ndarray,
                               lam: float,
                               horizon: int,
                               interpolation_points: int,
                               candidate_samples: int,
                               client: "Client",
                               reservoir_seeds: List[int],
                               N: int,
                               K: int,
                               L: int,
                               task_retries: int,
                               dask_jobs: int,
                               dask_processes_per_job: int) -> Tuple[List[ReservoirParams], List[float]]:
    """Sampled grid-search sweep over reservoir hyperparameters using responsibility weights."""
    tuned_params: List[ReservoirParams] = []
    tuned_errors: List[float] = []

    sample_budget = max(1, candidate_samples)

    worker_threads = client.nthreads()

    if worker_threads:
        active_workers = len(worker_threads)
        total_threads = int(sum(worker_threads.values()))
        expected_workers = max(1, dask_jobs * dask_processes_per_job)
        print(f"[Dask] Active workers: {active_workers}/{expected_workers} | total threads={total_threads}")

    windows_future = client.scatter(windows, broadcast=True)

    for reservoir_index, reservoir in enumerate(reservoirs):
        base_params = reservoir.params
        candidate_seed = reservoir_seeds[reservoir_index % len(reservoir_seeds)] + 991 * (reservoir_index + 1)
        candidate_params = _generate_param_candidates(
            base_params,
            interpolation_points=interpolation_points,
            sample_budget=sample_budget,
            rng_seed=candidate_seed,
        )
        responsibility_vector = responsibilities[:, reservoir_index].astype(np.float32)
        responsibilities_future = client.scatter(responsibility_vector, broadcast=True)

        futures = []
        for candidate_idx, candidate in enumerate(candidate_params):
            seed = reservoir_seeds[reservoir_index % len(reservoir_seeds)] + (candidate_idx + 1) * 17
            future = client.submit(
                _evaluate_reservoir_candidate,
                reservoir_index,
                candidate,
                windows_future,
                responsibilities_future,
                lam,
                horizon,
                seed,
                N,
                K,
                L,
                resources={"reservoir_eval": 1},
                retries=task_retries,
                pure=False,
            )
            futures.append(future)

        print(f"Expert {reservoir_index}: evaluating {len(candidate_params)} candidates on Dask")
        candidate_errors = client.gather(futures)
        best_index = int(np.argmin(np.array(candidate_errors, dtype=float)))
        tuned_params.append(candidate_params[best_index])
        tuned_errors.append(float(candidate_errors[best_index]))

        client.cancel(futures, force=True)
        client.cancel(responsibilities_future)

    client.cancel(windows_future)

    return tuned_params, tuned_errors


__all__ = [
    "_candidate_param_values",
    "_generate_param_candidates",
    "_evaluate_reservoir_candidate",
    "em_round_hyperparam_tuning",
]
