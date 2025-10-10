"""Local Mackey-Glass GA search leveraging shared ESN optimization utilities."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataset import Regime, generate_mackey_glass
from moe.genetic_search import (
    ENERGY_EPS,
    ENERGY_PENALTY_MAX,
    ENERGY_PENALTY_MIN,
    GASettings,
    GeneticReservoirOptimizer,
    _evaluate_reservoir_candidate,
)
from moe.gating import compute_nrmse
from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import fit_linear_readout, teacher_forced_states

# Reservoir dimensions (mirrors local_mackey_grid)
INPUT_DIM = 1
OUTPUT_DIM = 1
RESERVOIR_SIZE = 400

# Window specification
WARMUP_LEN = 500
TEACHER_FORCED_LEN = 3000
FREE_RUN_HORIZON = 500
NRMSE_HORIZON = 100
TOTAL_WINDOW_LEN = WARMUP_LEN + TEACHER_FORCED_LEN + FREE_RUN_HORIZON

# Training defaults
RIDGE_ALPHA = 1e-4
DATA_SEED = 31415
BASE_RESERVOIR_SEED = 9000
GA_SEED = 2024

# GA configuration (formerly CLI flags)
POPULATION_SIZE = 500
TOTAL_GENERATIONS = 30
EXPLORATION_GENERATIONS = 5
ELITE_FRACTION = 0.10
RANDOM_START = 0.20
RANDOM_END = 0.05
MUTATION_SCALE_START = 1.0
MUTATION_SCALE_END = 0.30
TOURNAMENT_SIZE = 4

MAX_WORKERS = max(1, min(14, os.cpu_count() or 1))
PLOT_THRESHOLD = float("inf")


def build_window(series: np.ndarray) -> Dict:
    y = series.reshape(-1, 1).astype(np.float32)
    return {
        "y": y,
        "idx_warmup_end": WARMUP_LEN,
        "idx_fit_end": WARMUP_LEN + TEACHER_FORCED_LEN,
        "idx_eval_end": TOTAL_WINDOW_LEN,
        "regime": int(Regime.MACKEY_GLASS),
        "id": "mackey_glass_window",
    }


def format_config_summary(params: ReservoirParams) -> str:
    return (
        f"rho={params.spectral_radius:.2f}, "
        f"C={params.C:.2f}, "
        f"leak={params.decay_rate:.2f}, "
        f"w={params.w_scale:.2f}, "
        f"w_in={params.w_in_scale:.2f}, "
        f"w_back={params.w_back_scale:.2f}, "
        f"sparsity={params.w_sparsity:.3f}"
    )


def make_ga_settings() -> GASettings:
    return GASettings(
        population_size=POPULATION_SIZE,
        exploration_generations=EXPLORATION_GENERATIONS,
        elite_fraction=ELITE_FRACTION,
        random_injection_start=RANDOM_START,
        random_injection_end=RANDOM_END,
        mutation_scale_start=MUTATION_SCALE_START,
        mutation_scale_end=MUTATION_SCALE_END,
        total_generations=TOTAL_GENERATIONS,
        rng_seed=GA_SEED,
        tournament_size=TOURNAMENT_SIZE,
    )


def worker_evaluate(task: Tuple[ReservoirParams, Dict, np.ndarray, float, int, int]) -> float:
    params, window, responsibility_column, ridge_alpha, nrmse_horizon, seed = task
    return _evaluate_reservoir_candidate(
        reservoir_index=0,
        params=params,
        windows=[window],
        responsibility_column=responsibility_column,
        lam=ridge_alpha,
        horizon=nrmse_horizon,
        rng_seed=seed,
        N=RESERVOIR_SIZE,
        K=INPUT_DIM,
        L=OUTPUT_DIM,
    )


def compute_plot_diagnostics(params: ReservoirParams,
                              window: Dict,
                              ridge_alpha: float,
                              free_run_horizon: int,
                              nrmse_horizon: int,
                              reservoir_seed: int) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(reservoir_seed)
    reservoir = Reservoir(RESERVOIR_SIZE, INPUT_DIM, OUTPUT_DIM, params, rng)

    targets = window["y"].astype(np.float32)
    states = teacher_forced_states(reservoir, targets.T)
    fit_start = window["idx_warmup_end"]
    fit_end = window["idx_fit_end"]
    design_matrix = states[:, fit_start + 1:fit_end + 1].T.astype(np.float32)
    fit_targets = targets[fit_start:fit_end, :]
    W_out = fit_linear_readout(design_matrix, fit_targets, alpha=ridge_alpha)

    y_hat, y_true = reservoir.free_run(window, W_out, horizon=free_run_horizon)

    y_hat_eval = y_hat[:nrmse_horizon].reshape(-1)
    y_true_eval = y_true[:nrmse_horizon].reshape(-1)
    raw_nrmse = compute_nrmse(y_hat_eval, y_true_eval)
    predicted_energy = float(np.mean((y_hat_eval - y_hat_eval.mean()) ** 2))
    target_energy = float(np.mean((y_true_eval - y_true_eval.mean()) ** 2))
    energy_ratio = target_energy / max(predicted_energy, ENERGY_EPS)
    penalty = float(np.clip(energy_ratio, ENERGY_PENALTY_MIN, ENERGY_PENALTY_MAX))
    penalized = float(raw_nrmse * penalty)

    return penalized, raw_nrmse, penalty, y_hat.reshape(-1), y_true.reshape(-1)


def plot_prediction(params: ReservoirParams,
                    generation_index: int,
                    penalized_nrmse: float,
                    raw_nrmse: float,
                    energy_penalty: float,
                    y_hat: np.ndarray,
                    y_true: np.ndarray,
                    free_run_horizon: int,
                    nrmse_horizon: int) -> None:
    time_axis = np.arange(free_run_horizon)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, y_true, label="target", linewidth=1.2)
    ax.plot(time_axis, y_hat, label="free-run", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(
        f"Generation {generation_index + 1} | {format_config_summary(params)}\n"
        f"NRMSE@{nrmse_horizon} = {raw_nrmse:.4f} | penalty={energy_penalty:.2f} | penalized={penalized_nrmse:.4f}"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    ga_settings = make_ga_settings()

    data_rng = np.random.default_rng(DATA_SEED)
    _, series_std = generate_mackey_glass(H=TOTAL_WINDOW_LEN, rng=data_rng)
    window = build_window(series_std)
    responsibility_column = np.ones((1,), dtype=np.float32)

    optimizer = GeneticReservoirOptimizer(num_experts=1, settings=ga_settings)
    state = optimizer.states[0]

    best_overall_score = float("inf")
    best_overall_params: Optional[ReservoirParams] = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for generation_index in range(ga_settings.total_generations):
            population = optimizer._ensure_population(state)
            pop_size = len(population)
            seeds = [BASE_RESERVOIR_SEED + generation_index * pop_size + idx for idx in range(pop_size)]

            tasks = [
                (
                    population[idx],
                    window,
                    responsibility_column,
                    RIDGE_ALPHA,
                    NRMSE_HORIZON,
                    seeds[idx],
                )
                for idx in range(pop_size)
            ]
            scores_iter = executor.map(worker_evaluate, tasks, chunksize=1)
            errors = np.fromiter(scores_iter, dtype=np.float64, count=pop_size)

            best_idx = int(np.argmin(errors))
            best_error = float(errors[best_idx])
            median_error = float(np.median(errors))
            best_params = population[best_idx]

            if best_error < best_overall_score:
                best_overall_score = best_error
                best_overall_params = ReservoirParams(**vars(best_params))

            random_fraction, mutation_sigmas = optimizer._prepare_next_generation(state, population, errors)
            state.generation += 1

            mutation_sigma_mean = float(np.mean(list(mutation_sigmas.values()))) if mutation_sigmas else 0.0
            print(
                f"[Gen {generation_index + 1}/{ga_settings.total_generations}] "
                f"penalized NRMSE={best_error:.4f} | median={median_error:.4f} | "
                f"random_frac={random_fraction:.3f} | mutation_sigma_mean={mutation_sigma_mean:.5f}"
            )
            print(f"  Best params: {format_config_summary(best_params)}")

            best_seed = seeds[best_idx]
            penalized, raw_nrmse, penalty, y_hat, y_true = compute_plot_diagnostics(
                best_params,
                window,
                RIDGE_ALPHA,
                FREE_RUN_HORIZON,
                NRMSE_HORIZON,
                best_seed,
            )
            print(f"  Penalized score: {penalized:.4f} (raw={raw_nrmse:.4f}, penalty={penalty:.2f})")

            if penalized <= PLOT_THRESHOLD:
                plot_prediction(
                    best_params,
                    generation_index,
                    penalized,
                    raw_nrmse,
                    penalty,
                    y_hat,
                    y_true,
                    FREE_RUN_HORIZON,
                    NRMSE_HORIZON,
                )
            else:
                print(
                    f"  Skipping plot; penalized NRMSE {penalized:.4f} exceeds threshold {PLOT_THRESHOLD:.4f}."
                )

    if best_overall_params is not None:
        print(
            f"Best overall penalized NRMSE@{NRMSE_HORIZON}: {best_overall_score:.4f} | "
            f"params: {format_config_summary(best_overall_params)}"
        )


if __name__ == "__main__":
    main()
