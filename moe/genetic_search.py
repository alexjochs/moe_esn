"""Genetic hyperparameter search for mixture-of-reservoirs experts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import fit_linear_readout, teacher_forced_states
from moe.gating import compute_nrmse


PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "spectral_radius": (0.3, 1.2),
    "C": (0.1, 1.0),
    "decay_rate": (0.01, 0.99),
    "w_scale": (0.01, 1.0),
    "w_sparsity": (0.5, 0.995),
    "w_back_scale": (0.1, 1.0),
    "w_in_scale": (0.02, 1.0),
    "w_in_sparsity": (0.1, 0.99),
    "bias_value": (-0.9, 0.9),
}

ENERGY_EPS = 1e-6
ENERGY_PENALTY_MIN = 0.5
ENERGY_PENALTY_MAX = 20.0

_MAX_IN_FLIGHT_ENV = os.environ.get("ESN_MAX_INFLIGHT_EVALS", "1000")
try:
    _parsed = int(_MAX_IN_FLIGHT_ENV)
except ValueError:
    MAX_IN_FLIGHT_EVALS: Optional[int] = 1000
else:
    MAX_IN_FLIGHT_EVALS = _parsed if _parsed > 0 else None

# Per-parameter mutation widths are expressed as a fraction of the allowed range.
MUTATION_FRACTION_START: Dict[str, float] = {
    "spectral_radius": 0.20,
    "C": 0.18,
    "decay_rate": 0.20,
    "w_scale": 0.25,
    "w_sparsity": 0.08,
    "w_back_scale": 0.22,
    "w_in_scale": 0.22,
    "w_in_sparsity": 0.08,
    "bias_value": 0.30,
}

MUTATION_FRACTION_END: Dict[str, float] = {
    "spectral_radius": 0.05,
    "C": 0.04,
    "decay_rate": 0.05,
    "w_scale": 0.07,
    "w_sparsity": 0.02,
    "w_back_scale": 0.06,
    "w_in_scale": 0.08,
    "w_in_sparsity": 0.02,
    "bias_value": 0.08,
}


@dataclass
class GASettings:
    population_size: int
    exploration_generations: int
    elite_fraction: float
    random_injection_start: float
    random_injection_end: float
    mutation_scale_start: float
    mutation_scale_end: float
    total_generations: int
    rng_seed: int
    tournament_size: int = 3


@dataclass
class ExpertState:
    rng: np.random.Generator
    generation: int = 0
    population: List[ReservoirParams] = field(default_factory=list)
    last_random_fraction: float = 0.0
    last_mutation_sigmas: Dict[str, float] = field(default_factory=dict)


def _evaluate_reservoir_candidate(
    reservoir_index: int,
    params: ReservoirParams,
    windows: List[Dict],
    responsibility_column: np.ndarray,
    lam: float,
    horizon: int,
    rng_seed: int,
    N: int,
    K: int,
    L: int,
) -> float:
    """Evaluate a single reservoir candidate and return responsibility-weighted NRMSE."""
    candidate_rng = np.random.default_rng(rng_seed)
    candidate_reservoir = Reservoir(N, K, L, params, candidate_rng)

    active_indices = np.where(responsibility_column > 1e-6)[0]
    if active_indices.size == 0:
        active_indices = np.arange(len(windows))

    design_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []
    sample_weight_blocks: List[np.ndarray] = []

    for window_index in active_indices:
        window = windows[window_index]
        weight = float(responsibility_column[window_index])
        if weight <= 0.0:
            weight = 1.0
        targets = window["y"].astype(np.float32)
        states = teacher_forced_states(candidate_reservoir, targets.T)
        warmup_end = window["idx_warmup_end"]
        fit_end = window["idx_fit_end"]
        X_fit = states[:, warmup_end + 1 : fit_end + 1].T.astype(np.float32)
        Y_fit = targets[warmup_end:fit_end, :].astype(np.float32)
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
        window = windows[window_index]
        weight = float(responsibility_column[window_index])
        if weight <= 0.0:
            weight = 1.0
        y_hat, y_true = candidate_reservoir.free_run(window, W_out, horizon=horizon)
        error_value = compute_nrmse(y_hat, y_true)

        y_hat_flat = y_hat.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        # penalize flat line predictions 
        predicted_energy = float(np.mean((y_hat_flat - y_hat_flat.mean()) ** 2))
        target_energy = float(np.mean((y_true_flat - y_true_flat.mean()) ** 2))
        energy_ratio = target_energy / max(predicted_energy, ENERGY_EPS)
        penalty = float(np.clip(energy_ratio, ENERGY_PENALTY_MIN, ENERGY_PENALTY_MAX))
        error_value *= penalty

        errors.append(error_value)
        error_weights.append(weight)

    if not errors:
        return float("inf")

    error_weights = np.asarray(error_weights, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if np.all(error_weights <= 0.0):
        return float(errors.mean())
    return float(np.average(errors, weights=error_weights))


class GeneticReservoirOptimizer:
    """Maintain per-expert GA populations and evolve them across EM iterations."""

    def __init__(
        self,
        num_experts: int,
        settings: GASettings,
        param_bounds: Dict[str, Tuple[float, float]] = PARAM_BOUNDS,
        mutation_fraction_start: Dict[str, float] = MUTATION_FRACTION_START,
        mutation_fraction_end: Dict[str, float] = MUTATION_FRACTION_END,
    ) -> None:
        self.num_experts = num_experts
        self.settings = settings
        self.param_bounds = param_bounds
        self.population_size = max(1, settings.population_size)
        self.exploration_generations = max(0, settings.exploration_generations)
        self.elite_fraction = np.clip(settings.elite_fraction, 0.0, 1.0)
        self.random_start = np.clip(settings.random_injection_start, 0.0, 1.0)
        self.random_end = np.clip(settings.random_injection_end, 0.0, 1.0)
        self.mutation_fraction_start = mutation_fraction_start
        self.mutation_fraction_end = mutation_fraction_end
        self.mutation_scale_start = max(settings.mutation_scale_start, 0.0)
        self.mutation_scale_end = max(settings.mutation_scale_end, 0.0)
        self.total_generations = max(1, settings.total_generations)
        self.tournament_size = max(2, settings.tournament_size)

        base_rng = np.random.default_rng(settings.rng_seed)
        self.states: List[ExpertState] = []
        for expert_index in range(num_experts):
            state_seed = base_rng.integers(0, np.iinfo(np.int64).max)
            self.states.append(
                ExpertState(rng=np.random.default_rng(int(state_seed)))
            )

        if MAX_IN_FLIGHT_EVALS is None:
            self.max_inflight_evaluations = None
        else:
            self.max_inflight_evaluations = max(
                1,
                min(self.population_size, MAX_IN_FLIGHT_EVALS),
            )

        self.mutation_sigma_start = {
            name: (bounds[1] - bounds[0]) * mutation_fraction_start[name]
            for name, bounds in param_bounds.items()
        }
        self.mutation_sigma_end = {
            name: (bounds[1] - bounds[0]) * mutation_fraction_end[name]
            for name, bounds in param_bounds.items()
        }

    def iterate(
        self,
        client,
        windows: List[Dict],
        responsibilities: np.ndarray,
        lam: float,
        horizon: int,
        reservoir_seeds: Sequence[int],
        N: int,
        K: int,
        L: int,
        task_retries: int,
    ) -> Tuple[List[ReservoirParams], List[Dict[str, float]]]:
        """Advance every expert population by one GA evaluation round."""
        windows_future = client.scatter(windows, broadcast=True)
        best_params: List[ReservoirParams] = []
        metrics: List[Dict[str, float]] = []

        for expert_index in range(self.num_experts):
            responsibility_vector = responsibilities[:, expert_index].astype(np.float32)
            responsibility_future = client.scatter(responsibility_vector, broadcast=True)
            params, info = self._evaluate_expert(
                expert_index,
                client,
                windows_future,
                responsibility_future,
                lam,
                horizon,
                reservoir_seeds,
                N,
                K,
                L,
                task_retries,
            )
            client.cancel(responsibility_future)
            best_params.append(params)
            metrics.append(info)

        client.cancel(windows_future)
        return best_params, metrics

    def _evaluate_expert(
        self,
        expert_index: int,
        client,
        windows_future,
        responsibility_future,
        lam: float,
        horizon: int,
        reservoir_seeds: Sequence[int],
        N: int,
        K: int,
        L: int,
        task_retries: int,
    ) -> Tuple[ReservoirParams, Dict[str, float]]:
        state = self.states[expert_index]
        population = self._ensure_population(state)

        reservoir_seed = reservoir_seeds[expert_index % len(reservoir_seeds)]
        population_size = len(population)
        chunk_limit = self.max_inflight_evaluations or population_size
        chunk_size = max(1, min(population_size, chunk_limit))
        candidate_errors_buffer = np.empty(population_size, dtype=np.float64)

        for start in range(0, population_size, chunk_size):
            stop = min(population_size, start + chunk_size)
            chunk_futures = []
            for candidate_index in range(start, stop):
                candidate = population[candidate_index]
                eval_seed = (
                    reservoir_seed
                    + 7919 * (state.generation + 1)
                    + 217 * candidate_index
                )
                chunk_futures.append(
                    client.submit(
                        _evaluate_reservoir_candidate,
                        expert_index,
                        candidate,
                        windows_future,
                        responsibility_future,
                        lam,
                        horizon,
                        int(eval_seed),
                        N,
                        K,
                        L,
                        resources={"reservoir_eval": 1},
                        retries=task_retries,
                        pure=False,
                    )
                )
            try:
                errors_chunk = client.gather(chunk_futures)
            finally:
                client.cancel(chunk_futures, force=True)
            candidate_errors_buffer[start:stop] = errors_chunk

        candidate_errors = candidate_errors_buffer
        best_index = int(np.argmin(candidate_errors))
        best_error = float(candidate_errors[best_index])
        median_error = float(np.median(candidate_errors))

        random_fraction, sigma_values = self._prepare_next_generation(
            state,
            population,
            candidate_errors,
        )
        best_params = population[best_index]

        metrics = {
            "generation": float(state.generation),
            "best_error": best_error,
            "median_error": median_error,
            "random_fraction": float(random_fraction),
            "mutation_sigma_mean": float(np.mean(list(sigma_values.values()))) if sigma_values else 0.0,
            "population_size": float(len(population)),
        }
        state.generation += 1
        return best_params, metrics

    def _ensure_population(self, state: ExpertState) -> List[ReservoirParams]:
        if not state.population:
            state.population = [
                self._sample_uniform(state.rng) for _ in range(self.population_size)
            ]
        return state.population

    def _prepare_next_generation(
        self,
        state: ExpertState,
        population: List[ReservoirParams],
        fitness: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        next_generation_index = state.generation + 1
        best_index = int(np.argmin(fitness)) if len(population) > 0 else 0
        best_candidate = population[best_index] if population else None

        if next_generation_index <= self.exploration_generations:
            # Preserve the strongest candidate even while resampling the rest.
            preserved_best = (
                ReservoirParams(**vars(best_candidate))
                if best_candidate is not None
                else self._sample_uniform(state.rng)
            )
            random_count = max(0, self.population_size - 1)
            random_members = [
                self._sample_uniform(state.rng) for _ in range(random_count)
            ]
            state.population = [preserved_best, *random_members]
            if len(state.population) < self.population_size:
                state.population.extend(
                    self._sample_uniform(state.rng)
                    for _ in range(self.population_size - len(state.population))
                )
            state.last_random_fraction = (
                0.0 if self.population_size <= 1 else (self.population_size - 1) / self.population_size
            )
            state.last_mutation_sigmas = {}
            return state.last_random_fraction, state.last_mutation_sigmas

        progress = self._anneal_progress(next_generation_index - self.exploration_generations)
        random_fraction = self._interp(self.random_start, self.random_end, progress)
        mutation_scale = self._interp(
            self.mutation_scale_start, self.mutation_scale_end, progress
        )
        mutation_sigmas = {
            name: self._interp(
                self.mutation_sigma_start[name],
                self.mutation_sigma_end[name],
                progress,
            )
            * mutation_scale
            for name in self.param_bounds.keys()
        }

        elite_count = max(1, int(math.ceil(self.population_size * self.elite_fraction)))
        sorted_indices = np.argsort(fitness)
        elites = [population[idx] for idx in sorted_indices[:elite_count]]

        next_population: List[ReservoirParams] = list(elites)
        random_target = int(self.population_size * random_fraction)

        # Fill with offspring.
        while len(next_population) < self.population_size - random_target:
            parent_a = self._select_parent(population, fitness, state.rng)
            parent_b = self._select_parent(population, fitness, state.rng)
            child = self._crossover(parent_a, parent_b, state.rng)
            child = self._mutate(child, mutation_sigmas, state.rng)
            next_population.append(child)

        while len(next_population) < self.population_size:
            next_population.append(self._sample_uniform(state.rng))

        state.population = next_population
        state.last_random_fraction = float(random_fraction)
        state.last_mutation_sigmas = mutation_sigmas
        return state.last_random_fraction, state.last_mutation_sigmas

    def _anneal_progress(self, completed_generations: int) -> float:
        denom = max(1, self.total_generations - self.exploration_generations)
        return float(np.clip(completed_generations / denom, 0.0, 1.0))

    @staticmethod
    def _interp(start: float, end: float, t: float) -> float:
        return (1.0 - t) * start + t * end

    def _sample_uniform(self, rng: np.random.Generator) -> ReservoirParams:
        values = {
            name: float(rng.uniform(bounds[0], bounds[1]))
            for name, bounds in self.param_bounds.items()
        }
        return ReservoirParams(**values)

    def _mutate(
        self,
        params: ReservoirParams,
        sigmas: Dict[str, float],
        rng: np.random.Generator,
    ) -> ReservoirParams:
        mutated = {}
        for name, bounds in self.param_bounds.items():
            base_value = getattr(params, name)
            sigma = sigmas.get(name, 0.0)
            mutated_value = float(rng.normal(base_value, sigma))
            mutated_value = float(np.clip(mutated_value, bounds[0], bounds[1]))
            mutated[name] = mutated_value
        return ReservoirParams(**mutated)

    def _crossover(
        self,
        parent_a: ReservoirParams,
        parent_b: ReservoirParams,
        rng: np.random.Generator,
    ) -> ReservoirParams:
        child = {}
        for name in self.param_bounds.keys():
            child[name] = (
                getattr(parent_a, name)
                if rng.random() < 0.5
                else getattr(parent_b, name)
            )
        return ReservoirParams(**child)

    def _select_parent(
        self,
        population: List[ReservoirParams],
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> ReservoirParams:
        indices = rng.integers(len(population), size=self.tournament_size)
        best_idx = indices[0]
        best_fitness = fitness[best_idx]
        for idx in indices[1:]:
            if fitness[idx] < best_fitness:
                best_idx = idx
                best_fitness = fitness[idx]
        return population[int(best_idx)]


__all__ = [
    "GASettings",
    "GeneticReservoirOptimizer",
    "PARAM_BOUNDS",
]
