from typing import List, Optional, Sequence, Tuple

import numpy as np

from reservoir import Reservoir, ReservoirParams


RESERVOIR_SEEDS = [12345, 22345, 32345]

RESERVOIR_PARAM_DEFAULTS: List[ReservoirParams] = [
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
        bias_value=0.0,
    ),
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
        bias_value=0.0,
    ),
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
        bias_value=0.0,
    ),
]


def _instantiate_reservoirs(param_list: List[ReservoirParams],
                            N: int,
                            K: int,
                            L: int,
                            seeds: Optional[Sequence[int]] = None) -> List[Reservoir]:
    instantiated: List[Reservoir] = []
    for expert_index, params in enumerate(param_list):
        if seeds is not None and expert_index < len(seeds):
            seed = int(seeds[expert_index])
        else:
            seed = RESERVOIR_SEEDS[expert_index % len(RESERVOIR_SEEDS)] + expert_index * 1000
        expert_rng = np.random.default_rng(seed)
        instantiated.append(Reservoir(N, K, L, params, expert_rng))
    return instantiated


def reset_reservoir_bank(N: int, K: int, L: int) -> Tuple[List[ReservoirParams], List[Reservoir]]:
    configs = [ReservoirParams(**vars(p)) for p in RESERVOIR_PARAM_DEFAULTS]
    bank = _instantiate_reservoirs(configs, N, K, L)
    return configs, bank


__all__ = [
    "RESERVOIR_SEEDS",
    "RESERVOIR_PARAM_DEFAULTS",
    "_instantiate_reservoirs",
    "reset_reservoir_bank",
]
