"""Interactive Mackey-Glass hyperparameter sweep for a single reservoir.

This script iterates through hyperparameters picked from hard-coded candidate
lists derived from the mixture-of-reservoirs ranges to run a local sanity check
without routing or Dask. It generates one teacher-forced Mackey-Glass window,
loops over each ``ReservoirParams`` value set, fits a ridge readout, and then
launches an interactive free-run plot for configurations whose 100-step NRMSE
is at most 10. Close each figure window to move to the next configuration.
Plots are not saved to disk.
"""

import itertools
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from dataset import Regime, generate_mackey_glass
from moe.gating import compute_nrmse
from moe.reservoir_bank import RESERVOIR_PARAM_DEFAULTS
from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import fit_linear_readout, teacher_forced_states

# Reservoir dimensions
INPUT_DIM = 1
RESERVOIR_SIZE = 100
OUTPUT_DIM = 1

# Window specification (mirrors mixture_of_reservoirs defaults)
WARMUP_LEN = 200
TEACHER_FORCED_LEN = 1000
FREE_RUN_HORIZON = 500
NRMSE_HORIZON = 100

TOTAL_WINDOW_LEN = WARMUP_LEN + TEACHER_FORCED_LEN + FREE_RUN_HORIZON

# Training configuration
RIDGE_ALPHA = 1e-4
DATA_SEED = 31415
BASE_RESERVOIR_SEED = 9000

NRMSE_THRERSHOLD = 0.1 # value a reservoir needs to be under to be visualized

BASE_PARAMS = ReservoirParams(**vars(RESERVOIR_PARAM_DEFAULTS[0]))
PARAM_KEYS: List[str] = [
    "spectral_radius",
    "C",
    "decay_rate",
    "w_scale",
    "w_sparsity",
    "w_back_scale",
    "w_in_scale",
    "w_in_sparsity",
    "bias_value",
]


# Hyperparameter candidate values derived from mixture-of-reservoirs ranges
HPARAM_VALUE_OPTIONS: Dict[str, List[float]] = {
    "spectral_radius": [0.30, 0.75, 0.95],
    "C": [0.10, 0.55, 1.00],
    "decay_rate": [0.01, 0.50, 0.99],
    "w_scale": [0.01, 0.50, 1.00],
    "w_sparsity": [0.50, 0.75, 0.995],
    "w_back_scale": [0.10, 0.55, 1.00],
    "w_in_scale": [0.02, 0.50, 1.00],
    "w_in_sparsity": [0.10, 0.50, 0.99],
    "bias_value": [-0.90, 0.00, 0.90],
}


def build_window(series: np.ndarray) -> Dict:
    """Convert a 1-D series into the window dict expected by Reservoir helpers."""
    y = series.reshape(-1, 1).astype(np.float32)
    return {
        "y": y,
        "idx_warmup_end": WARMUP_LEN,
        "idx_fit_end": WARMUP_LEN + TEACHER_FORCED_LEN,
        "idx_eval_end": TOTAL_WINDOW_LEN,
        "regime": int(Regime.MACKEY_GLASS),
        "id": "mackey_glass_window",
    }


def iter_param_configs() -> Iterable[ReservoirParams]:
    """Yield every ``ReservoirParams`` combination from the per-parameter lists."""
    value_axes = [HPARAM_VALUE_OPTIONS[key] for key in PARAM_KEYS]

    for combo in itertools.product(*value_axes):
        kwargs = {key: float(value) for key, value in zip(PARAM_KEYS, combo)}
        yield ReservoirParams(**kwargs)


def format_config_summary(params: ReservoirParams) -> str:
    """Return a short string summarizing key hyperparameters."""
    return (
        f"rho={params.spectral_radius:.2f}, "
        f"C={params.C:.2f}, "
        f"leak={params.decay_rate:.2f}, "
        f"w={params.w_scale:.2f}, "
        f"w_in={params.w_in_scale:.2f}, "
        f"w_back={params.w_back_scale:.2f}, "
        f"sparsity={params.w_sparsity:.3f}"
    )


def main() -> None:
    rng = np.random.default_rng(DATA_SEED)
    _, series_std = generate_mackey_glass(H=TOTAL_WINDOW_LEN, rng=rng)
    window = build_window(series_std)

    configs = list(iter_param_configs())
    total_configs = len(configs)
    fit_start = WARMUP_LEN
    fit_end = WARMUP_LEN + TEACHER_FORCED_LEN

    for index, params in enumerate(configs):
        reservoir_seed = BASE_RESERVOIR_SEED + index
        reservoir_rng = np.random.default_rng(reservoir_seed)
        reservoir = Reservoir(RESERVOIR_SIZE, INPUT_DIM, OUTPUT_DIM, params, reservoir_rng)

        targets = window["y"].astype(np.float32)
        states = teacher_forced_states(reservoir, targets.T)
        design_matrix = states[:, fit_start + 1:fit_end + 1].T.astype(np.float32)
        fit_targets = targets[fit_start:fit_end, :]
        W_out = fit_linear_readout(design_matrix, fit_targets, alpha=RIDGE_ALPHA)

        y_hat, y_true = reservoir.free_run(window, W_out, horizon=FREE_RUN_HORIZON)
        y_hat = y_hat.reshape(-1)
        y_true = y_true.reshape(-1)
        nrmse_100 = compute_nrmse(y_hat[:NRMSE_HORIZON], y_true[:NRMSE_HORIZON])

        if nrmse_100 > NRMSE_THRERSHOLD:
            continue

        combo_label = format_config_summary(params)

        time_axis = np.arange(FREE_RUN_HORIZON)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_axis, y_true, label="target", linewidth=1.2)
        ax.plot(time_axis, y_hat, label="free-run", linewidth=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.set_title(f"{combo_label}\nNRMSE_100 = {nrmse_100:.4f}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
