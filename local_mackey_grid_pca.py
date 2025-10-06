"""Parallel hyperparameter landscape visualizer for the Mackey-Glass ESN.

This script samples hyperparameter configurations uniformly at random within
the configured ranges, records the resulting 100-step NRMSE scores, and renders
a weighted PCA projection that emphasizes low-error regions to highlight how
sparse the performant settings are. Axes automatically stretch when a component
collapses onto a small set of discrete values, and the plot annotates each PCA
direction with its dominant parameter loadings. Use ``--smoke-test`` to cap the
sweep at 2,000 combinations for quick iteration.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors

from dataset import Regime, generate_mackey_glass
from moe.gating import compute_nrmse
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

# Hyperparameter sampling configuration
DEFAULT_NUM_SAMPLES = 5000

# NRMSE thresholds for coloring
THRESH_RED = 0.5
THRESH_YELLOW = 1.0
THRESH_GREEN = 10.0

# Default output paths
DEFAULT_PLOT_PATH = Path("runs/local_mackey_grid_pca/pca_landscape.png")
DEFAULT_CSV_PATH = Path("runs/local_mackey_grid_pca/results.csv")

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

# (low, high, clip_to_unit_interval)
PARAM_RANGES: Dict[str, Tuple[float, float, bool]] = {
    "spectral_radius": (0.3, 1.2, False),
    "C": (0.1, 1.0, True),
    "decay_rate": (0.01, 0.99, True),
    "w_scale": (0.01, 1.0, False),
    "w_sparsity": (0.5, 0.995, True),
    "w_back_scale": (0.1, 1.0, False),
    "w_in_scale": (0.02, 1.0, False),
    "w_in_sparsity": (0.1, 0.99, True),
    "bias_value": (-0.9, 0.9, False),
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

def _sample_param_matrix(total_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``total_samples`` random configurations within the parameter ranges."""
    if total_samples <= 0:
        raise ValueError("Number of parameter samples must be positive.")

    matrix = np.empty((total_samples, len(PARAM_KEYS)), dtype=np.float64)
    for column, key in enumerate(PARAM_KEYS):
        lo, hi, clip = PARAM_RANGES[key]
        samples = rng.uniform(lo, hi, size=total_samples)
        if clip:
            samples = np.clip(samples, 0.0, 1.0)
        matrix[:, column] = samples
    return matrix


_WINDOW: Dict | None = None
_FIT_START: int | None = None
_FIT_END: int | None = None


def _worker_init(window: Dict) -> None:
    """Initializer to share common data with worker processes."""
    global _WINDOW, _FIT_START, _FIT_END
    _WINDOW = window
    _FIT_START = int(window["idx_warmup_end"])
    _FIT_END = int(window["idx_fit_end"])


def iter_param_combinations(params_matrix: np.ndarray) -> Iterator[Tuple[int, Tuple[float, ...]]]:
    """Yield ``(index, values)`` pairs for each sampled hyperparameter tuple."""
    for index in range(params_matrix.shape[0]):
        yield index, tuple(float(value) for value in params_matrix[index])


def _evaluate_combination(task: Tuple[int, Tuple[float, ...]]) -> Tuple[int, Tuple[float, ...], float]:
    """Worker helper to evaluate a single hyperparameter combination."""
    if _WINDOW is None or _FIT_START is None or _FIT_END is None:
        raise RuntimeError("Worker process not initialized with window data.")

    index, combo_values = task
    params_kwargs = {key: value for key, value in zip(PARAM_KEYS, combo_values)}
    params = ReservoirParams(**params_kwargs)

    reservoir_seed = BASE_RESERVOIR_SEED + index
    reservoir_rng = np.random.default_rng(reservoir_seed)
    reservoir = Reservoir(RESERVOIR_SIZE, INPUT_DIM, OUTPUT_DIM, params, reservoir_rng)

    targets = _WINDOW["y"].astype(np.float32)
    states = teacher_forced_states(reservoir, targets.T)
    design_matrix = states[:, _FIT_START + 1:_FIT_END + 1].T.astype(np.float32)
    fit_targets = targets[_FIT_START:_FIT_END, :]
    W_out = fit_linear_readout(design_matrix, fit_targets, alpha=RIDGE_ALPHA)

    y_hat, y_true = reservoir.free_run(_WINDOW, W_out, horizon=FREE_RUN_HORIZON)
    y_hat = y_hat.reshape(-1)
    y_true = y_true.reshape(-1)
    nrmse_100 = compute_nrmse(y_hat[:NRMSE_HORIZON], y_true[:NRMSE_HORIZON])

    return index, combo_values, float(nrmse_100)


def evaluate_grid(num_samples: int, max_combinations: int | None, max_workers: int, window: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate random hyperparameter samples in parallel and return (params, scores)."""
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    total_combinations = num_samples
    if max_combinations is not None:
        total_combinations = min(num_samples, max_combinations)

    if total_combinations == 0:
        raise ValueError("No hyperparameter combinations generated.")

    print(
        f"Evaluating {total_combinations:,} random hyperparameter configurations "
        f"with {max_workers} worker(s)."
    )

    param_rng = np.random.default_rng(DATA_SEED + 1)
    params_matrix = _sample_param_matrix(total_combinations, param_rng)
    scores = np.empty(total_combinations, dtype=np.float64)

    combination_iter = iter_param_combinations(params_matrix)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(window,)) as executor:
        for index, _, score in executor.map(_evaluate_combination, combination_iter, chunksize=32):
            scores[index] = score

    return params_matrix, scores


def compute_pca_weights(scores: np.ndarray) -> np.ndarray:
    """Derive PCA weights so low NRMSE configurations drive the projection."""
    clipped = np.clip(scores, 1e-6, None)
    weights = np.power(clipped, -0.25)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Failed to compute valid PCA weights.")
    return weights / total


def project_pca(
    matrix: np.ndarray,
    components: int = 2,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project matrix onto the leading principal components using SVD."""
    if components <= 0:
        raise ValueError("Number of PCA components must be positive.")

    if weights is None:
        mean_vec = matrix.mean(axis=0)
        centered = matrix - mean_vec
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:components].T
        return centered @ basis, basis

    if weights.ndim != 1 or weights.size != matrix.shape[0]:
        raise ValueError("Weights must be a 1-D array matching the number of samples.")

    mean_vec = np.average(matrix, axis=0, weights=weights)
    centered = matrix - mean_vec
    weighted_centered = centered * np.sqrt(weights[:, None])
    _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
    basis = vh[:components].T
    return centered @ basis, basis


def map_colors(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign RGBA colors, marker sizes, and highlight mask based on scores."""
    colors = np.empty((scores.size, 4), dtype=np.float32)
    sizes = np.full(scores.size, 20.0, dtype=np.float32)

    red_mask = scores < THRESH_RED
    yellow_mask = (scores >= THRESH_RED) & (scores < THRESH_YELLOW)
    green_mask = (scores >= THRESH_YELLOW) & (scores < THRESH_GREEN)
    blue_mask = scores >= THRESH_GREEN

    colors[red_mask] = mcolors.to_rgba("#d73027")
    colors[yellow_mask] = mcolors.to_rgba("#fee08b")
    colors[green_mask] = mcolors.to_rgba("#1a9850")

    sizes[red_mask] = 90.0
    sizes[yellow_mask] = 42.0
    sizes[green_mask] = 28.0

    if np.any(blue_mask):
        blue_scores = scores[blue_mask]
        log_scores = np.log10(np.clip(blue_scores, 1.0, None))
        log_min = log_scores.min()
        log_max = log_scores.max()
        if math.isclose(log_min, log_max):
            normed = np.zeros_like(log_scores)
        else:
            normed = (log_scores - log_min) / (log_max - log_min)
        colors[blue_mask] = cm.Blues(normed)

    return colors, sizes, red_mask


def ensure_output_dir(path: Path) -> None:
    """Create parent directories for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_results_csv(params: np.ndarray, scores: np.ndarray, csv_path: Path) -> None:
    """Persist the results to a CSV with header columns."""
    ensure_output_dir(csv_path)
    header = ",".join([*PARAM_KEYS, "nrmse_100"])
    data = np.column_stack((params, scores.reshape(-1, 1)))
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")


def _format_component_loadings(basis: np.ndarray, top_k: int = 4) -> str:
    lines: List[str] = []
    for component_index in range(basis.shape[1]):
        coeffs = basis[:, component_index]
        ordered = np.argsort(-np.abs(coeffs))[:top_k]
        formatted = [f"{coeffs[i]:+.2f}·{PARAM_KEYS[i]}" for i in ordered]
        lines.append(f"PCA{component_index + 1}: " + ", ".join(formatted))
    return "\n".join(lines)


def render_pca_plot(
    pca_points: np.ndarray,
    scores: np.ndarray,
    plot_path: Path,
    basis: np.ndarray,
) -> None:
    """Render and save the PCA scatter plot with color-coded scores."""
    ensure_output_dir(plot_path)
    colors, sizes, red_mask = map_colors(scores)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        pca_points[:, 0],
        pca_points[:, 1],
        c=colors,
        s=sizes,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.85,
    )

    if np.any(red_mask):
        ax.scatter(
            pca_points[red_mask, 0],
            pca_points[red_mask, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            s=sizes[red_mask] * 1.4,
        )

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("Mackey-Glass Reservoir Hyperparameter Landscape (NRMSE_100)")
    ax.grid(alpha=0.2)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="NRMSE < 0.5", markerfacecolor="#d73027", markeredgecolor="black", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="NRMSE < 1", markerfacecolor="#fee08b", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="NRMSE < 10", markerfacecolor="#1a9850", markeredgecolor="black", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="NRMSE ≥ 10", markerfacecolor="#2166ac", markeredgecolor="black", markersize=7),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    loadings_text = _format_component_loadings(basis)
    ax.text(
        0.02,
        0.98,
        loadings_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Mackey-Glass hyperparameter landscape via PCA.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of random hyperparameter configurations to evaluate.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional cap on the number of hyperparameter configurations to evaluate.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit the sweep to 2,000 combinations for a quick smoke test.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Output path for the PCA scatter plot.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Optional CSV output path to store raw combinations and scores.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes to launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_samples = max(1, args.num_samples)
    max_workers = max(1, args.max_workers)
    max_combinations = args.max_combinations
    if args.smoke_test:
        max_combinations = 2000 if max_combinations is None else min(max_combinations, 2000)

    rng = np.random.default_rng(DATA_SEED)
    _, series_std = generate_mackey_glass(H=TOTAL_WINDOW_LEN, rng=rng)
    window = build_window(series_std)

    params_matrix, scores = evaluate_grid(num_samples, max_combinations, max_workers, window)

    ensure_output_dir(args.csv_path)
    save_results_csv(params_matrix, scores, args.csv_path)

    pca_weights = compute_pca_weights(scores)
    pca_points, pca_basis = project_pca(params_matrix, components=2, weights=pca_weights)

    component_summary = _format_component_loadings(pca_basis)
    print("PCA component loadings:\n" + component_summary)

    render_pca_plot(pca_points, scores, args.plot_path, pca_basis)


if __name__ == "__main__":
    main()
