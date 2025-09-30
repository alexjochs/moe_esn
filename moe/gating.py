from typing import Dict, List, Tuple

import numpy as np

from dataset import Regime


def compute_nrmse(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """Compute normalized root mean squared error (NRMSE)."""
    mse = np.mean((preds - targets) ** 2)
    var = np.var(targets)
    return np.sqrt(mse / (var + eps))


def compute_responsibilities_with_regularization(
    errors: np.ndarray,
    tau: float,
    eps_uniform: float,
    lambda_load: float,
    previous_responsibilities: np.ndarray | None = None,
    alpha: float = 0.2,
) -> np.ndarray:
    """Convert per-sequence errors to responsibilities with regularization controls."""
    num_sequences, num_experts = errors.shape
    logits = -errors / max(tau, 1e-8)

    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    q = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if lambda_load > 0.0:
        p = q.mean(axis=0)
        target = 1.0 / float(num_experts)
        bias = -lambda_load * (p - target)[None, :]
        logits = logits + bias
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        q = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if previous_responsibilities is not None:
        q = alpha * q + (1.0 - alpha) * previous_responsibilities
        q /= q.sum(axis=1, keepdims=True)

    if eps_uniform > 0.0:
        q = (1.0 - eps_uniform) * q + (eps_uniform / float(num_experts))
        q /= q.sum(axis=1, keepdims=True)

    return q


def mean_responsibility_by_regime(windows: List[Dict], responsibilities: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Average responsibilities per regime; returns (mean_by_regime, counts)."""
    if responsibilities.shape[0] != len(windows):
        raise ValueError("responsibilities rows must match windows length")
    totals: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for window_index, window in enumerate(windows):
        regime_id = int(window['regime'])
        if regime_id not in totals:
            totals[regime_id] = np.zeros(responsibilities.shape[1], dtype=np.float64)
            counts[regime_id] = 0
        totals[regime_id] += responsibilities[window_index]
        counts[regime_id] += 1
    means = {regime_id: totals[regime_id] / counts[regime_id] for regime_id in totals}
    return means, counts


def serialize_regime_means(regime_means: Dict[int, np.ndarray]) -> Dict[str, List[float]]:
    summary: Dict[str, List[float]] = {}
    for regime_id, values in regime_means.items():
        try:
            regime_name = Regime(regime_id).name
        except ValueError:
            regime_name = str(regime_id)
        summary[regime_name] = values.tolist()
    return summary


def serialize_regime_counts(regime_counts: Dict[int, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for regime_id, count in regime_counts.items():
        try:
            regime_name = Regime(regime_id).name
        except ValueError:
            regime_name = str(regime_id)
        out[regime_name] = int(count)
    return out


__all__ = [
    "compute_nrmse",
    "compute_responsibilities_with_regularization",
    "mean_responsibility_by_regime",
    "serialize_regime_means",
    "serialize_regime_counts",
]
