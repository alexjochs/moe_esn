import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from dataset import Regime
from reservoir import Reservoir, ReservoirParams


def teacher_forced_states(reservoir: Reservoir,
                          targets: np.ndarray,
                          inputs: Optional[np.ndarray] = None) -> np.ndarray:
    """Return reservoir states for a teacher-forced rollout over ``targets``.

    Args:
        reservoir: Frozen reservoir instance.
        targets: Array shaped (L, T) or (T, L) of supervised outputs.
        inputs: Optional array shaped (K, T) providing external inputs per step.

    Returns:
        states shaped (N, T) where column ``t`` contains ``x_t``.
    """
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if targets.shape[0] != reservoir.L and targets.shape[1] == reservoir.L:
        targets = targets.T
    if targets.shape[0] != reservoir.L:
        raise ValueError(f"targets shape {targets.shape} incompatible with reservoir output dim {reservoir.L}")

    if inputs is not None:
        if inputs.ndim == 1:
            inputs = inputs.reshape(reservoir.K, -1)
        if inputs.shape[0] != reservoir.K and inputs.shape[1] == reservoir.K:
            inputs = inputs.T
        if inputs.shape[0] != reservoir.K:
            raise ValueError(f"inputs shape {inputs.shape} incompatible with reservoir input dim {reservoir.K}")
        if inputs.shape[1] != targets.shape[1]:
            raise ValueError("inputs and targets must share the same T length")

    T = targets.shape[1]
    states = np.zeros((reservoir.N, T), dtype=np.float32)
    x = reservoir.x0.copy()
    states[:, 0] = x

    for t in range(T - 1):
        u_t = None
        if inputs is not None and reservoir.K > 0:
            u_t = inputs[:, t]
        y_prev = targets[:, t]
        x = reservoir._step(x, u_t, y_prev)
        states[:, t + 1] = x

    return states


def fit_linear_readout(design_matrix: np.ndarray,
                       targets: np.ndarray,
                       alpha: float,
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Fit ridge regression readout ``W_out`` solving ``design @ W_out.T`` ~= ``targets``.

    Args:
        design_matrix: Array shaped (num_samples, N) of reservoir states per time step.
        targets: Array shaped (num_samples, L) or (num_samples,) of supervised outputs.
        alpha: Ridge regularization strength.
        sample_weight: Optional 1-D array of weights per sample.

    Returns:
        ``W_out`` shaped (L, N).
    """
    if design_matrix.ndim != 2:
        raise ValueError("design_matrix must be 2-D (num_samples, num_features)")
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if targets.ndim != 2:
        raise ValueError("targets must be 1-D or 2-D")
    if design_matrix.shape[0] != targets.shape[0]:
        raise ValueError("design_matrix and targets must share the same number of samples")
    if sample_weight is not None and sample_weight.shape[0] != design_matrix.shape[0]:
        raise ValueError("sample_weight must align with the number of samples")

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design_matrix, targets, sample_weight=sample_weight)
    coef = ridge.coef_.astype(np.float32)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return coef


def cycle_starts(labels: np.ndarray) -> List[int]:
    """Return indices where a new Mackey-Glass cycle begins."""
    starts: List[int] = []
    for i, label in enumerate(labels):
        if label == Regime.MACKEY_GLASS and (i == 0 or labels[i - 1] == Regime.TRANSITION):
            starts.append(i)
    return starts


def regime_start_indices_in_range(labels: np.ndarray, start: int, end: int) -> List[Tuple[int, int]]:
    """Yield (index, regime) pairs for starts that fall within [start, end)."""
    out: List[Tuple[int, int]] = []
    for i in range(start, end):
        label = labels[i]
        if label == Regime.TRANSITION:
            continue
        if i == start:
            if i == 0 or labels[i - 1] == Regime.TRANSITION:
                out.append((i, int(label)))
        else:
            if labels[i - 1] == Regime.TRANSITION:
                out.append((i, int(label)))
    return out


def segment_end(labels: np.ndarray, start: int) -> int:
    """Return the index where the current non-transition segment ends."""
    i = start
    while i < len(labels) and labels[i] != Regime.TRANSITION:
        i += 1
    return i


def free_run_from_state(reservoir: Reservoir,
                        x0: np.ndarray,
                        W_out: np.ndarray,
                        horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run closed loop for ``horizon`` steps starting from ``x0``.

    Returns predicted sequence and final reservoir state.
    """
    x = x0.copy()
    y_seq = np.zeros((horizon,), dtype=np.float32)
    for t in range(horizon):
        z_hat = (W_out @ x).reshape(reservoir.L)
        y_hat = np.tanh(z_hat)
        y_seq[t] = float(y_hat[0])
        u_t = None
        if reservoir.K > 0:
            u_t = reservoir._zero_u
        x = reservoir._step(x, u_t, y_hat)
    return y_seq, x


def nrmse(y_hat: np.ndarray, y_true: np.ndarray, denom: float) -> float:
    err = y_hat - y_true
    rmse = math.sqrt(float(np.mean(err * err)))
    return rmse / (denom + 1e-12)


def train_readout(states: np.ndarray,
                  targets: np.ndarray,
                  labels: np.ndarray,
                  alpha: float,
                  discard: int = 0,
                  train_fraction: float = 0.7) -> Tuple[np.ndarray, int]:
    """Fit ridge regression readout and return (W_out, train_end_index)."""
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if states.ndim != 2:
        raise ValueError("states must be shaped (N, T)")
    if targets.shape[1] != states.shape[1]:
        raise ValueError("targets and states must share the same temporal length")

    T = targets.shape[1]
    idx = np.arange(T)
    cycles = cycle_starts(labels)
    n_cycles = len(cycles)
    if n_cycles == 0:
        raise RuntimeError("No cycles detected; check labels.")
    n_train_cycles = max(1, int(np.floor(train_fraction * n_cycles)))
    if n_train_cycles < n_cycles:
        train_end = cycles[n_train_cycles]
    else:
        train_end = int(train_fraction * T)

    train_mask = (idx >= discard) & (idx < train_end) & (labels != Regime.TRANSITION)
    train_idx = idx[train_mask]

    X_train = states[:, train_idx].T
    Y_train = targets[:, train_idx].T
    z_train = np.arctanh(np.clip(Y_train, -0.999, 0.999))

    W_out = fit_linear_readout(X_train, z_train, alpha=alpha)

    return W_out, train_end


def evaluate(reservoir: Reservoir,
             states: np.ndarray,
             targets: np.ndarray,
             labels: np.ndarray,
             train_end: int,
             horizons: Sequence[int],
             probe_offsets: Sequence[int],
             W_out: np.ndarray) -> Tuple[float, Dict]:
    """Run probe evaluations and return (fitness, diagnostics)."""
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    total_steps = targets.shape[1]
    target_series = targets[0]

    regime_std: Dict[int, float] = {}
    indices = np.arange(total_steps)
    train_region = indices < train_end
    for regime in (Regime.MACKEY_GLASS, Regime.LORENZ, Regime.ROSSLER):
        mask = train_region & (labels == regime)
        vals = target_series[mask]
        regime_std[int(regime)] = float(np.std(vals)) if np.any(mask) else 1.0

    evaluation_records: List[Dict] = []
    test_start_index = train_end
    test_end_index = total_steps

    segment_starts = regime_start_indices_in_range(labels, test_start_index, test_end_index)
    for segment_start, regime_id in segment_starts:
        segment_stop = segment_end(labels, segment_start)
        for probe_offset in probe_offsets:
            probe_index = segment_start + probe_offset
            if probe_index >= segment_stop - 1:
                continue
            for horizon in horizons:
                if probe_index + horizon >= segment_stop:
                    continue
                state_at_probe = states[:, probe_index]
                y_pred, _ = free_run_from_state(reservoir, state_at_probe, W_out, horizon)
                y_true = target_series[probe_index + 1:probe_index + 1 + horizon]
                normalization = regime_std[regime_id]
                error_value = nrmse(y_pred, y_true, normalization)
                if not np.isfinite(error_value):
                    error_value = 1e6
                evaluation_records.append({
                    'probe_index': int(probe_index),
                    'probe_offset': int(probe_offset),
                    'horizon': int(horizon),
                    'regime': int(regime_id),
                    'nrmse': float(error_value),
                })

    if not evaluation_records:
        return 1e6, {'probes': 0}

    median_nrmse_by_regime_horizon: Dict[Tuple[int, int], float] = {}
    for record in evaluation_records:
        key = (record['regime'], record['horizon'])
        median_nrmse_by_regime_horizon.setdefault(key, []).append(record['nrmse'])
    for key, values in median_nrmse_by_regime_horizon.items():
        median_nrmse_by_regime_horizon[key] = float(np.median(np.array(values)))

    fitness = float(np.mean(list(median_nrmse_by_regime_horizon.values()))) if median_nrmse_by_regime_horizon else 1e6

    return fitness, {
        'median_nrmse_by_regime_horizon': median_nrmse_by_regime_horizon,
        'probes': len(evaluation_records),
        'records': evaluation_records,
    }
def build_reservoir(rng: np.random.Generator,
                    N: int,
                    K: int,
                    L: int,
                    *,
                    C: float,
                    decay_rate: float,
                    spectral_radius: float,
                    w_scale: float,
                    wback_scale: float,
                    p_nonzero: float,
                    p_in: float,
                    w_in_scale: float,
                    bias_value: float) -> Reservoir:
    """Construct a ``Reservoir`` using SimpleNamespace-style hyperparameters."""
    params = ReservoirParams(
        spectral_radius=spectral_radius,
        C=C,
        decay_rate=decay_rate,
        w_scale=w_scale,
        w_sparsity=float(np.clip(1.0 - p_nonzero, 0.0, 1.0)),
        w_back_scale=wback_scale,
        w_in_scale=w_in_scale,
        w_in_sparsity=float(np.clip(1.0 - p_in, 0.0, 1.0)),
        bias_value=bias_value,
    )
    return Reservoir(N, K, L, params, rng)
