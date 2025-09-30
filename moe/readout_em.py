from typing import Dict, List, Optional, Tuple

import numpy as np

from reservoir import Reservoir
from single_reservoir_core import teacher_forced_states, fit_linear_readout

from moe.gating import compute_nrmse, compute_responsibilities_with_regularization


def build_state_target_pairs(reservoirs: List[Reservoir], windows: List[Dict]) -> List[List[Dict[str, np.ndarray]]]:
    """Run each reservoir and capture teacher-forced states/targets for every window."""
    state_target_pairs: List[List[Dict[str, np.ndarray]]] = []
    for reservoir in reservoirs:
        reservoir_state_target_pairs: List[Dict[str, np.ndarray]] = []
        for window in windows:
            targets = window['y'].astype(np.float32)
            states = teacher_forced_states(reservoir, targets.T)
            idx_warmup_end = window['idx_warmup_end']
            idx_fit_end = window['idx_fit_end']
            X_n_fit = states[:, idx_warmup_end + 1:idx_fit_end + 1].T.astype(np.float32)
            Y_teach_fit = targets[idx_warmup_end:idx_fit_end, :].astype(np.float32)
            reservoir_state_target_pairs.append({'X_n_fit': X_n_fit, 'Y_teach_fit': Y_teach_fit})
        state_target_pairs.append(reservoir_state_target_pairs)
    return state_target_pairs


def init_readouts_unweighted(state_target_pairs: List[List[Dict[str, np.ndarray]]], lam: float) -> List[np.ndarray]:
    """Solve unweighted ridge per expert using concatenated state/target pairs."""
    W_out_list: List[np.ndarray] = []
    for _, reservoir_state_target_pairs in enumerate(state_target_pairs):
        X_n_blocks: List[np.ndarray] = []
        Y_teach_blocks: List[np.ndarray] = []
        for state_target_pair in reservoir_state_target_pairs:
            X_n_blocks.append(state_target_pair['X_n_fit'])
            Y_teach_blocks.append(state_target_pair['Y_teach_fit'])
        X_n_concat = np.vstack(X_n_blocks)
        Y_teach_concat = np.vstack(Y_teach_blocks)
        W_out = fit_linear_readout(X_n_concat, Y_teach_concat, alpha=lam)
        W_out_list.append(W_out)
    return W_out_list


def refit_readouts_weighted(designs: List[List[Dict[str, np.ndarray]]],
                            responsibilities: np.ndarray,
                            lam: float,
                            prev_W: List[np.ndarray]) -> List[np.ndarray]:
    """Refit each expert's readout using per-window responsibilities as sample weights."""
    num_experts = len(designs)
    W_out_new: List[np.ndarray] = []
    for expert_idx in range(num_experts):
        H_blocks: List[np.ndarray] = []
        Y_blocks: List[np.ndarray] = []
        sample_weights: List[np.ndarray] = []
        for window_idx, row in enumerate(designs[expert_idx]):
            weight = responsibilities[window_idx, expert_idx]
            X_n_fit = row['X_n_fit']
            Y_fit = row['Y_teach_fit']
            H_blocks.append(X_n_fit)
            Y_blocks.append(Y_fit)
            sample_weights.append(np.full(X_n_fit.shape[0], weight, dtype=np.float32))
        H_concat = np.vstack(H_blocks)
        Y_concat = np.vstack(Y_blocks)
        sample_weight_array = np.concatenate(sample_weights)
        W_out = fit_linear_readout(H_concat, Y_concat, alpha=lam, sample_weight=sample_weight_array)
        W_out_new.append(W_out.astype(np.float32))
    return W_out_new


def compute_errors_matrix(reservoirs: List[Reservoir],
                          windows: List[Dict],
                          W_out_list: List[np.ndarray],
                          horizon: int = 10) -> np.ndarray:
    """Return errors matrix e[sequence_index, reservoir_index] = NRMSE@horizon."""
    errors = np.zeros((len(windows), len(reservoirs)), dtype=np.float32)
    for sequence_index, window in enumerate(windows):
        for reservoir_index, reservoir in enumerate(reservoirs):
            y_hat, y_true = reservoir.free_run(window, W_out_list[reservoir_index], horizon=horizon)
            errors[sequence_index, reservoir_index] = compute_nrmse(y_hat, y_true)
    return errors


def em_round_readouts_only(reservoirs: List[Reservoir],
                          windows: List[Dict],
                          designs: List[List[Dict[str, np.ndarray]]],
                          W_out_list: List[np.ndarray],
                          lam: float,
                          tau: float,
                          eps_uniform: float,
                          lambda_load: float,
                          ema_prev: Optional[np.ndarray] = None,
                          alpha: float = 0.2,
                          horizon: int = 10) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Perform one EM round using the readout-only update."""
    sequence_error_matrix = compute_errors_matrix(reservoirs, windows, W_out_list, horizon=horizon)
    responsibilities = compute_responsibilities_with_regularization(
        sequence_error_matrix,
        tau=tau,
        eps_uniform=eps_uniform,
        lambda_load=lambda_load,
        previous_responsibilities=ema_prev,
        alpha=alpha,
    )
    W_new = refit_readouts_weighted(designs, responsibilities, lam, W_out_list)
    return W_new, responsibilities, sequence_error_matrix


def prepare_em_readouts_only(reservoirs: List[Reservoir],
                             windows: List[Dict],
                             lam: float) -> Tuple[List[np.ndarray], List[List[Dict[str, np.ndarray]]]]:
    """Precompute teacher-forced design matrices and initialize readouts."""
    state_target_pairs = build_state_target_pairs(reservoirs, windows)
    W_out_list = init_readouts_unweighted(state_target_pairs, lam)
    return W_out_list, state_target_pairs


def run_one_em_round(reservoirs: List[Reservoir],
                     windows: List[Dict],
                     W_out_list: List[np.ndarray],
                     designs: List[List[Dict[str, np.ndarray]]],
                     tau: float = 0.8,
                     lam: float = 1e-3,
                     eps_uniform: float = 0.0,
                     lambda_load: float = 0.0,
                     horizon: int = 10,
                     ema_prev: Optional[np.ndarray] = None,
                     alpha: float = 0.2):
    """Wrapper that runs a single EM iteration and returns weights, responsibilities, and errors."""
    return em_round_readouts_only(
        reservoirs,
        windows,
        designs,
        W_out_list,
        lam=lam,
        tau=tau,
        eps_uniform=eps_uniform,
        lambda_load=lambda_load,
        ema_prev=ema_prev,
        alpha=alpha,
        horizon=horizon,
    )


__all__ = [
    "build_state_target_pairs",
    "init_readouts_unweighted",
    "refit_readouts_weighted",
    "compute_errors_matrix",
    "em_round_readouts_only",
    "prepare_em_readouts_only",
    "run_one_em_round",
]
