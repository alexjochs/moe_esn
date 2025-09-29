# imports
import numpy as np
from typing import Dict, Tuple, List, Optional
from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import teacher_forced_states, fit_linear_readout

from dataset import generate_lorenz, generate_mackey_glass, generate_rossler, Regime


rng = np.random.default_rng(42)

# -----------------------------------------------------------------------------
# Dataset windowing
#   Window spec: [warmup | teacher-forced fit | eval (free-run)]
# -----------------------------------------------------------------------------
W = 200                # warmup length
T_FIT = 200            # teacher-forced fit span; set to 0 to disable
H_MAX = 50             # maximum eval horizon placeholder (not used yet)
WINDOW_T = W + T_FIT + H_MAX
N_WINDOWS_PER_REGIME = 200


def _gen_regime_window(regime_id: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return y:(T,1) for a single pure-regime rollout using the standardized output
    from each generator.
    """
    if regime_id == int(Regime.MACKEY_GLASS):
        _, y_std = generate_mackey_glass(H=T, rng=rng)
    elif regime_id == int(Regime.LORENZ):
        _, y_std = generate_lorenz(H=T, rng=rng)
    elif regime_id == int(Regime.ROSSLER):
        _, y_std = generate_rossler(H=T, rng=rng)
    else:
        raise ValueError("Unknown regime id")
    return y_std.reshape(T, 1).astype(np.float32)


def _build_fixed_windows(n_per_regime: int, T: int, rng: np.random.Generator) -> Dict[str, list]:
    """
    Build pure-regime windows with index markers for warmup/fit/eval.
    Returns dict with keys 'train' and 'test'; each is a list of dicts:
      { 'y': (T,1), 'idx_warmup_end': W, 'idx_fit_end': W+T_FIT, 'idx_eval_end': T, 'regime': r, 'id': int }
    Split is per-regime 70/30 without shuffling fuss.
    """
    all_windows = []
    uid = 0
    for r in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        for _ in range(n_per_regime):
            y = _gen_regime_window(r, T, rng)
            all_windows.append({
                'y': y,
                'idx_warmup_end': W,
                'idx_fit_end': W + T_FIT,
                'idx_eval_end': T,
                'regime': r,
                'id': uid,
            })
            uid += 1

    # per-regime 70/30 split
    train, test = [], []
    for r in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        Rs = [w for w in all_windows if w['regime'] == r]
        n_tr = int(0.7 * len(Rs))
        train += Rs[:n_tr]
        test  += Rs[n_tr:]
    return {'train': train, 'test': test}


# -----------------------------------------------------------------------------
# Utility functions for Step 3 (EM round with readouts only)
# -----------------------------------------------------------------------------

def sequence_error(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute normalized root mean squared error (NRMSE) over the horizon=10.
    preds and targets are arrays of shape (horizon, L).
    """
    mse = np.mean((preds - targets) ** 2)
    var = np.var(targets)
    return np.sqrt(mse / (var + eps))


def soft_responsibilities(errors: np.ndarray, tau: float, prior: np.ndarray | None = None,
                          ema_prev: np.ndarray | None = None, alpha: float = 0.2) -> np.ndarray:
    """
    Convert per-sequence errors to soft responsibilities via temperature softmax.
    Optionally multiply by prior and smooth with exponential moving average.

    Args:
        errors: shape (num_sequences, num_experts), error values per sequence and expert
        tau: temperature parameter > 0
        prior: optional prior weights shape (num_experts,)
        ema_prev: optional previous EMA responsibilities shape (num_sequences, num_experts)
        alpha: EMA smoothing factor in [0,1]

    Returns:
        responsibilities: shape (num_sequences, num_experts), rows sum to 1
    """
    # Negative error scaled by temperature
    logits = -errors / tau
    if prior is not None:
        logits += np.log(prior + 1e-12)  # avoid log(0)
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    resp = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if ema_prev is not None:
        resp = alpha * resp + (1 - alpha) * ema_prev
        # Re-normalize after EMA smoothing
        resp /= resp.sum(axis=1, keepdims=True)

    return resp


def build_window_designs(reservoirs: List[Reservoir], windows: List[Dict]) -> List[List[Dict[str, np.ndarray]]]:
    """Cache teacher-forced states for the fit spans of each window and expert."""
    designs: List[List[Dict[str, np.ndarray]]] = []
    for res in reservoirs:
        res_designs: List[Dict[str, np.ndarray]] = []
        for w in windows:
            targets = w['y'].astype(np.float32)
            if targets.ndim == 1:
                targets = targets.reshape(-1, res.L)
            states = teacher_forced_states(res, targets.T)
            wu = w['idx_warmup_end']
            fit_end = w['idx_fit_end']
            # Align states with labels: column t+1 corresponds to label at index t.
            H_fit = states[:, wu + 1:fit_end + 1].T.astype(np.float32)
            Y_fit = targets[wu:fit_end, :].astype(np.float32)
            res_designs.append({'H_fit': H_fit, 'Y_fit': Y_fit})
        designs.append(res_designs)
    return designs


def init_readouts_unweighted(designs: List[List[Dict[str, np.ndarray]]], lam: float) -> List[np.ndarray]:
    """Solve unweighted ridge per expert using concatenated H/Y pairs."""
    W_out_list: List[np.ndarray] = []
    for expert_idx, design_rows in enumerate(designs):
        H_blocks: List[np.ndarray] = []
        Y_blocks: List[np.ndarray] = []
        for row in design_rows:
            H_fit = row['H_fit']
            if H_fit.size == 0:
                continue
            H_blocks.append(H_fit)
            Y_blocks.append(row['Y_fit'])
        if not H_blocks:
            res = reservoirs[expert_idx]
            W_out_list.append(np.zeros((res.L, res.N), dtype=np.float32))
            continue
        H_concat = np.vstack(H_blocks)
        Y_concat = np.vstack(Y_blocks)
        W_out = fit_linear_readout(H_concat, Y_concat, alpha=lam)
        W_out_list.append(W_out.astype(np.float32))
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
            if weight <= 0.0:
                continue
            H_fit = row['H_fit']
            if H_fit.size == 0:
                continue
            Y_fit = row['Y_fit']
            H_blocks.append(H_fit)
            Y_blocks.append(Y_fit)
            sample_weights.append(np.full(H_fit.shape[0], weight, dtype=np.float32))
        if not H_blocks:
            W_out_new.append(prev_W[expert_idx])
            continue
        H_concat = np.vstack(H_blocks)
        Y_concat = np.vstack(Y_blocks)
        sw = np.concatenate(sample_weights) if sample_weights else None
        W_out = fit_linear_readout(H_concat, Y_concat, alpha=lam, sample_weight=sw)
        W_out_new.append(W_out.astype(np.float32))
    return W_out_new


def compute_errors_matrix(reservoirs: List[Reservoir], windows: List[Dict], W_out_list: List[np.ndarray], horizon: int = 10) -> np.ndarray:
    """Return errors matrix e[s,i] = NRMSE@horizon for each window and expert."""
    S = len(windows)
    M = len(reservoirs)
    e = np.zeros((S, M), dtype=np.float32)
    for si, w in enumerate(windows):
        for i, res in enumerate(reservoirs):
            y_hat, y_true = res.free_run(w, W_out_list[i], horizon=horizon)
            e[si, i] = sequence_error(y_hat, y_true)
    return e


def em_round_readouts_only(reservoirs: List[Reservoir],
                          windows: List[Dict],
                          designs: List[List[Dict[str, np.ndarray]]],
                          W_out_list: List[np.ndarray],
                          lam: float,
                          tau: float,
                          prior: Optional[np.ndarray] = None,
                          ema_prev: Optional[np.ndarray] = None,
                          alpha: float = 0.2,
                          horizon: int = 10) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Perform one EM round:
      E-step: compute errors and responsibilities at the given horizon.
      M-step: refit each expert's readout with weighted ridge using per-sequence responsibilities.
    Returns: (W_out_list_new, responsibilities r, errors e)
    """
    # E-step
    e = compute_errors_matrix(reservoirs, windows, W_out_list, horizon=horizon)
    r = soft_responsibilities(e, tau=tau, prior=prior, ema_prev=ema_prev, alpha=alpha)

    # M-step per expert
    W_new = refit_readouts_weighted(designs, r, lam, W_out_list)

    return W_new, r, e


# -----------------------------------------------------------------------------
# Model dimensions (shared across experts)
# -----------------------------------------------------------------------------
K = 1   # number of input units
N = 100 # number of reservoir units
L = 1   # number of output units

# -----------------------------------------------------------------------------
# Expert parameter sets (three experts). Keep as provided but fixed.
# -----------------------------------------------------------------------------
# Reservoir 0 parameters
SPECTRAL_RADIUS_0 = 0.9
C_0 = 0.60
DECAY_RATE_0 = 0.85
W_IN_SCALE_0 = 0.08
W_SCALE_0 = 0.30
W_BACK_SCALE_0 = 0.56
W_IN_SPARSITY_0 = 0.60
W_SPARSITY_0 = 0.990

# Reservoir 1 parameters
SPECTRAL_RADIUS_1 = 0.9
C_1 = 0.60
DECAY_RATE_1 = 0.85
W_IN_SCALE_1 = 0.08
W_SCALE_1 = 0.30
W_BACK_SCALE_1 = 0.56
W_IN_SPARSITY_1 = 0.60
W_SPARSITY_1 = 0.990

# Reservoir 2 parameters
SPECTRAL_RADIUS_2 = 0.9
C_2 = 0.60
DECAY_RATE_2 = 0.85
W_IN_SCALE_2 = 0.08
W_SCALE_2 = 0.30
W_BACK_SCALE_2 = 0.56
W_IN_SPARSITY_2 = 0.60
W_SPARSITY_2 = 0.990

# Instantiate three frozen reservoirs
# TODO(alexochs): migrate mixture_of_reservoirs to the simplified helpers in single_reservoir_core.
reservoirs = [
    Reservoir(N, K, L, ReservoirParams(
        spectral_radius=SPECTRAL_RADIUS_0,
        C=C_0,
        decay_rate=DECAY_RATE_0,
        w_scale=W_SCALE_0,
        w_sparsity=W_SPARSITY_0,
        w_back_scale=W_BACK_SCALE_0,
        w_in_scale=W_IN_SCALE_0,
        w_in_sparsity=W_IN_SPARSITY_0,
    ), rng),
    Reservoir(N, K, L, ReservoirParams(
        spectral_radius=SPECTRAL_RADIUS_1,
        C=C_1,
        decay_rate=DECAY_RATE_1,
        w_scale=W_SCALE_1,
        w_sparsity=W_SPARSITY_1,
        w_back_scale=W_BACK_SCALE_1,
        w_in_scale=W_IN_SCALE_1,
        w_in_sparsity=W_IN_SPARSITY_1,
    ), rng),
    Reservoir(N, K, L, ReservoirParams(
        spectral_radius=SPECTRAL_RADIUS_2,
        C=C_2,
        decay_rate=DECAY_RATE_2,
        w_scale=W_SCALE_2,
        w_sparsity=W_SPARSITY_2,
        w_back_scale=W_BACK_SCALE_2,
        w_in_scale=W_IN_SCALE_2,
        w_in_sparsity=W_IN_SPARSITY_2,
    ), rng),
]

def prepare_em_readouts_only(lam: float = 1e-3) -> Tuple[List[np.ndarray], List[List[Dict[str, np.ndarray]]]]:
    """
    Precompute teacher-forced design matrices and initialize readouts equally.
    Returns ``(W_out_list, designs)``.
    """
    designs = build_window_designs(reservoirs, TRAIN_WINDOWS)
    W_out_list = init_readouts_unweighted(designs, lam)
    return W_out_list, designs


def run_one_em_round(W_out_list: List[np.ndarray],
                     designs: List[List[Dict[str, np.ndarray]]],
                     tau: float = 0.8,
                     lam: float = 1e-3,
                     horizon: int = 10,
                     prior: Optional[np.ndarray] = None,
                     ema_prev: Optional[np.ndarray] = None,
                     alpha: float = 0.2):
    """Run one EM round on TRAIN_WINDOWS and return updated weights, responsibilities, and errors."""
    return em_round_readouts_only(reservoirs, TRAIN_WINDOWS, designs, W_out_list, lam=lam, tau=tau, prior=prior, ema_prev=ema_prev, alpha=alpha, horizon=horizon)


if __name__ == "__main__":
    import time
    np.set_printoptions(precision=4, suppress=True)

    # Build fixed windows
    DATA_SPLITS = _build_fixed_windows(N_WINDOWS_PER_REGIME, WINDOW_T, rng)
    TRAIN_WINDOWS = DATA_SPLITS['train']
    TEST_WINDOWS  = DATA_SPLITS['test']

    lam = 1e-3 # ridge regression term
    tau = 0.8 # temperature for softmax
    horizon = 10 # predict 10 steps out (for now)

    t0 = time.time()
    W_out_list, designs = prepare_em_readouts_only(lam=lam)
    t1 = time.time()
    print(f"Prepared in {t1 - t0:.2f}s. Windows: {len(TRAIN_WINDOWS)} | Experts: {len(reservoirs)} | N={N}")

    # Sanity: shapes
    print("W_out shapes:", [w.shape for w in W_out_list])

    print("Running one EM round (readouts only)...")
    t2 = time.time()
    W_out_list, r, e = run_one_em_round(W_out_list, designs, tau=tau, lam=lam, horizon=horizon)
    t3 = time.time()
    print(f"EM round done in {t3 - t2:.2f}s")

    # Diagnostics
    e_mean_per_expert = e.mean(axis=0)
    r_mean_per_expert = r.mean(axis=0)
    print("Mean NRMSE@10 per expert:", e_mean_per_expert)
    print("Mean responsibility per expert:", r_mean_per_expert)

    # Simple load-balance indicator
    print("Load-balance (should be ~1/M early):", r_mean_per_expert)

    # Top-1 assignment histogram
    top1 = np.argmax(r, axis=1)
    hist = np.bincount(top1, minlength=len(reservoirs))
    print("Top-1 assignment counts:", hist.tolist())

    # Peek first 5 responsibility rows
    print("Sample responsibilities (first 5 windows):\n", r[:5])

    # Quick check: recompute errors after refit to see if mean error decreased
    e_after = compute_errors_matrix(reservoirs, TRAIN_WINDOWS, W_out_list, horizon=horizon)
    print("Mean NRMSE@10 per expert after refit:", e_after.mean(axis=0))
