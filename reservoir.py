import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

def _spectral_radius(mat: np.ndarray) -> float:
    # For sparse-like tri-valued matrices this is OK for N ~ 1e3
    vals = np.linalg.eigvals(mat)
    return float(np.max(np.abs(vals))) if vals.size > 0 else 0.0


@dataclass
class ReservoirParams:
    spectral_radius: float
    leak: float                  # a.k.a. decay rate in [0,1]
    w_in_scale: float
    w_scale: float
    w_back_scale: float
    w_in_sparsity: float         # prob of zero in W_in
    w_sparsity: float            # prob of zero in W


class Reservoir:
    """Frozen reservoir with teacher forcing for warmup/fit.

    State update:
        x[t+1] = (1 - leak)*x[t] + leak * tanh(W @ x[t] + W_in @ u[t] + W_back @ y_prev[t])
    During warmup and fit, y_prev[t] = y_true[t]. During eval, y_prev[t] = y_hat[t].
    """
    def __init__(self, N: int, K: int, L: int, params: ReservoirParams, rng: np.random.Generator):
        self.N, self.K, self.L = N, K, L
        self.params = params
        p = params

        # Input matrix with tri-valued sampling and sparsity
        self.W_in = rng.choice([0.0, -p.w_in_scale, p.w_in_scale], size=(N, K),
                               p=[p.w_in_sparsity, (1 - p.w_in_sparsity) / 2, (1 - p.w_in_sparsity) / 2]).astype(np.float32)
        # Recurrent matrix with sparsity
        self.W = rng.choice([0.0, -p.w_scale, p.w_scale], size=(N, N),
                            p=[p.w_sparsity, (1 - p.w_sparsity) / 2, (1 - p.w_sparsity) / 2]).astype(np.float32)
        # Spectral normalization
        rho = _spectral_radius(self.W)
        if rho > 0:
            self.W *= (p.spectral_radius / rho)

        # Output feedback
        self.W_back = rng.uniform(-p.w_back_scale, p.w_back_scale, size=(N, L)).astype(np.float32)

        # State
        self.x0 = np.zeros((N,), dtype=np.float32)

    def _step(self, x: np.ndarray, u_t: np.ndarray, y_prev_t: np.ndarray) -> np.ndarray:
        p = self.params
        pre = self.W @ x + self.W_in @ u_t + self.W_back @ y_prev_t
        x_new = (1.0 - p.leak) * x + p.leak * np.tanh(pre)
        return x_new.astype(np.float32)

    def run_window(self, window: Dict, W_out: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Run one window with teacher forcing through warmup+fit and optionally free-run for eval
        if W_out is provided. Returns sufficient pieces for readout fitting.

        Returns dict with keys:
            'H_fit': (T_fit, N)     reservoir states for ridge
            'Y_fit': (T_fit, L)     supervised targets matching H_fit rows
            # Eval outputs reserved for later when EM is implemented
        """
        y = window['y']  # shape (T,1)
        T = y.shape[0]
        assert T >= window['idx_fit_end']

        x = self.x0.copy()
        H_rows = []

        # Warmup and Fit spans with teacher forcing
        idx_wu = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']

        # Warmup
        for t in range(idx_wu):
            u_t = y[t:t+1, :].reshape(self.K)  # here input equals target; adapt if external input later
            y_prev_t = y[t:t+1, :].reshape(self.L)
            x = self._step(x, u_t, y_prev_t)

        # Fit span
        for t in range(idx_wu, idx_fit_end):
            u_t = y[t:t+1, :].reshape(self.K)
            y_prev_t = y[t:t+1, :].reshape(self.L)
            x = self._step(x, u_t, y_prev_t)
            H_rows.append(x.copy())

        H_fit = np.stack(H_rows, axis=0) if H_rows else np.zeros((0, self.N), dtype=np.float32)
        Y_fit = y[idx_wu:idx_fit_end, :].astype(np.float32)
        return {'H_fit': H_fit, 'Y_fit': Y_fit}

    def sufficient_stats(self, window: Dict, cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (S, b) = (H^T H, H^T Y) for the fit span. Optionally caches by window id."""
        wid = window.get('id', None)
        if cache is not None and wid is not None and wid in cache:
            return cache[wid]
        out = self.run_window(window)
        H, Y = out['H_fit'], out['Y_fit']
        S = H.T @ H
        b = H.T @ Y
        if cache is not None and wid is not None:
            cache[wid] = (S, b)
        return S, b

    def free_run(self, window: Dict, W_out: np.ndarray, horizon: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run warmup+fit with teacher forcing to get reservoir state at fit boundary,
        then run closed-loop free-run for `horizon` steps using predictions as inputs and feedback.

        Returns:
            y_hat: (horizon, L) predicted outputs
            y_true: (horizon, L) true outputs from window after fit span
        """
        y = window['y']  # shape (T, L)
        idx_wu = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']
        T = y.shape[0]
        assert T >= idx_fit_end + horizon

        x = self.x0.copy()

        # Warmup
        for t in range(idx_wu):
            u_t = y[t:t+1, :].reshape(self.K)
            y_prev_t = y[t:t+1, :].reshape(self.L)
            x = self._step(x, u_t, y_prev_t)

        # Fit span with teacher forcing
        for t in range(idx_wu, idx_fit_end):
            u_t = y[t:t+1, :].reshape(self.K)
            y_prev_t = y[t:t+1, :].reshape(self.L)
            x = self._step(x, u_t, y_prev_t)

        y_hat = np.zeros((horizon, self.L), dtype=np.float32)
        # Closed-loop free-run
        y_prev_t = y[idx_fit_end-1:idx_fit_end, :].reshape(self.L)  # last true output at fit boundary
        for h in range(horizon):
            u_t = y_hat[h-1] if h > 0 else y[idx_fit_end-1].reshape(self.K)  # for h=0 use last true output as input
            x = self._step(x, u_t, y_prev_t)
            y_hat[h] = W_out.T @ x
            y_prev_t = y_hat[h]

        y_true = y[idx_fit_end:idx_fit_end + horizon, :].astype(np.float32)
        return y_hat, y_true