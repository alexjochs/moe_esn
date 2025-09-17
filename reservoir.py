import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


def _scale_by_spectral_norm(W_mat: np.ndarray, target_rho: float, n_iter: int = 50,
                            rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
    """Scale ``W_mat`` so its spectral norm approximates ``target_rho`` using power iteration."""
    if rng is None:
        rng = np.random.default_rng(42)
    v = rng.normal(size=W_mat.shape[1])
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        v = W_mat.T @ (W_mat @ v)
        v_norm = np.linalg.norm(v) + 1e-12
        v = v / v_norm
    sigma = float(np.sqrt(v @ (W_mat.T @ (W_mat @ v))))
    if sigma > 0:
        return W_mat * (target_rho / sigma), sigma
    return W_mat, 0.0


@dataclass
class ReservoirParams:
    spectral_radius: float
    C: float
    decay_rate: float              # effective leak factor combines C * decay_rate
    w_scale: float
    w_sparsity: float              # probability of zero in W
    w_back_scale: float
    w_in_scale: float = 0.0
    w_in_sparsity: float = 1.0     # probability of zero in W_in
    bias_value: float = 0.0        # constant input injected via W_in distribution


class Reservoir:
    """Frozen reservoir with teacher forcing helpers and closed-loop rollout."""

    def __init__(self, N: int, K: int, L: int, params: ReservoirParams, rng: np.random.Generator):
        self.N, self.K, self.L = N, K, L
        self.params = params
        self.rng = rng

        p = params

        # Optional input matrix (may be empty when K == 0)
        if self.K > 0 and p.w_in_scale > 0.0:
            p0 = np.clip(p.w_in_sparsity, 0.0, 1.0)
            nonzero_prob = 1.0 - p0
            self.W_in = rng.choice(
                [0.0, -p.w_in_scale, p.w_in_scale],
                size=(N, K),
                p=[p0, nonzero_prob / 2.0, nonzero_prob / 2.0],
            ).astype(np.float32)
        elif self.K > 0:
            self.W_in = np.zeros((N, K), dtype=np.float32)
        else:
            self.W_in = None

        self.bias_weights = None
        if p.bias_value != 0.0:
            p0 = np.clip(p.w_in_sparsity, 0.0, 1.0)
            nonzero_prob = 1.0 - p0
            self.bias_weights = rng.choice(
                [0.0, -p.w_in_scale, p.w_in_scale],
                size=(N,),
                p=[p0, nonzero_prob / 2.0, nonzero_prob / 2.0],
            ).astype(np.float32)

        # Recurrent matrix with tri-valued sparsity pattern
        p0 = np.clip(p.w_sparsity, 0.0, 1.0)
        nonzero_prob = 1.0 - p0
        self.W = rng.choice(
            [0.0, -p.w_scale, p.w_scale],
            size=(N, N),
            p=[p0, nonzero_prob / 2.0, nonzero_prob / 2.0],
        ).astype(np.float32)
        self.W, self._spectral_sigma = _scale_by_spectral_norm(self.W, target_rho=p.spectral_radius, rng=rng)

        # Output feedback matrix
        self.W_back = rng.uniform(-p.w_back_scale, p.w_back_scale, size=(N, L)).astype(np.float32)

        self.x0 = np.zeros((N,), dtype=np.float32)
        self._zero_u = np.zeros((self.K,), dtype=np.float32)
        self._zero_y = np.zeros((self.L,), dtype=np.float32)
        self._effective_leak = np.clip(p.C * p.decay_rate, 0.0, 1.0)

    def _step(self, x: np.ndarray, u_t: Optional[np.ndarray], y_prev_t: Optional[np.ndarray]) -> np.ndarray:
        p = self.params
        pre = self.W @ x
        if self.W_in is not None and u_t is not None:
            pre += self.W_in @ u_t
        if self.bias_weights is not None:
            pre += self.bias_weights * p.bias_value
        if y_prev_t is not None:
            pre += self.W_back @ y_prev_t
        return ((1.0 - self._effective_leak) * x + p.C * np.tanh(pre)).astype(np.float32)

    def _input_at(self, arr: np.ndarray, idx: int) -> np.ndarray:
        if self.K == 0:
            return self._zero_u
        return arr[idx:idx + 1, :].reshape(self.K)

    def _target_at(self, arr: np.ndarray, idx: int) -> np.ndarray:
        if self.L == 0:
            return self._zero_y
        return arr[idx:idx + 1, :].reshape(self.L)

    def run_window(self, window: Dict, W_out: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Teacher-force through warmup+fit spans and optionally return free-run outputs."""
        y = window['y']  # shape (T, L)
        T = y.shape[0]
        idx_wu = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']
        assert T >= idx_fit_end

        x = self.x0.copy()
        states: List[np.ndarray] = []

        for t in range(idx_wu):
            x = self._step(x, self._input_at(y, t), self._target_at(y, t))

        for t in range(idx_wu, idx_fit_end):
            x = self._step(x, self._input_at(y, t), self._target_at(y, t))
            states.append(x.copy())

        H_fit = np.stack(states, axis=0) if states else np.zeros((0, self.N), dtype=np.float32)
        Y_fit = y[idx_wu:idx_fit_end, :].astype(np.float32)

        out = {'H_fit': H_fit, 'Y_fit': Y_fit}

        if W_out is not None and 'idx_eval_end' in window:
            horizon = window['idx_eval_end'] - idx_fit_end
            y_hat, y_true = self.free_run(window, W_out, horizon=horizon)
            out['y_hat'] = y_hat
            out['y_true'] = y_true

        return out

    def sufficient_stats(self, window: Dict, cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(H^T H, H^T Y)`` for the fit span (cache optional)."""
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
        """Closed-loop rollout starting from the fit boundary for ``horizon`` steps."""
        y = window['y']
        idx_wu = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']
        T = y.shape[0]
        assert T >= idx_fit_end + horizon

        x = self.x0.copy()

        for t in range(idx_wu):
            x = self._step(x, self._input_at(y, t), self._target_at(y, t))

        for t in range(idx_wu, idx_fit_end):
            x = self._step(x, self._input_at(y, t), self._target_at(y, t))

        y_hat = np.zeros((horizon, self.L), dtype=np.float32)
        y_prev = self._target_at(y, idx_fit_end - 1)
        for h in range(horizon):
            u_t = y_hat[h - 1] if (h > 0 and self.K > 0) else (self._input_at(y, idx_fit_end - 1) if self.K > 0 else self._zero_u)
            x = self._step(x, u_t if self.K > 0 else self._zero_u, y_prev)
            y_hat[h] = (W_out @ x).reshape(self.L)
            y_prev = y_hat[h]

        y_true = y[idx_fit_end:idx_fit_end + horizon, :].astype(np.float32)
        return y_hat, y_true
