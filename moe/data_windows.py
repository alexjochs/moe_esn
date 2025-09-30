import numpy as np
from typing import Dict, List

from dataset import generate_lorenz, generate_mackey_glass, generate_rossler, Regime


def _gen_regime_window(regime_id: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Return a single standardized rollout for the requested regime."""
    if regime_id == int(Regime.MACKEY_GLASS):
        _, y_std = generate_mackey_glass(H=T, rng=rng)
    elif regime_id == int(Regime.LORENZ):
        _, y_std = generate_lorenz(H=T, rng=rng)
    elif regime_id == int(Regime.ROSSLER):
        _, y_std = generate_rossler(H=T, rng=rng)
    else:
        raise ValueError("Unknown regime id")
    return y_std.reshape(T, 1).astype(np.float32)


def _build_fixed_windows(n_per_regime: int,
                         rng: np.random.Generator,
                         warmup_len: int,
                         teacher_forced_len: int,
                         window_len_total: int) -> List[Dict]:
    """Build per-regime windows with warmup/fit/eval markers; returns train set only."""
    all_windows = []
    uid = 0
    for regime_number in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        for _ in range(n_per_regime):
            y = _gen_regime_window(regime_number, window_len_total, rng)
            all_windows.append({
                'y': y,
                'idx_warmup_end': warmup_len,
                'idx_fit_end': warmup_len + teacher_forced_len,
                'idx_eval_end': window_len_total,
                'regime': regime_number,
                'id': uid,
            })
            uid += 1

    train: List[Dict] = []
    for regime_number in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        regime_windows = [window for window in all_windows if window['regime'] == regime_number]
        n_train = int(0.7 * len(regime_windows))
        train += regime_windows[:n_train]
    return train


__all__ = [
    "_gen_regime_window",
    "_build_fixed_windows",
]
