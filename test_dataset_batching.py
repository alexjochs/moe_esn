# imports
import numpy as np
from dataset import generate_lorenz, generate_mackey_glass, generate_rossler, Regime

# constants
rng = np.random.default_rng(42)

# dataset Constants
N_DISCARD = 100  # discard first 100 states
TRAIN_FRACTION = 0.7 
TEST_FRACTION = 0.3
N_WARMUP = 1_000  # free-run steps before scoring during test
H = 15_000        # max timestep

# generate dataset (mixture of mackey glass, rossler, lorenz)
# Window spec: [warmup | teacher-forced fit | eval (free-run)]
W = 200
T_FIT = 200   # set to 0 if you really want only warmup+eval (not recommended)
H_MAX = 50
WINDOW_T = W + T_FIT + H_MAX
N_WINDOWS_PER_REGIME = 200  # tweak for your compute

def _gen_regime_window(regime_id, T, rng):
    """
    Return y:(T,1) for a single pure-regime rollout using the standardized output
    (SECOND return value) from each generator.
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

def _build_fixed_windows(n_per_regime, T, rng):
    """
    Build small, pure-regime windows with index markers for warmup/fit/eval.
    Returns dict with keys 'train' and 'test'; each is a list of dicts:
      { 'y': (T,1), 'idx_warmup_end': W, 'idx_fit_end': W+T_FIT, 'idx_eval_end': T, 'regime': r }
    Split is a simple per-regime 70/30 without shuffling fuss.
    """
    all_windows = []
    for r in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        for _ in range(n_per_regime):
            y = _gen_regime_window(r, T, rng)
            all_windows.append({
                'y': y,
                'idx_warmup_end': W,
                'idx_fit_end': W + T_FIT,
                'idx_eval_end': T,
                'regime': r,
            })
    # per-regime 70/30 split
    train, test = [], []
    for r in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        Rs = [w for w in all_windows if w['regime'] == r]
        n_tr = int(0.7 * len(Rs))
        train += Rs[:n_tr]
        test  += Rs[n_tr:]
    return {'train': train, 'test': test}

# Build fixed windows (replaces any mixed-stream carving, but doesn’t remove it)
DATA_SPLITS = _build_fixed_windows(N_WINDOWS_PER_REGIME, WINDOW_T, rng)
TRAIN_WINDOWS = DATA_SPLITS['train']
TEST_WINDOWS  = DATA_SPLITS['test']
print(len(TRAIN_WINDOWS))