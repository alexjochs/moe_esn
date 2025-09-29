import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from enum import IntEnum

# ===== Tunable constants =====
REGIME_LEN = 500          # points per regime
CROSSFADE_LEN = 10        # points blended at each transition
SEED = 42                 # RNG seed for reproducibility
REGIME_SEQUENCE = ("mackey_glass", "lorenz", "rossler")  # repeating order
TOTAL_POINTS_PLOT = 5000  # default length to visualize at the bottom
SCALE_MARGIN = 0.95       # shrink to keep values strictly within (-1, 1)
# =============================

class Regime(IntEnum):
    MACKEY_GLASS = 0
    LORENZ = 1
    ROSSLER = 2
    TRANSITION = 3

def zscore_standardize(x):
    mean = np.mean(x)
    std = np.std(x)
    eps = 1e-10
    return (x - mean) / (std + eps)

def generate_lorenz(H=3000, sigma=10.0, rho=28.0, beta=8/3, dt=0.01, subsample=10, rng=None):
    """Generate Lorenz system time series using RK4 integration.
    Returns:
        x_raw: Raw x-component values (length H after subsampling)
        x_std: Standardized values with zero mean and unit variance
    """
    if rng is None:
        rng = np.random.default_rng(42)
    H_fine = H * subsample
    # Initial state: (1,1,1) + small noise
    state = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
    data = np.zeros((H_fine + 1, 3), dtype=float)
    data[0] = state
    for i in range(H_fine):
        x, y, z = data[i]
        def f(s):
            x, y, z = s
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])
        k1 = f(data[i])
        k2 = f(data[i] + 0.5 * dt * k1)
        k3 = f(data[i] + 0.5 * dt * k2)
        k4 = f(data[i] + dt * k3)
        data[i + 1] = data[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Subsample and return x-component
    x_raw = data[::subsample, 0][:H]
    x_std = zscore_standardize(x_raw)
    return x_raw, x_std

def generate_rossler(H=3000, a=0.2, b=0.2, c=5.7, dt=0.01, subsample=10, rng=None):
    """Generate Rössler system time series using RK4 integration.
    Returns:
        x_raw: Raw x-component values (length H after subsampling)
        x_std: Standardized values with zero mean and unit variance
    """
    if rng is None:
        rng = np.random.default_rng(42)
    H_fine = H * subsample
    # Initial state near (0,0,0) with small noise
    state = np.array([0.0, 0.0, 0.0]) + 0.01 * rng.standard_normal(3)
    data = np.zeros((H_fine + 1, 3), dtype=float)
    data[0] = state
    for i in range(H_fine):
        x, y, z = data[i]
        def f(s):
            x, y, z = s
            dx = -(y + z)
            dy = x + a * y
            dz = b + z * (x - c)
            return np.array([dx, dy, dz])
        k1 = f(data[i])
        k2 = f(data[i] + 0.5 * dt * k1)
        k3 = f(data[i] + 0.5 * dt * k2)
        k4 = f(data[i] + dt * k3)
        data[i + 1] = data[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Subsample and return x-component
    x_raw = data[::subsample, 0][:H]
    x_std = zscore_standardize(x_raw)
    return x_raw, x_std

def generate_mackey_glass(H=3000, TAU=30, N_EXP=10, BETA=0.2, GAMMA=0.1, DELTA=0.1, subsample=10, rng=None):
    """Generate Mackey-Glass chaotic time series.
    
    Returns:
        mg_raw: Raw Mackey-Glass values
        mg_std: Standardized values with zero mean and unit variance
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    H_fine = H * subsample
    TAU_steps = int(TAU / DELTA)
    
    mg_fine = np.zeros(H_fine + TAU_steps + 1, dtype=float)
    mg_fine[:TAU_steps] = 1.2 + 0.01 * rng.standard_normal(TAU_steps)
    
    for i in range(TAU_steps, H_fine + TAU_steps):
        delayed = mg_fine[i - TAU_steps]
        mg_fine[i + 1] = mg_fine[i] + DELTA * (BETA * (delayed / (1.0 + delayed ** N_EXP)) - GAMMA * mg_fine[i])
    
    mg_raw = mg_fine[TAU_steps::subsample][:H]
    mg_std = zscore_standardize(mg_raw)
    
    return mg_raw, mg_std

def generate_dataset(total_points=TOTAL_POINTS_PLOT,
                     regime_sequence=REGIME_SEQUENCE,
                     regime_len=REGIME_LEN,
                     crossfade_len=CROSSFADE_LEN,
                     rng_seed=SEED):
    """Create an interleaved standardized time series and per-point labels.

    Strategy:
      1) Pre-generate a long, standardized buffer for each system. No reuse within a call.
      2) Interleave non-overlapping windows of length `regime_len` according to `regime_sequence`.
      3) Apply a linear crossfade over `crossfade_len` points at regime boundaries.

    Returns:
        y: np.ndarray, shape (total_points,)
        labels: np.ndarray[int32], shape (total_points,), values from Regime enum
        Output y is globally centered and max-abs scaled with margin SCALE_MARGIN so that |y| < 1.
    """
    assert crossfade_len >= 0 and crossfade_len < regime_len, "CROSSFADE_LEN must be in [0, REGIME_LEN)"

    name_to_enum = {
        "mackey_glass": Regime.MACKEY_GLASS,
        "lorenz": Regime.LORENZ,
        "rossler": Regime.ROSSLER,
    }

    # How many regime windows will we place?
    n_segments = int(np.ceil(total_points / regime_len))
    # Count how many segments per system are needed
    counts = {k: 0 for k in name_to_enum}
    for i in range(n_segments):
        sys_name = regime_sequence[i % len(regime_sequence)]
        if sys_name not in counts:
            raise ValueError(f"Unknown regime '{sys_name}' in regime_sequence")
        counts[sys_name] += 1

    # RNG shared across generators for reproducibility
    rng = np.random.default_rng(rng_seed)

    # Map names to generator callables returning standardized series
    def _gen(name, n):
        if name == "mackey_glass":
            _, x = generate_mackey_glass(H=n, rng=rng)
        elif name == "lorenz":
            _, x = generate_lorenz(H=n, rng=rng)
        elif name == "rossler":
            _, x = generate_rossler(H=n, rng=rng)
        else:
            raise ValueError(f"Unknown regime '{name}'")
        return x

    # Build non-overlapping source buffers long enough for all slices
    sources = {}
    ptrs = {}
    # Allocate ~1/2 of total per system. With three regimes (each ~1/3), this comfortably avoids reuse in these use cases.
    for name, c in counts.items():
        sources[name] = _gen(name, total_points // 2)
        ptrs[name] = 0

    out_segments = []
    label_segments = []

    # Helper to append data and labels
    def _append_segment(x, label_value):
        out_segments.append(x)
        label_segments.append(np.full(x.shape, label_value, dtype=np.int32))

    # Warm-up skip: advance Mackey-Glass by 50 samples to avoid early transient.
    ptrs[regime_sequence[Regime.MACKEY_GLASS]] = 50
    # First segment
    seg_idx = 0
    name_a = regime_sequence[seg_idx % len(regime_sequence)]
    a_buf = sources[name_a]
    a_ptr = ptrs[name_a]
    a = a_buf[a_ptr:a_ptr + regime_len]
    if a.shape[0] < regime_len:
        raise ValueError(f"Insufficient source buffer for '{name_a}'. Increase total_points or reduce regime_len.")
    ptrs[name_a] += regime_len

    if crossfade_len > 0:
        core_a = a[:regime_len - crossfade_len]
        tail_a = a[regime_len - crossfade_len:regime_len]
        _append_segment(core_a, name_to_enum[name_a])
    else:
        _append_segment(a, name_to_enum[name_a])
        tail_a = None

    produced = sum(len(s) for s in out_segments)

    # Add remaining segments with crossfades
    while produced < total_points:
        seg_idx += 1
        name_b = regime_sequence[seg_idx % len(regime_sequence)]

        b_buf = sources[name_b]
        b_ptr = ptrs[name_b]
        b = b_buf[b_ptr:b_ptr + regime_len]
        if b.shape[0] < regime_len:
            raise ValueError(f"Insufficient source buffer for '{name_b}'. Increase total_points or reduce regime_len.")
        ptrs[name_b] += regime_len

        if crossfade_len > 0:
            head_b = b[:crossfade_len]
            w = np.linspace(1.0, 0.0, crossfade_len)
            blended = w * tail_a + (1.0 - w) * head_b
            _append_segment(blended, Regime.TRANSITION)
            _append_segment(b[crossfade_len:], name_to_enum[name_b])
            tail_a = b[regime_len - crossfade_len:regime_len]
        else:
            _append_segment(b, name_to_enum[name_b])

        produced = sum(len(s) for s in out_segments)

    y = np.concatenate(out_segments)
    labels = np.concatenate(label_segments)

    # Trim to exact length
    y = y[:total_points]
    labels = labels[:total_points]

    # Global centering and max-abs scaling with a safety margin
    # This preserves dynamics and guarantees inputs fall strictly inside (-1, 1)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    absmax = np.max(np.abs(y_centered)) + 1e-12  # avoid divide-by-zero
    y = (SCALE_MARGIN / absmax) * y_centered

    return y, labels

if __name__ == "__main__":
    # Generate a single MG, Lorenz, and Rossler series for comparison
    H = 1000
    mg_raw, mg_std = generate_mackey_glass(H=H, TAU=17, BETA=1.0)
    lorenz_raw, lorenz_std = generate_lorenz(H=H)
    rossler_raw, rossler_std = generate_rossler(H=H)

    # Time-series overlay plot
    plt.figure(figsize=(10, 5))
    plt.plot(mg_std, label="Mackey-Glass (std)")
    plt.plot(lorenz_std, label="Lorenz x (std)")
    plt.plot(rossler_std, label="Rössler x (std)")
    plt.title("Time Series Overlay: Mackey-Glass vs Lorenz x vs Rössler x")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Spectrogram comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].specgram(mg_std, NFFT=256, Fs=1, noverlap=128)
    axes[0].set_title("Mackey-Glass Spectrogram")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Frequency")
    axes[1].specgram(lorenz_std, NFFT=256, Fs=1, noverlap=128)
    axes[1].set_title("Lorenz x Spectrogram")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Frequency")
    axes[2].specgram(rossler_std, NFFT=256, Fs=1, noverlap=128)
    axes[2].set_title("Rössler x Spectrogram")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # PSD comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    freqs_mg, Pxx_mg = welch(mg_std, fs=1, nperseg=256)
    freqs_lor, Pxx_lor = welch(lorenz_std, fs=1, nperseg=256)
    freqs_ros, Pxx_ros = welch(rossler_std, fs=1, nperseg=256)

    # Align PSD subplot scaling so cross-series differences are visually comparable
    positive_psd = np.concatenate([
        Pxx_mg[Pxx_mg > 0],
        Pxx_lor[Pxx_lor > 0],
        Pxx_ros[Pxx_ros > 0],
    ])
    y_min, y_max = positive_psd.min(), positive_psd.max()

    axes[0].semilogy(freqs_mg, Pxx_mg)
    axes[0].set_title("Mackey-Glass PSD")
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Power Spectral Density")
    axes[1].semilogy(freqs_lor, Pxx_lor)
    axes[1].set_title("Lorenz x PSD")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Power Spectral Density")
    axes[2].semilogy(freqs_ros, Pxx_ros)
    axes[2].set_title("Rössler x PSD")
    axes[2].set_xlabel("Frequency")
    axes[2].set_ylabel("Power Spectral Density")
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

    # Interleaved dataset demo
    ds, ds_labels = generate_dataset(total_points=TOTAL_POINTS_PLOT,
                                     regime_sequence=REGIME_SEQUENCE,
                                     regime_len=REGIME_LEN,
                                     crossfade_len=CROSSFADE_LEN,
                                     rng_seed=SEED)
    plt.figure(figsize=(10, 4))
    plt.plot(ds[:TOTAL_POINTS_PLOT])
    plt.title("Interleaved Dataset (first 5000 points)")
    plt.xlabel("Time step")
    plt.ylabel("Standardized value")
    plt.tight_layout()
    plt.show()
