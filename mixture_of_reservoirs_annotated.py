# imports
import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

from dataset import generate_lorenz, generate_mackey_glass, generate_rossler, Regime
from reservoir import Reservoir, ReservoirParams
from single_reservoir_core import teacher_forced_states, fit_linear_readout


rng = np.random.default_rng(42)

# -----------------------------------------------------------------------------
# Dataset windowing
#   Window spec: [warmup | teacher-forced fit | eval (free-run)]
# -----------------------------------------------------------------------------
WARMUP_LEN = 200                # warmup length
TEACHER_FORCED_LEN = 400            # teacher-forced fit span; set to 0 to disable
MAX_EVAL_HORIZON = 50               # extra tail so free-run rollouts have targets
WINDOW_LEN_TOTAL = WARMUP_LEN + TEACHER_FORCED_LEN + MAX_EVAL_HORIZON
N_WINDOWS_PER_REGIME = 200

NUM_TRAINING_ITERATIONS = 50
HYPERPARAM_INTERPOLATION_POINTS = 20
HYPERPARAM_SAMPLE_BUDGET = 400

# Dask cluster defaults (tuned for the EECS preempt partition; tweak as needed).
DASK_JOBS = int(os.environ.get("ESN_DASK_JOBS", 200))
DASK_WORKER_CORES = int(os.environ.get("ESN_DASK_WORKER_CORES", 1))
DASK_WORKER_MEM = os.environ.get("ESN_DASK_WORKER_MEM", "2GB")
DASK_WORKER_WALLTIME = os.environ.get("ESN_DASK_WORKER_WALLTIME", "04:00:00")
DASK_PARTITION = os.environ.get("ESN_DASK_PARTITION", "preempt")
DASK_ACCOUNT = os.environ.get("ESN_DASK_ACCOUNT", "eecs")
DASK_REQUEUE = os.environ.get("ESN_DASK_REQUEUE", "1") != "0"
DASK_PROCESSES_PER_JOB = int(os.environ.get("ESN_DASK_PROCESSES_PER_JOB", DASK_WORKER_CORES))

RUN_LOG_FILENAME = "iteration_log.jsonl"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


def _build_fixed_windows(n_per_regime: int, rng: np.random.Generator) -> Dict[str, list]:
    """
    Build pure-regime windows with index markers for warmup/fit/eval.
    Returns dict with keys 'train' and 'test'; each is a list of dicts:
      { 'y': (T,1), 'idx_warmup_end': W, 'idx_fit_end': W+T_FIT, 'idx_eval_end': T, 'regime': r, 'id': int }
    Split is per-regime 70/30 without shuffling fuss.

    god I hate chatGPT's variable naming.
    n_per_regime is number of windows per regime (MG, Rossler, Lorenz)
    """
    all_windows = []
    uid = 0
    for regime_number in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        for _ in range(n_per_regime):
            y = _gen_regime_window(regime_number, WINDOW_LEN_TOTAL, rng)
            all_windows.append({
                'y': y,
                'idx_warmup_end': WARMUP_LEN,
                'idx_fit_end': WARMUP_LEN + TEACHER_FORCED_LEN,
                'idx_eval_end': WINDOW_LEN_TOTAL,
                'regime': regime_number,
                'id': uid,
            })
            uid += 1

    # per-regime 70/30 split
    train, test = [], []
    for regime_number in (int(Regime.MACKEY_GLASS), int(Regime.LORENZ), int(Regime.ROSSLER)):
        regime_windows = [window for window in all_windows if window['regime'] == regime_number]
        n_train = int(0.7 * len(regime_windows))
        train += regime_windows[:n_train]
        test  += regime_windows[n_train:]
    # returns a dictionary of train windows and test windows, each window is a dictionary of data (Y) and some important indexes and regime id
    return {'train': train, 'test': test}


# -----------------------------------------------------------------------------
# Utility functions for Step 3 (EM round with readouts only)
# -----------------------------------------------------------------------------

def compute_nrmse(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute normalized root mean squared error (NRMSE)
    """
    mse = np.mean((preds - targets) ** 2)
    var = np.var(targets)
    return np.sqrt(mse / (var + eps))


def soft_responsibilities(errors: np.ndarray, tau: float, prior: np.ndarray | None = None,
                          previous_responsibilities: np.ndarray | None = None, alpha: float = 0.2) -> np.ndarray:
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
    responsibilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if previous_responsibilities is not None:
        responsibilities = alpha * responsibilities + (1 - alpha) * previous_responsibilities
        # re-normalize to ensure total responsiblities add to 1
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    return responsibilities


def build_state_target_pairs(reservoirs: List[Reservoir], windows: List[Dict]) -> List[List[Dict[str, np.ndarray]]]:
    """runs each reservoir, and saves the states of the reservoir and the target for each timestep during teacher forcing"""
    state_target_pairs: List[List[Dict[str, np.ndarray]]] = []
    for reservoir in reservoirs:
        reservoir_state_target_pairs: List[Dict[str, np.ndarray]] = []
        for window in windows:
            targets = window['y'].astype(np.float32)
            states = teacher_forced_states(reservoir, targets.T)
            idx_warmup_end = window['idx_warmup_end']
            idx_fit_end = window['idx_fit_end']
            # Align states with labels: column t+1 corresponds to label at index t.
            X_n_fit = states[:, idx_warmup_end + 1:idx_fit_end + 1].T.astype(np.float32)
            Y_teach_fit = targets[idx_warmup_end:idx_fit_end, :].astype(np.float32)
            reservoir_state_target_pairs.append({'X_n_fit': X_n_fit, 'Y_teach_fit': Y_teach_fit})
        state_target_pairs.append(reservoir_state_target_pairs)
    return state_target_pairs


def init_readouts_unweighted(state_target_pairs: List[List[Dict[str, np.ndarray]]], lam: float) -> List[np.ndarray]:
    """Solve unweighted ridge (each regime is weighted equally) per expert using concatenated State/Target pairs."""
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
    # returns readout matrices for each reservoir/expert, should be len=3
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
            X_n_fit = row['X_n_fit']
            if X_n_fit.size == 0:
                continue
            Y_fit = row['Y_teach_fit']
            H_blocks.append(X_n_fit)
            Y_blocks.append(Y_fit)
            sample_weights.append(np.full(X_n_fit.shape[0], weight, dtype=np.float32))
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
    """Return errors matrix e[sequence_index,reservoir_index] = NRMSE@horizon for each window and expert.
                [Reservoir1 Reservoir2 Reservoir3]
    [sequence1] [0.1        0.2        0.1      ]
    [sequence2] [0.3        0.1        0.9      ]
    ...         ...         ...        ...
    """
    errors = np.zeros((len(windows), len(reservoirs)), dtype=np.float32)
    for sequence_index, window in enumerate(windows):
        for reservoir_index, reservoir in enumerate(reservoirs):
            y_hat, y_true = reservoir.free_run(window, W_out_list[reservoir_index], horizon=horizon)
            errors[sequence_index, reservoir_index] = compute_nrmse(y_hat, y_true)
    return errors


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


def _candidate_param_values(base: float, span: Tuple[float, float], interpolation_points: int, clip_to_unit: bool = False) -> np.ndarray:
    lo, hi = span
    values = np.linspace(lo, hi, interpolation_points, dtype=np.float64)
    if clip_to_unit:
        values = np.clip(values, 0.0, 1.0)
    return values


def _generate_param_candidates(base_params: ReservoirParams,
                               interpolation_points: int,
                               sample_budget: int,
                               rng_seed: Optional[int] = None) -> List[ReservoirParams]:
    """Build candidate list by sampling from the per-parameter interpolation grids."""
    rng_local = np.random.default_rng(rng_seed)

    grid_axes = [
        _candidate_param_values(base_params.spectral_radius, (0.5, 1.2), interpolation_points),
        _candidate_param_values(base_params.C, (0.4, 1.0), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.decay_rate, (0.6, 0.99), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.w_scale, (0.15, 0.5), interpolation_points),
        _candidate_param_values(base_params.w_sparsity, (0.94, 0.995), interpolation_points, clip_to_unit=True),
        _candidate_param_values(base_params.w_back_scale, (0.2, 0.8), interpolation_points),
        _candidate_param_values(base_params.w_in_scale, (0.02, 0.12), interpolation_points),
        _candidate_param_values(base_params.w_in_sparsity, (0.3, 0.9), interpolation_points, clip_to_unit=True),
    ]

    # Total combinations if we enumerated the full Cartesian grid.
    total_combinations = int(np.prod([axis.size for axis in grid_axes], dtype=np.int64))

    def candidate_from_tuple(values: Tuple[float, ...]) -> ReservoirParams:
        return ReservoirParams(
            spectral_radius=float(values[0]),
            C=float(values[1]),
            decay_rate=float(values[2]),
            w_scale=float(values[3]),
            w_sparsity=float(values[4]),
            w_back_scale=float(values[5]),
            w_in_scale=float(values[6]),
            w_in_sparsity=float(values[7]),
        )

    samples: Dict[Tuple[float, ...], ReservoirParams] = {}

    if total_combinations <= sample_budget:
        for combo in itertools.product(*grid_axes):
            samples.setdefault(tuple(round(value, 6) for value in combo), candidate_from_tuple(combo))
    else:
        while len(samples) < sample_budget:
            combo = tuple(axis[rng_local.integers(axis.size)] for axis in grid_axes)
            rounded = tuple(round(value, 6) for value in combo)
            if rounded not in samples:
                samples[rounded] = candidate_from_tuple(combo)

    # Ensure baseline params are present even if outside sampled set.
    baseline_tuple = (
        round(base_params.spectral_radius, 6),
        round(base_params.C, 6),
        round(base_params.decay_rate, 6),
        round(base_params.w_scale, 6),
        round(base_params.w_sparsity, 6),
        round(base_params.w_back_scale, 6),
        round(base_params.w_in_scale, 6),
        round(base_params.w_in_sparsity, 6),
    )
    samples.setdefault(baseline_tuple, ReservoirParams(**vars(base_params)))

    return list(samples.values())


def _evaluate_reservoir_candidate(reservoir_index: int,
                                  params: ReservoirParams,
                                  windows: List[Dict],
                                  responsibility_column: np.ndarray,
                                  lam: float,
                                  horizon: int,
                                  rng_seed: int) -> float:
    """Evaluate a single reservoir candidate and return responsibility-weighted NRMSE."""

    candidate_rng = np.random.default_rng(rng_seed)
    candidate_reservoir = Reservoir(N, K, L, params, candidate_rng)

    active_indices = np.where(responsibility_column > 1e-6)[0]
    if active_indices.size == 0:
        return float('inf')

    design_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []
    sample_weight_blocks: List[np.ndarray] = []

    for window_index in active_indices:
        weight = float(responsibility_column[window_index])
        if weight <= 0.0:
            continue
        window = windows[window_index]
        targets = window['y'].astype(np.float32)
        states = teacher_forced_states(candidate_reservoir, targets.T)
        idx_warmup_end = window['idx_warmup_end']
        idx_fit_end = window['idx_fit_end']
        X_fit = states[:, idx_warmup_end + 1:idx_fit_end + 1].T.astype(np.float32)
        if X_fit.size == 0:
            continue
        Y_fit = targets[idx_warmup_end:idx_fit_end, :].astype(np.float32)
        design_blocks.append(X_fit)
        target_blocks.append(Y_fit)
        sample_weight_blocks.append(np.full(X_fit.shape[0], weight, dtype=np.float32))

    if not design_blocks:
        return float('inf')

    design_matrix = np.vstack(design_blocks)
    target_matrix = np.vstack(target_blocks)
    sample_weights = np.concatenate(sample_weight_blocks)
    W_out = fit_linear_readout(design_matrix, target_matrix, alpha=lam, sample_weight=sample_weights)

    errors: List[float] = []
    error_weights: List[float] = []
    for window_index in active_indices:
        weight = float(responsibility_column[window_index])
        if weight <= 0.0:
            continue
        window = windows[window_index]
        y_hat, y_true = candidate_reservoir.free_run(window, W_out, horizon=horizon)
        error_value = compute_nrmse(y_hat, y_true)
        errors.append(error_value)
        error_weights.append(weight)

    if not errors:
        return float('inf')

    weighted_error = float(np.average(errors, weights=error_weights))
    return weighted_error


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
    sequence_error_matrix = compute_errors_matrix(reservoirs, windows, W_out_list, horizon=horizon)
    r = soft_responsibilities(sequence_error_matrix, tau=tau, prior=prior, previous_responsibilities=ema_prev, alpha=alpha)

    # M-step per expert
    W_new = refit_readouts_weighted(designs, r, lam, W_out_list)

    return W_new, r, sequence_error_matrix

def em_round_hyperparam_tuning(reservoirs: List[Reservoir],
                               windows: List[Dict],
                               responsibilities: np.ndarray,
                               lam: float,
                               horizon: int,
                               interpolation_points: int,
                               candidate_samples: int,
                               client: "Client") -> Tuple[List[ReservoirParams], List[float]]:
    """Sampled grid-search sweep over reservoir hyperparameters using responsibility weights."""
    tuned_params: List[ReservoirParams] = []
    tuned_errors: List[float] = []

    sample_budget = max(1, candidate_samples)

    windows_future = client.scatter(windows, broadcast=True)

    for reservoir_index, reservoir in enumerate(reservoirs):
        base_params = reservoir.params
        candidate_seed = RESERVOIR_SEEDS[reservoir_index % len(RESERVOIR_SEEDS)] + 991 * (reservoir_index + 1)
        candidate_params = _generate_param_candidates(
            base_params,
            interpolation_points=interpolation_points,
            sample_budget=sample_budget,
            rng_seed=candidate_seed,
        )
        if not candidate_params:
            tuned_params.append(base_params)
            tuned_errors.append(float('inf'))
            continue

        responsibility_vector = responsibilities[:, reservoir_index].astype(np.float32)
        responsibilities_future = client.scatter(responsibility_vector, broadcast=True)

        futures = []
        for candidate_idx, candidate in enumerate(candidate_params):
            seed = RESERVOIR_SEEDS[reservoir_index % len(RESERVOIR_SEEDS)] + (candidate_idx + 1) * 17
            future = client.submit(
                _evaluate_reservoir_candidate,
                reservoir_index,
                candidate,
                windows_future,
                responsibilities_future,
                lam,
                horizon,
                seed,
                pure=False,
            )
            futures.append(future)

        print(f"Expert {reservoir_index}: evaluating {len(candidate_params)} candidates on Dask")
        candidate_errors = client.gather(futures)
        if not candidate_errors:
            tuned_params.append(base_params)
            tuned_errors.append(float('inf'))
            continue

        best_index = int(np.argmin(np.array(candidate_errors, dtype=float)))
        tuned_params.append(candidate_params[best_index])
        tuned_errors.append(float(candidate_errors[best_index]))

        client.cancel(futures, force=True)
        client.cancel(responsibilities_future)

    client.cancel(windows_future)

    return tuned_params, tuned_errors
# -----------------------------------------------------------------------------
# Model dimensions (shared across experts)
# -----------------------------------------------------------------------------
K = 1   # number of input units
N = 500 # number of reservoir units
L = 1   # number of output units

# -----------------------------------------------------------------------------
# Expert parameter sets (three experts). Keep as provided but fixed.
# -----------------------------------------------------------------------------
# Reservoir parameter seeds and defaults (three experts)
RESERVOIR_SEEDS = [12345, 22345, 32345]

RESERVOIR_PARAM_DEFAULTS: List[ReservoirParams] = [
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
    ),
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
    ),
    ReservoirParams(
        spectral_radius=0.9,
        C=0.60,
        decay_rate=0.85,
        w_scale=0.30,
        w_sparsity=0.990,
        w_back_scale=0.56,
        w_in_scale=0.08,
        w_in_sparsity=0.60,
    ),
]


def _instantiate_reservoirs(param_list: List[ReservoirParams]) -> List[Reservoir]:
    instantiated: List[Reservoir] = []
    for expert_index, params in enumerate(param_list):
        seed = RESERVOIR_SEEDS[expert_index % len(RESERVOIR_SEEDS)] + expert_index * 1000
        expert_rng = np.random.default_rng(seed)
        instantiated.append(Reservoir(N, K, L, params, expert_rng))
    return instantiated


def reset_reservoir_bank() -> None:
    global reservoir_param_configs, reservoirs
    reservoir_param_configs = [ReservoirParams(**vars(p)) for p in RESERVOIR_PARAM_DEFAULTS]
    reservoirs = _instantiate_reservoirs(reservoir_param_configs)


def start_dask_client(run_dir: Path):
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError as exc:
        raise RuntimeError("Dask is required for hyperparameter tuning") from exc

    client = None
    cluster = None

    if os.environ.get('SLURM_JOB_ID'):
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError as exc:
            raise RuntimeError("dask_jobqueue is required when running under SLURM") from exc

        log_dir = run_dir / 'dask_logs'
        tmp_dir = run_dir / 'dask_tmp'
        log_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        processes = max(1, DASK_PROCESSES_PER_JOB)
        threads_per_process = max(1, DASK_WORKER_CORES // processes)

        job_script_prologue: List[str] = [
            f'export OMP_NUM_THREADS={threads_per_process}',
            f'export OPENBLAS_NUM_THREADS={threads_per_process}',
            f'export MKL_NUM_THREADS={threads_per_process}',
            f'export VECLIB_MAXIMUM_THREADS={threads_per_process}',
            f'export NUMEXPR_NUM_THREADS={threads_per_process}',
        ]
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            job_script_prologue.insert(0, f"source {venv_path}/bin/activate")

        job_extra = ['--requeue'] if DASK_REQUEUE else []

        cluster = SLURMCluster(
            queue=DASK_PARTITION,
            account=DASK_ACCOUNT,
            processes=processes,
            cores=DASK_WORKER_CORES,
            memory=DASK_WORKER_MEM,
            walltime=DASK_WORKER_WALLTIME,
            local_directory=str(tmp_dir),
            log_directory=str(log_dir),
            job_script_prologue=job_script_prologue,
            job_extra_directives=job_extra,
        )
        target_jobs = max(1, DASK_JOBS)
        cluster.scale(jobs=target_jobs)
        client = Client(cluster)
        try:
            expected_workers = target_jobs * DASK_PROCESSES_PER_JOB
            client.wait_for_workers(n_workers=expected_workers, timeout=600)
        except Exception:
            cluster.close()
            raise
    else:
        # Local fallback for development and debugging
        threads_per_process = max(1, DASK_WORKER_CORES // max(1, DASK_PROCESSES_PER_JOB))
        total_workers = max(1, min(DASK_JOBS * DASK_PROCESSES_PER_JOB, os.cpu_count() or 1))
        cluster = LocalCluster(n_workers=total_workers, threads_per_worker=threads_per_process)
        client = Client(cluster)

    return client, cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative mixture-of-experts ESN trainer")
    parser.add_argument('--tag', type=str, required=True, help='run identifier for logs and artifacts')
    parser.add_argument('--outdir', type=str, default='runs', help='output directory root (default: runs)')
    parser.add_argument('--iterations', type=int, default=NUM_TRAINING_ITERATIONS,
                        help='number of EM+tuning iterations to run (default: %(default)s)')
    return parser.parse_args()


def ensure_run_directory(base_dir: Path, tag: str) -> Path:
    run_dir = base_dir / f"{tag}-{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Dict) -> None:
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


reset_reservoir_bank()

def prepare_em_readouts_only(lam: float) -> Tuple[List[np.ndarray], List[List[Dict[str, np.ndarray]]]]:
    """
    Precompute teacher-forced design matrices and initialize readouts equally.
    Returns ``(W_out_list, designs)``.
    """
    state_target_pairs = build_state_target_pairs(reservoirs, train_windows)
    W_out_list = init_readouts_unweighted(state_target_pairs, lam)
    return W_out_list, state_target_pairs


def run_one_em_round(W_out_list: List[np.ndarray],
                     designs: List[List[Dict[str, np.ndarray]]],
                     tau: float = 0.8,
                     lam: float = 1e-3,
                     horizon: int = 10,
                     prior: Optional[np.ndarray] = None,
                     ema_prev: Optional[np.ndarray] = None,
                     alpha: float = 0.2):
    """Run one EM round on TRAIN_WINDOWS and return updated weights, responsibilities, and errors."""
    return em_round_readouts_only(reservoirs, train_windows, designs, W_out_list, lam=lam, tau=tau, prior=prior, ema_prev=ema_prev, alpha=alpha, horizon=horizon)

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


def run_training_loop(iterations: int,
                      lam: float,
                      tau: float,
                      horizon: int,
                      run_dir: Path) -> None:
    import time

    global train_windows, test_windows, reservoirs, reservoir_param_configs

    splits = _build_fixed_windows(N_WINDOWS_PER_REGIME, rng)
    train_windows = splits['train']
    test_windows = splits['test']

    reset_reservoir_bank()

    client, cluster = start_dask_client(run_dir)
    log_path = run_dir / RUN_LOG_FILENAME

    responsibilities_prev: Optional[np.ndarray] = None

    try:
        with log_path.open('w') as log_file:
            for iteration_index in range(iterations):
                iteration_label = f"Iter {iteration_index + 1}/{iterations}"
                print(f"\n=== {iteration_label} ===")

                t0 = time.time()
                W_out_list, state_target_pairs = prepare_em_readouts_only(lam=lam)
                t1 = time.time()
                print(f"[{iteration_label}] Prepared designs in {t1 - t0:.2f}s. Windows: {len(train_windows)} | Experts: {len(reservoirs)} | reservoir size={N}")
                print(f"[{iteration_label}] W_out shapes: {[w.shape for w in W_out_list]}")

                print(f"[{iteration_label}] Running EM (readouts only)...")
                t2 = time.time()
                W_out_list, responsibilities, errors = run_one_em_round(
                    W_out_list,
                    state_target_pairs,
                    tau=tau,
                    lam=lam,
                    horizon=horizon,
                    ema_prev=responsibilities_prev,
                    alpha=0.2,
                )
                t3 = time.time()
                print(f"[{iteration_label}] EM round done in {t3 - t2:.2f}s")

                e_mean_per_expert = errors.mean(axis=0)
                r_mean_per_expert = responsibilities.mean(axis=0)
                print(f"[{iteration_label}] Mean NRMSE@{horizon} per expert: {e_mean_per_expert}")
                print(f"[{iteration_label}] Mean responsibility per expert: {r_mean_per_expert}")

                top1 = np.argmax(responsibilities, axis=1)
                hist = np.bincount(top1, minlength=len(reservoirs))
                print(f"[{iteration_label}] Top-1 assignment counts: {hist.tolist()}")
                print(f"[{iteration_label}] Sample responsibilities (first 5 windows):\n", responsibilities[:5])

                regime_means, regime_counts = mean_responsibility_by_regime(train_windows, responsibilities)
                print(f"[{iteration_label}] Mean responsibilities per expert by regime:")
                header = "Regime".ljust(12) + " " + " ".join(f"Expert{idx}".rjust(10) for idx in range(len(reservoirs)))
                print(header)
                for regime_id, mean_values in sorted(regime_means.items(), key=lambda item: item[0]):
                    try:
                        regime_name = Regime(regime_id).name
                    except ValueError:
                        regime_name = str(regime_id)
                    row = regime_name.ljust(12)
                    for value in mean_values:
                        row += " " + f"{value:10.4f}"
                    row += f"  (n={regime_counts[regime_id]})"
                    print(row)

                errors_after_refit = compute_errors_matrix(reservoirs, train_windows, W_out_list, horizon=horizon)
                print(f"[{iteration_label}] Mean NRMSE@{horizon} per expert after refit: {errors_after_refit.mean(axis=0)}")

                tuned_params, tuned_errors = em_round_hyperparam_tuning(
                    reservoirs,
                    train_windows,
                    responsibilities,
                    lam=lam,
                    horizon=horizon,
                    interpolation_points=HYPERPARAM_INTERPOLATION_POINTS,
                    candidate_samples=HYPERPARAM_SAMPLE_BUDGET,
                    client=client,
                )
                reservoir_param_configs = [ReservoirParams(**vars(p)) for p in tuned_params]
                reservoirs = _instantiate_reservoirs(reservoir_param_configs)

                print(f"[{iteration_label}] Hyperparameter tuning results (responsibility-weighted NRMSE):")
                for expert_index, (params, error_value) in enumerate(zip(reservoir_param_configs, tuned_errors)):
                    param_summary = {field: round(getattr(params, field), 4) for field in vars(params)}
                    print(f"  Expert {expert_index}: error={error_value:.4f} params={param_summary}")

                iteration_record = {
                    'timestamp': timestamp(),
                    'iteration': iteration_index + 1,
                    'mean_nrmse': e_mean_per_expert.tolist(),
                    'mean_responsibility': r_mean_per_expert.tolist(),
                    'top1_counts': hist.tolist(),
                    'regime_mean_responsibility': serialize_regime_means(regime_means),
                    'regime_counts': serialize_regime_counts(regime_counts),
                    'post_refit_mean_nrmse': errors_after_refit.mean(axis=0).tolist(),
                    'tuned_errors': [float(err) for err in tuned_errors],
                    'tuned_params': [
                        {field: float(getattr(params, field)) for field in vars(params)}
                        for params in reservoir_param_configs
                    ],
                }
                log_file.write(json.dumps(iteration_record) + '\n')
                log_file.flush()

                responsibilities_prev = responsibilities
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    args = parse_args()

    base_outdir = Path(args.outdir).expanduser().resolve()
    base_outdir.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_run_directory(base_outdir, args.tag)

    config_snapshot = {
        'tag': args.tag,
        'outdir': str(base_outdir),
        'run_dir': str(run_dir),
        'iterations': int(args.iterations),
        'lam': 1e-3,
        'tau': 0.8,
        'horizon': 10,
        'dask': {
            'jobs': DASK_JOBS,
            'worker_cores': DASK_WORKER_CORES,
            'worker_mem': DASK_WORKER_MEM,
            'worker_walltime': DASK_WORKER_WALLTIME,
            'partition': DASK_PARTITION,
            'account': DASK_ACCOUNT,
            'processes_per_job': DASK_PROCESSES_PER_JOB,
        },
    }
    write_json(run_dir / 'config.json', config_snapshot)

    run_training_loop(
        iterations=int(args.iterations),
        lam=1e-3,
        tau=0.8,
        horizon=10,
        run_dir=run_dir,
    )
