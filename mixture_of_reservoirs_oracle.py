"""Mixture-of-reservoirs grid-search trainer with oracle routing."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import Regime
from reservoir import Reservoir, ReservoirParams

from moe.data_windows import _build_fixed_windows, _gen_regime_window
from moe.dask_utils import start_dask_client
from moe.gating import (
    compute_responsibilities_with_regularization,
    mean_responsibility_by_regime,
    serialize_regime_counts,
    serialize_regime_means,
)
from moe.hparam_search import em_round_hyperparam_tuning
from moe.logging import timestamp, write_json
from moe.readout_em import (
    compute_errors_matrix,
    prepare_em_readouts_only,
    refit_readouts_weighted,
)
from moe.reservoir_bank import (
    RESERVOIR_SEEDS,
    reset_reservoir_bank,
    _instantiate_reservoirs,
)


rng = np.random.default_rng(42)

# -----------------------------------------------------------------------------
# Dataset windowing
#   Window spec: [warmup | teacher-forced fit | eval (free-run)]
# -----------------------------------------------------------------------------
WARMUP_LEN = 200                # warmup length
TEACHER_FORCED_LEN = 200            # teacher-forced fit span; set to 0 to disable
MAX_EVAL_HORIZON = 50               # extra tail so free-run rollouts have targets
WINDOW_LEN_TOTAL = WARMUP_LEN + TEACHER_FORCED_LEN + MAX_EVAL_HORIZON
N_WINDOWS_PER_REGIME = 200

FREE_RUN_DIAGNOSTIC_HORIZON = 500
DIAGNOSTIC_ITERATIONS = {10, 20, 30}
DIAGNOSTIC_DIRNAME = "free_run_diagnostics"

NUM_TRAINING_ITERATIONS = 50
HYPERPARAM_INTERPOLATION_POINTS = 10
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
ESN_DASK_TASK_RETIRES = int(os.environ.get("ESN_DASK_TASK_RETIRES", 25))
DASK_ALLOWED_FAILURES = int(os.environ.get("ESN_DASK_ALLOWED_FAILURES", 100))

RUN_LOG_FILENAME = "iteration_log.jsonl"

# -----------------------------------------------------------------------------
# Gating regularization & annealing (epsilon smoothing + load-balancing bias)
# -----------------------------------------------------------------------------
# Linear annealing from *_START to *_END over [ANNEAL_START_ITER, ANNEAL_END_ITER]
ANNEAL_START_ITER = 1
ANNEAL_END_ITER = 15

TAU_START = 2.0
TAU_END = 0.7

EPSILON_START = 0.10   # uniform smoothing mass on responsibilities
EPSILON_END = 0.00

LAMBDA_LOAD_START = 1.0  # strength of load-balancing bias term
LAMBDA_LOAD_END = 0.0


def _linear_anneal(start: float, end: float, it: int, it0: int, it1: int) -> float:
    """Linearly anneal from start to end as it runs from it0..it1 (clamped)."""
    if it1 <= it0:
        return end
    if it <= it0:
        return start
    if it >= it1:
        return end
    t = (it - it0) / float(it1 - it0)
    return (1.0 - t) * start + t * end


# -----------------------------------------------------------------------------
# Model dimensions (shared across experts)
# -----------------------------------------------------------------------------
K = 1   # number of input units
N = 100 # number of reservoir units
L = 1   # number of output units

REGIME_TO_ORACLE_EXPERT: Dict[int, int] = {
    int(Regime.MACKEY_GLASS): 0,
    int(Regime.ROSSLER): 1,
    int(Regime.LORENZ): 2,
}


def _build_oracle_responsibilities(windows: List[Dict], num_experts: int) -> np.ndarray:
    """Return a one-hot responsibility matrix enforcing the oracle routing."""
    responsibilities = np.zeros((len(windows), num_experts), dtype=np.float32)
    for index, window in enumerate(windows):
        regime_id = int(window['regime'])
        expert_index = REGIME_TO_ORACLE_EXPERT.get(regime_id)
        if expert_index is None:
            raise ValueError(f"No oracle expert mapped for regime id {regime_id}")
        if expert_index >= num_experts:
            raise ValueError(
                f"Oracle expert index {expert_index} exceeds available experts ({num_experts})"
            )
        responsibilities[index, expert_index] = 1.0
    return responsibilities


def _build_diagnostic_windows(horizon: int) -> Dict[int, Dict]:
    """Construct deterministic windows for long-horizon diagnostics by regime."""
    total_length = WARMUP_LEN + TEACHER_FORCED_LEN + horizon
    diag_rng = np.random.default_rng(314159)
    windows: Dict[int, Dict] = {}
    for regime_id in (int(Regime.MACKEY_GLASS), int(Regime.ROSSLER), int(Regime.LORENZ)):
        y = _gen_regime_window(regime_id, total_length, diag_rng)
        windows[regime_id] = {
            'y': y,
            'idx_warmup_end': WARMUP_LEN,
            'idx_fit_end': WARMUP_LEN + TEACHER_FORCED_LEN,
            'idx_eval_end': total_length,
            'regime': regime_id,
            'id': f'diagnostic_{regime_id}',
        }
    return windows


def _plot_free_run_diagnostics(iteration: int,
                               run_dir: Path,
                               reservoirs: List[Reservoir],
                               readouts: List[np.ndarray],
                               windows_by_regime: Dict[int, Dict],
                               horizon: int) -> None:
    """Save per-expert free-run rollouts across regimes without rendering to screen."""
    diagnostic_dir = (run_dir / DIAGNOSTIC_DIRNAME)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    regime_sequence = [
        (int(Regime.MACKEY_GLASS), "Mackey-Glass"),
        (int(Regime.ROSSLER), "Rossler"),
        (int(Regime.LORENZ), "Lorenz"),
    ]
    time_axis = np.arange(horizon)

    for expert_index, (reservoir, w_out) in enumerate(zip(reservoirs, readouts)):
        fig, axes = plt.subplots(1, len(regime_sequence), figsize=(18, 5), sharey=True)
        axes = np.atleast_1d(axes)

        for axis, (regime_id, regime_label) in zip(axes, regime_sequence):
            window = windows_by_regime[regime_id]
            y_hat, y_true = reservoir.free_run(window, w_out, horizon=horizon)
            y_true_flat = y_true.reshape(-1)
            y_hat_flat = y_hat.reshape(-1)
            axis.plot(time_axis, y_true_flat, label="target", linewidth=1.0)
            axis.plot(time_axis, y_hat_flat, label="free-run", linewidth=1.0)
            axis.set_title(regime_label)
            axis.set_xlabel("step")
            if axis is axes[0]:
                axis.set_ylabel("value")

        axes[0].legend(loc="upper right")
        fig.suptitle(f"Iteration {iteration}: Expert {expert_index} free-run diagnostics")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = diagnostic_dir / f"iter_{iteration:02d}_expert_{expert_index}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


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


def run_training_loop(iterations: int,
                      lam: float,
                      horizon: int,
                      run_dir: Path) -> None:
    import time

    train_windows = _build_fixed_windows(
        N_WINDOWS_PER_REGIME,
        rng,
        WARMUP_LEN,
        TEACHER_FORCED_LEN,
        WINDOW_LEN_TOTAL,
    )
    reservoir_param_configs, reservoirs = reset_reservoir_bank(N, K, L)
    diagnostic_windows = _build_diagnostic_windows(FREE_RUN_DIAGNOSTIC_HORIZON)

    client, cluster = start_dask_client(
        run_dir,
        DASK_JOBS,
        DASK_WORKER_CORES,
        DASK_WORKER_MEM,
        DASK_WORKER_WALLTIME,
        DASK_PARTITION,
        DASK_ACCOUNT,
        DASK_PROCESSES_PER_JOB,
        DASK_REQUEUE,
    )
    log_path = run_dir / RUN_LOG_FILENAME

    responsibilities_prev: Optional[np.ndarray] = None

    try:
        with log_path.open('w') as log_file:
            for iteration_index in range(iterations):
                iteration_label = f"Iter {iteration_index + 1}/{iterations}"
                print(f"\n=== {iteration_label} ===")

                iter_num = iteration_index + 1
                tau_cur = _linear_anneal(TAU_START, TAU_END, iter_num, ANNEAL_START_ITER, ANNEAL_END_ITER)
                eps_cur = _linear_anneal(EPSILON_START, EPSILON_END, iter_num, ANNEAL_START_ITER, ANNEAL_END_ITER)
                lambda_load_cur = _linear_anneal(LAMBDA_LOAD_START, LAMBDA_LOAD_END, iter_num, ANNEAL_START_ITER, ANNEAL_END_ITER)

                t0 = time.time()
                W_out_list, state_target_pairs = prepare_em_readouts_only(reservoirs, train_windows, lam=lam)
                t1 = time.time()
                print(f"[{iteration_label}] Prepared designs in {t1 - t0:.2f}s. Windows: {len(train_windows)} | Experts: {len(reservoirs)} | reservoir size={N}")
                print(f"[{iteration_label}] W_out shapes: {[w.shape for w in W_out_list]}")

                print(f"[{iteration_label}] Evaluating reservoirs before oracle refit...")
                print(f"[{iteration_label}] Gating knobs: tau={tau_cur:.3f} | eps_uniform={eps_cur:.3f} | lambda_load={lambda_load_cur:.3f}")
                t2 = time.time()
                errors = compute_errors_matrix(reservoirs, train_windows, W_out_list, horizon=horizon)
                responsibilities_eval = compute_responsibilities_with_regularization(
                    errors,
                    tau=tau_cur,
                    eps_uniform=eps_cur,
                    lambda_load=lambda_load_cur,
                    previous_responsibilities=responsibilities_prev,
                    alpha=0.2,
                )
                t3 = time.time()
                print(f"[{iteration_label}] Computed evaluation responsibilities in {t3 - t2:.2f}s")

                oracle_responsibilities = _build_oracle_responsibilities(train_windows, len(reservoirs))
                t4 = time.time()
                W_out_list = refit_readouts_weighted(
                    state_target_pairs,
                    oracle_responsibilities,
                    lam,
                    W_out_list,
                )
                t5 = time.time()
                print(f"[{iteration_label}] Oracle-weighted readout refit done in {t5 - t4:.2f}s")

                e_mean_per_expert = errors.mean(axis=0)
                r_mean_per_expert = responsibilities_eval.mean(axis=0)
                print(f"[{iteration_label}] Mean NRMSE@{horizon} per expert (pre-refit): {e_mean_per_expert}")
                print(f"[{iteration_label}] Mean responsibility per expert (evaluation): {r_mean_per_expert}")

                top1 = np.argmax(responsibilities_eval, axis=1)
                hist = np.bincount(top1, minlength=len(reservoirs))
                print(f"[{iteration_label}] Top-1 assignment counts (evaluation): {hist.tolist()}")
                print(f"[{iteration_label}] Sample evaluation responsibilities (first 5 windows):\n", responsibilities_eval[:5])

                oracle_top1 = np.argmax(oracle_responsibilities, axis=1)
                oracle_hist = np.bincount(oracle_top1, minlength=len(reservoirs))
                print(f"[{iteration_label}] Oracle assignment counts: {oracle_hist.tolist()}")

                regime_means, regime_counts = mean_responsibility_by_regime(train_windows, responsibilities_eval)
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
                print(f"[{iteration_label}] Mean NRMSE@{horizon} per expert after oracle refit: {errors_after_refit.mean(axis=0)}")

                if iter_num in DIAGNOSTIC_ITERATIONS:
                    _plot_free_run_diagnostics(
                        iter_num,
                        run_dir,
                        reservoirs,
                        W_out_list,
                        diagnostic_windows,
                        FREE_RUN_DIAGNOSTIC_HORIZON,
                    )

                tuned_params, tuned_errors = em_round_hyperparam_tuning(
                    reservoirs,
                    train_windows,
                    oracle_responsibilities,
                    lam=lam,
                    horizon=horizon,
                    interpolation_points=HYPERPARAM_INTERPOLATION_POINTS,
                    candidate_samples=HYPERPARAM_SAMPLE_BUDGET,
                    client=client,
                    reservoir_seeds=RESERVOIR_SEEDS,
                    N=N,
                    K=K,
                    L=L,
                    task_retries=ESN_DASK_TASK_RETIRES,
                    dask_jobs=DASK_JOBS,
                    dask_processes_per_job=DASK_PROCESSES_PER_JOB,
                )
                reservoir_param_configs = [ReservoirParams(**vars(p)) for p in tuned_params]
                reservoirs = _instantiate_reservoirs(reservoir_param_configs, N, K, L)

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
                    'oracle_assignment_counts': oracle_hist.tolist(),
                    'regime_mean_responsibility': serialize_regime_means(regime_means),
                    'regime_counts': serialize_regime_counts(regime_counts),
                    'post_refit_mean_nrmse': errors_after_refit.mean(axis=0).tolist(),
                    'tuned_errors': [float(err) for err in tuned_errors],
                    'tuned_params': [
                        {field: float(getattr(params, field)) for field in vars(params)}
                        for params in reservoir_param_configs
                    ],
                    'tau_cur': float(tau_cur),
                    'eps_uniform': float(eps_cur),
                    'lambda_load': float(lambda_load_cur),
                }
                log_file.write(json.dumps(iteration_record) + '\n')
                log_file.flush()

                responsibilities_prev = responsibilities_eval
    finally:
        client.close()
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
        'router': {
            'type': 'oracle_fixed_regime',
            'mapping': {
                Regime.MACKEY_GLASS.name: REGIME_TO_ORACLE_EXPERT[int(Regime.MACKEY_GLASS)],
                Regime.ROSSLER.name: REGIME_TO_ORACLE_EXPERT[int(Regime.ROSSLER)],
                Regime.LORENZ.name: REGIME_TO_ORACLE_EXPERT[int(Regime.LORENZ)],
            },
        },
        'tau': {'start': TAU_START, 'end': TAU_END, 'anneal_iters': [ANNEAL_START_ITER, ANNEAL_END_ITER]},
        'epsilon_uniform': {'start': EPSILON_START, 'end': EPSILON_END},
        'lambda_load': {'start': LAMBDA_LOAD_START, 'end': LAMBDA_LOAD_END},
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
        horizon=10,
        run_dir=run_dir,
    )
