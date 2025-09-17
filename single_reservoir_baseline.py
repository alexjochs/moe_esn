"""
Creates a single resevoir, instituting an example with a fixed silicon budget,
constrained by the number of total RRAM devices. This serves as a purely theoretical model,
using the leaky-integrate method proposed by jaeger for continous time series work.

Refactor: baseline logic moved into functions; added optional GA optimizer (`--ga`).
"""

import numpy as np
import os 
import sys
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from sklearn.linear_model import Ridge
import math
import argparse
import concurrent.futures
from types import SimpleNamespace as _NS
from enum import IntEnum
from typing import Tuple, Dict, List
from pathlib import Path
import json
from datetime import datetime
import psutil

from dataset import generate_dataset

# =================== IO helpers for HPC runs ===================

def _genome_to_dict(g):
    return {
        'C': float(g.C),
        'DECAY_RATE': float(g.DECAY_RATE),
        'target_rho': float(g.target_rho),
        'p_nonzero': float(g.p_nonzero),
        'w_scale': float(g.w_scale),
        'p_in': float(g.p_in),
        'win_scale': float(g.win_scale),
        'wback_scale': float(g.wback_scale),
        'log10_alpha': float(g.log10_alpha),
    }


def _timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def _resolve_outdir(base: str|None, tag: str|None):
    # prefer user provided base; else use ./runs
    base_dir = Path(base) if base else Path('runs')
    # use SLURM info if present to avoid collisions on arrays
    slurm_id = os.environ.get('SLURM_JOB_ID')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    parts = [tag] if tag else []
    if slurm_id:
        parts.append(f"slurm{slurm_id}")
    if slurm_array_id:
        parts.append(f"arr{slurm_array_id}")
    parts.append(_timestamp())
    outdir = base_dir.joinpath(*[p for p in parts if p])
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _write_json(path: Path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

# =================== Memory reporting helper ===================
def _report_mem():
    """Print system memory and current process RSS using psutil."""
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    print(f"[MEM] System total: {vm.total/1e9:.2f} GB, available: {vm.available/1e9:.2f} GB; "
          f"Process RSS: {rss/1e9:.2f} GB")

# =================== Constants (kept from baseline) ===================
# dataset Constants
N_DISCARD = 100  # discard first 100 states
N_TRAIN = 11_000  # number of steps to train on
N_TEST = 4_000    # number of steps to test on
N_WARMUP = 1_000  # free-run steps before scoring during test
H = 15_000        # max timestep

# Neuron Dynamics Constants
C = 0.44
DECAY_RATE = 0.9

# Dimensions
K = 1  # number of input "units"
N = 1500  # number of network units (fixed per user request)
L = 1  # number of output units

# ===== Regime labels =====
class Regime(IntEnum):
    MACKEY_GLASS = 0
    LORENZ = 1
    ROSSLER = 2
    TRANSITION = 3

# =================== Core math helpers (kept) ===================

def _scale_by_spectral_norm(W_mat, target_rho=0.9, n_iter=50, rng=None):
    """
    Safely scale W so its spectral norm (upper bound on spectral radius) ≈ target_rho.
    Uses power iteration on W^T W to estimate the largest singular value.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    v = rng.normal(size=W_mat.shape[1])
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        v = W_mat.T @ (W_mat @ v)
        v_norm = np.linalg.norm(v) + 1e-12
        v = v / v_norm
    sigma = math.sqrt(float(v @ (W_mat.T @ (W_mat @ v))))
    if sigma > 0:
        return W_mat * (target_rho / sigma), sigma
    return W_mat, 0.0

# =================== Baseline construction ===================

def _make_W_in(rng, N, K, p_in=0.5, win_scale=0.14):
    # match baseline support {0, ±win_scale} with probability p_in/2 each for signs
    choices = [0.0, -win_scale, win_scale]
    p0 = max(0.0, 1.0 - p_in)
    p = [p0, (1.0 - p0) / 2.0, (1.0 - p0) / 2.0]
    return rng.choice(choices, size=(N, K), p=p)


def _make_W(rng, N, p_nonzero=0.0125, w_scale=0.4, target_rho=0.9):
    # sparse signed with {0, ±w_scale}
    p0 = max(0.0, 1.0 - p_nonzero)
    W = rng.choice([0.0, w_scale, -w_scale], size=(N, N), p=[p0, (1.0 - p0) / 2.0, (1.0 - p0) / 2.0])
    W, _ = _scale_by_spectral_norm(W, target_rho=target_rho, n_iter=30, rng=rng)
    return W


def _make_W_back(rng, N, L, wback_scale=0.56):
    return rng.uniform(-wback_scale, wback_scale, size=(N, L))


def _teacher_forced_states(W_in, W, W_back, Y_teach, C_val, DECAY_val):
    X_n = np.zeros((N, H))
    # teacher-forced state evolution across the full sequence
    for n in range(0, H - 1):
        preact = W @ X_n[:, n] + W_back @ (Y_teach[:, n])
        X_n[:, n + 1] = (1 - C_val * DECAY_val) * X_n[:, n] + C_val * np.tanh(preact)
    return X_n


def _cycle_starts(labels):
    starts = []
    for i in range(H):
        if labels[i] == Regime.MACKEY_GLASS and (i == 0 or labels[i-1] == Regime.TRANSITION):
            starts.append(i)
    return starts


def _regime_start_indices_in_range(labels, start, end):
    out = []
    for i in range(start, end):
        if labels[i] == Regime.TRANSITION:
            continue
        if i == start:
            if i == 0 or labels[i-1] == Regime.TRANSITION:
                out.append((i, int(labels[i])))
        else:
            if labels[i-1] == Regime.TRANSITION:
                out.append((i, int(labels[i])))
    return out


def _segment_end(labels, s):
    i = s
    while i < H and labels[i] != Regime.TRANSITION:
        i += 1
    return i


def _free_run_from_state(x0, h, W, W_back, W_out, C_val, DECAY_val):
    x = x0.copy()
    y_seq = np.zeros(h)
    for t in range(h):
        z_hat = (W_out @ x).reshape(-1)
        y_hat = np.tanh(z_hat)
        y_seq[t] = float(y_hat[0])
        preact = W @ x + W_back @ y_hat
        x = (1 - C_val * DECAY_val) * x + C_val * np.tanh(preact)
    return y_seq, x


def _nrmse(yhat, ytrue, denom):
    err = yhat - ytrue
    rmse = math.sqrt(float(np.mean(err * err)))
    return rmse / (denom + 1e-12)

# =================== Readout and evaluation ===================

def _train_readout(X_n, Y_teach, Y_labels, alpha):
    idx = np.arange(H)
    # train/test split by cycles
    _cycle_idx = _cycle_starts(Y_labels)
    _n_cycles = len(_cycle_idx)
    if _n_cycles == 0:
        raise RuntimeError("No cycles detected; check labels.")
    _n_train_cycles = max(1, int(np.floor(0.7 * _n_cycles)))
    TRAIN_END = _cycle_idx[_n_train_cycles] if _n_train_cycles < _n_cycles else int(0.7 * H)

    train_mask = (idx >= N_DISCARD) & (idx < TRAIN_END) & (Y_labels != Regime.TRANSITION)
    train_idx = idx[train_mask]

    X_train = X_n[:, train_idx].T
    Y_train = Y_teach[:, train_idx].T  # shape (n_train, 1)
    z_train = np.arctanh(np.clip(Y_train, -0.999, 0.999))

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_train, z_train)
    W_out = ridge.coef_

    return W_out, TRAIN_END


def _evaluate(W, W_back, W_out, X_n, Y_teach, Y_labels, TRAIN_END,
              HORIZONS=(5, 10, 50), PROBE_OFFSETS=(0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400),
              C_val=C, DECAY_val=DECAY_RATE):
    # per-regime normalization using TRAIN portion only
    regime_std: Dict[int, float] = {}
    for R in [Regime.MACKEY_GLASS, Regime.LORENZ, Regime.ROSSLER]:
        mask = (np.arange(H) < TRAIN_END) & (Y_labels == R)
        vals = Y_teach[0, mask]
        regime_std[int(R)] = float(np.std(vals)) if np.any(mask) else 1.0

    # evaluation probes
    idx = np.arange(H)
    TEST_START = TRAIN_END
    TEST_END = H
    eval_results: List[Dict] = []

    starts = _regime_start_indices_in_range(Y_labels, TEST_START, TEST_END)
    for s, R in starts:
        e = _segment_end(Y_labels, s)
        for d in PROBE_OFFSETS:
            t = s + d
            if t >= e - 1:
                continue
            for h in HORIZONS:
                if t + h >= e:
                    continue
                x_t = X_n[:, t].copy()
                y_pred, _ = _free_run_from_state(x_t, h, W, W_back, W_out, C_val, DECAY_val)
                y_true = Y_teach[0, t+1:t+1+h]
                nrmse = _nrmse(y_pred, y_true, regime_std[R])
                if not np.isfinite(nrmse):
                    nrmse = 1e6
                eval_results.append({'t': int(t), 'delta': int(d), 'h': int(h), 'regime': int(R), 'nrmse': float(nrmse)})

    # aggregate fitness
    if len(eval_results) == 0:
        return 1e6, {'probes': 0}

    # median per (regime, horizon)
    per_rh: Dict[Tuple[int, int], float] = {}
    for r in eval_results:
        key = (r['regime'], r['h'])
        per_rh.setdefault(key, []).append(r['nrmse'])
    for key, vals in per_rh.items():
        per_rh[key] = float(np.median(np.array(vals)))
    
    fitness = float(np.mean(list(per_rh.values()))) if per_rh else 1e6

    return fitness, {'per_rh': per_rh, 'probes': len(eval_results)}

# =================== Genome and GA ===================

def _default_genome():
    # Seed around current baseline values
    return _NS(
        C=C,
        DECAY_RATE=DECAY_RATE,
        target_rho=0.9,
        p_nonzero=0.0125,   # baseline implied ~1.25%
        w_scale=0.4,
        p_in=0.5,
        win_scale=0.14,
        wback_scale=0.56,
        log10_alpha=math.log10(1e-6),
    )


def _clip_and_repair(g):
    # bounds
    g.C = float(np.clip(g.C, 0.05, 1.0))
    g.DECAY_RATE = float(np.clip(g.DECAY_RATE, 0.1, 1.0))
    # keep effective leak bounded
    eff = g.C * g.DECAY_RATE
    eff = float(np.clip(eff, 0.02, 0.9))
    g.DECAY_RATE = eff / g.C

    g.target_rho = float(np.clip(g.target_rho, 0.5, 1.1))
    g.p_nonzero = float(np.clip(g.p_nonzero, 1e-4, 5e-3))
    g.w_scale = float(np.clip(g.w_scale, 0.05, 1.0))
    g.p_in = float(np.clip(g.p_in, 0.1, 0.9))
    g.win_scale = float(np.clip(g.win_scale, 0.01, 0.5))
    g.wback_scale = float(np.clip(g.wback_scale, 0.05, 1.5))
    g.log10_alpha = float(np.clip(g.log10_alpha, -10.0, -1.0))
    return g


def _mutate(g, rng):
    gg = _NS(**g.__dict__)
    # Gaussian in real space
    gg.C += rng.normal(0, 0.1)
    gg.DECAY_RATE += rng.normal(0, 0.1)
    gg.target_rho += rng.normal(0, 0.05)
    gg.p_nonzero += rng.normal(0, 5e-4)
    gg.w_scale += rng.normal(0, 0.05)
    gg.p_in += rng.normal(0, 0.05)
    gg.win_scale += rng.normal(0, 0.02)
    gg.wback_scale += rng.normal(0, 0.1)
    # log space for alpha
    gg.log10_alpha += rng.normal(0, 0.5)
    return _clip_and_repair(gg)


def _crossover(a, b, rng):
    child = {}
    for k in a.__dict__.keys():
        if rng.random() < 0.5:
            child[k] = a.__dict__[k]
        else:
            child[k] = b.__dict__[k]
    return _clip_and_repair(_NS(**child))




def _eval_genome(genome, base_seed=42):
    # local RNG for reproducibility
    rng = np.random.default_rng(base_seed)

    # data
    Y_teach, Y_labels = generate_dataset(total_points=H)
    Y_teach = Y_teach.reshape(1, H)

    # weights
    W_in = _make_W_in(rng, N, K, p_in=genome.p_in, win_scale=genome.win_scale)
    W = _make_W(rng, N, p_nonzero=genome.p_nonzero, w_scale=genome.w_scale, target_rho=genome.target_rho)
    W_back = _make_W_back(rng, N, L, wback_scale=genome.wback_scale)

    # states
    X_n = _teacher_forced_states(W_in, W, W_back, Y_teach, genome.C, genome.DECAY_RATE)

    # readout
    alpha = 10 ** genome.log10_alpha
    W_out, TRAIN_END = _train_readout(X_n, Y_teach, Y_labels, alpha)

    # eval
    HORIZONS = (5, 10, 50)
    fitness, aux = _evaluate(W, W_back, W_out, X_n, Y_teach, Y_labels, TRAIN_END,
                             HORIZONS=HORIZONS, C_val=genome.C, DECAY_val=genome.DECAY_RATE)
    return fitness, aux


def _eval_genome_multi_seed(genome, seeds):
    """Evaluate one genome across multiple seeds and return its mean fitness."""
    fs = []
    for seed in seeds:
        f, _aux = _eval_genome(genome, base_seed=seed)
        fs.append(float(f))
    return float(np.mean(np.array(fs, dtype=float)))


def _start_dask_cluster(args, outdir: Path):
    """Spin up a Dask SLURMCluster for GA evaluation."""
    try:
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
    except ImportError as exc:
        raise RuntimeError(
            "Dask GA mode requires dask[distributed] and dask-jobqueue; install them in the venv."
        ) from exc

    jobs = max(1, int(getattr(args, 'jobs', 1)))
    queue = getattr(args, 'dask_partition', 'share')
    account = getattr(args, 'dask_account', 'eecs')
    worker_cores = max(1, int(getattr(args, 'dask_worker_cores', 10)))
    worker_mem = getattr(args, 'dask_worker_mem', '16GB')
    worker_walltime = getattr(args, 'dask_worker_walltime', '01:00:00')
    preempt_requeue = bool(getattr(args, 'dask_preempt_requeue', False))
    processes_arg = getattr(args, 'dask_processes_per_worker', None)
    pop_size = max(1, int(getattr(args, 'pop', 20)))
    if processes_arg is None:
        per_job_target = math.ceil(pop_size / jobs) if jobs else pop_size
        processes = max(1, min(worker_cores, per_job_target))
    else:
        processes = max(1, int(processes_arg))

    if processes > worker_cores:
        print(
            f"[Dask] Requested processes/job ({processes}) exceeds cores/job ({worker_cores}); clamping to cores."
        )
        processes = worker_cores

    timeout = float(getattr(args, 'dask_timeout', 600))

    log_dir = outdir / 'dask_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path(os.environ.get('TMPDIR', '/tmp'))

    job_script_prologue = [
        'export OMP_NUM_THREADS=1',
        'export OPENBLAS_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'export VECLIB_MAXIMUM_THREADS=1',
        'export NUMEXPR_NUM_THREADS=1',
    ]
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        job_script_prologue.insert(0, f"source {venv_path}/bin/activate")

    print(
        f"[Dask] Starting cluster: jobs={jobs} cores/job={worker_cores} processes/job={processes} mem/job={worker_mem} "
        f"queue={queue} account={account}"
    )

    job_extra_directives = []
    if preempt_requeue:
        job_extra_directives.append('--requeue')

    cluster = SLURMCluster(
        queue=queue,
        account=account,
        processes=processes,
        cores=worker_cores,
        memory=worker_mem,
        walltime=worker_walltime,
        python=sys.executable,
        local_directory=str(local_dir),
        log_directory=str(log_dir),
        job_script_prologue=job_script_prologue,
        job_extra_directives=job_extra_directives,
        threads_per_worker=1,
    )

    desired_jobs = jobs
    try:
        cluster.scale(jobs=desired_jobs)
        client = Client(cluster, timeout=timeout)
        expected = jobs * processes
        client.wait_for_workers(n_workers=expected, timeout=timeout)
    except Exception:
        cluster.close()
        raise

    info = client.scheduler_info()
    address = info.get('address', 'unknown')
    print(f"[Dask] Scheduler reachable at {address}; expected workers: {expected}")
    return cluster, client, expected, desired_jobs


def _evaluate_chunk(genomes, seeds, worker_cores):
    """Sequentially evaluate a slice of genomes within a single Dask worker."""
    if not genomes:
        return []

    return [_eval_genome_multi_seed(genome, seeds) for genome in genomes]


def _dask_evaluate_population(client, population, seeds, chunk_size, worker_cores):
    """Evaluate genomes in Dask-managed chunks; each worker runs tasks sequentially."""
    if not population:
        return []

    chunk_size = max(1, int(chunk_size))
    futures = []
    slices: List[Tuple[int, int]] = []
    for start in range(0, len(population), chunk_size):
        end = min(start + chunk_size, len(population))
        chunk = population[start:end]
        fut = client.submit(
            _evaluate_chunk,
            chunk,
            seeds,
            worker_cores,
            pure=False,
        )
        futures.append(fut)
        slices.append((start, end))

    gathered = client.gather(futures)
    mean_fitness = [1e6] * len(population)
    for (start, end), chunk_vals in zip(slices, gathered):
        if len(chunk_vals) != (end - start):
            raise RuntimeError("Unexpected chunk result length from Dask worker")
        mean_fitness[start:end] = [float(v) for v in chunk_vals]
    return mean_fitness


def run_ga(args):
    print("running GA")
    pop = int(getattr(args, 'pop', 20))
    gens = int(getattr(args, 'gens', 2))
    jobs = int(getattr(args, 'jobs', 1))
    # allow seeds via CLI like --seeds 42 1337 2025
    seeds = [int(s) for s in getattr(args, 'seeds', [42, 1337, 2025])]

    # output directory setup
    outdir = _resolve_outdir(getattr(args, 'outdir', None), getattr(args, 'tag', None))
    (outdir / 'artifacts').mkdir(exist_ok=True)

    # save a run config snapshot
    cfg = {
        'pop': pop,
        'gens': gens,
        'jobs': jobs,
        'seeds': seeds,
        'N': N,
        'H': H,
        'timestamp': _timestamp(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    _write_json(outdir / 'config.json', cfg)

    num_workers = os.cpu_count() if jobs > 1 else 1
    rng = np.random.default_rng(42)

    # init population around baseline with jitter
    population = []
    base = _default_genome()
    for i in range(pop):
        g = _mutate(base, rng) if i > 0 else _clip_and_repair(base)
        population.append(g)

    best = None
    history = []

    # open a TSV log for per-generation summary
    log_path = outdir / 'ga_log.tsv'
    with open(log_path, 'w') as logf:
        logf.write('gen\tbest_fitness\n')

    dask_cluster = None
    dask_client = None
    expected_workers = None
    desired_jobs = None
    try:
        if getattr(args, 'dask', False):
            dask_cluster, dask_client, expected_workers, desired_jobs = _start_dask_cluster(args, outdir)

        for gen in range(gens):
            if dask_client is not None:
                n_workers = None
                try:
                    info = dask_client.scheduler_info()
                    n_workers = len(info.get('workers', {}))
                    if expected_workers is not None and n_workers < expected_workers:
                        missing = expected_workers - n_workers
                        print(
                            f"[Dask] Detected {missing} missing workers (have {n_workers}/{expected_workers}); rescaling"
                        )
                        if desired_jobs is not None:
                            dask_cluster.scale(jobs=desired_jobs)
                        dask_client.wait_for_workers(
                            n_workers=expected_workers,
                            timeout=float(getattr(args, 'dask_timeout', 600)),
                        )
                except Exception as exc:
                    print(f"[Dask] Warning: could not refresh worker pool: {exc}")
                chunk_cfg = getattr(args, 'dask_chunk_size', None)
                if chunk_cfg is None:
                    total_procs = n_workers if (n_workers and n_workers > 0) else expected_workers
                    if not total_procs:
                        total_procs = 1
                    # Aim for ~2× more chunks than worker processes to keep scheduler fed.
                    denom = max(1, 2 * total_procs)
                    chunk_size = max(1, len(population) // denom)
                else:
                    chunk_size = max(1, int(chunk_cfg))

                fitnesses = _dask_evaluate_population(
                    dask_client,
                    population,
                    seeds,
                    chunk_size,
                    max(1, int(getattr(args, 'dask_worker_cores', 10))),
                )
            else:
                fitnesses = []
                if jobs > 1:
                    # Parallel evaluation via local processes
                    with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, num_workers)) as executor:
                        jobs_list = []
                        for g in population:
                            jobs_list.append([executor.submit(_eval_genome, g, base_seed=s) for s in seeds])
                        for genome_jobs in jobs_list:
                            fs = []
                            for job in genome_jobs:
                                f, _ = job.result()
                                fs.append(f)
                            fitnesses.append(float(np.mean(fs)))
                else:
                    # Sequential evaluation
                    for g in population:
                        fs = []
                        for s in seeds:
                            f, _ = _eval_genome(g, base_seed=s)
                            fs.append(f)
                        fitnesses.append(float(np.mean(fs)))

            # track best
            order = np.argsort(np.array(fitnesses))
            best_idx = int(order[0])
            best = (population[best_idx], float(fitnesses[best_idx]))
            history.append(best[1])

            # write per-gen summary
            with open(log_path, 'a') as logf:
                logf.write(f"{gen}\t{best[1]:.8f}\n")
            # also dump current best genome each gen for resiliency
            _write_json(outdir / 'best_genome.json', {'fitness': best[1], 'genome': _genome_to_dict(best[0])})

            print(f"gen {gen:03d} best_fitness {best[1]:.6f}")

            # selection: tournament size 3
            def _tournament():
                idxs = rng.choice(len(population), size=3, replace=False)
                best_local = min(idxs, key=lambda i: fitnesses[i])
                return population[best_local]

            new_pop = []
            # elitism: keep top 2
            new_pop.append(population[int(order[0])])
            if pop > 1:
                new_pop.append(population[int(order[1])])
            # fill rest
            while len(new_pop) < pop:
                a = _tournament()
                b = _tournament()
                child = _crossover(a, b, rng)
                child = _mutate(child, rng)
                new_pop.append(child)
            population = new_pop
    finally:
        if dask_client is not None:
            dask_client.close()
        if dask_cluster is not None:
            dask_cluster.close()

    # final report and saves
    g_best, f_best = best
    print("best genome:", g_best)
    print("best fitness:", f_best)
    _report_mem()

    _write_json(outdir / 'best_genome_final.json', {'fitness': f_best, 'genome': _genome_to_dict(g_best)})
    np.save(outdir / 'fitness_history.npy', np.array(history, dtype=float))

    return g_best, f_best, history

# =================== Baseline runner (unchanged results) ===================

def run_baseline(args):
    rng = np.random.default_rng(42)

    # Readout (initialized; will be overwritten after training)
    W_in = rng.choice([0, -0.14, 0.14], size=(N, K), p=[0.5, 0.25, 0.25])
    W = rng.choice([0.0, 0.4, -0.4], size=(N, N), p=[0.9875, 0.00625, 0.00625])
    W, _ = _scale_by_spectral_norm(W, target_rho=0.9, n_iter=30, rng=rng)
    W_out = np.zeros((L, N))
    W_back = rng.uniform(-0.56, 0.56, size=(N, L))

    Y_teach, Y_labels = generate_dataset(total_points=H)
    Y_teach = Y_teach.reshape(1, H)

    X_n = _teacher_forced_states(W_in, W, W_back, Y_teach, C, DECAY_RATE)
    W_out, TRAIN_END = _train_readout(X_n, Y_teach, Y_labels, alpha=1e-6)

    # Evaluation and plots (kept)
    fitness, aux = _evaluate(W, W_back, W_out, X_n, Y_teach, Y_labels, TRAIN_END)

    # Lightweight textual output to confirm execution shape
    _cycle_idx = _cycle_starts(Y_labels)
    _n_cycles = len(_cycle_idx)
    _n_train_cycles = max(1, int(np.floor(0.7 * _n_cycles)))
    TRAIN_END2 = _cycle_idx[_n_train_cycles] if _n_train_cycles < _n_cycles else int(0.7 * H)
    TEST_START = TRAIN_END2
    TEST_END = H

    print("Baseline fitness:", fitness)
    print("Probes computed:", aux.get('probes', 0))
    _report_mem()

    # ===== Plots: first occurrence in TEST per regime =====
    # Find first start index in TEST for each regime
    starts = _regime_start_indices_in_range(Y_labels, TEST_START, TEST_END)
    first_start_by_regime = {}
    for s, R in starts:
        if int(R) not in first_start_by_regime:
            first_start_by_regime[int(R)] = int(s)

    HORIZONS = (5, 10, 50)
    PROBE_OFFSETS = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400)

    regime_names = {
        int(Regime.MACKEY_GLASS): "Mackey-Glass",
        int(Regime.LORENZ): "Lorenz",
        int(Regime.ROSSLER): "Rössler",
    }

    eval_results = []
    # Recompute eval_results needed for plotting per regime start
    for s, R in starts:
        e = _segment_end(Y_labels, s)
        for d in PROBE_OFFSETS:
            t = s + d
            if t >= e - 1:
                continue
            for h in HORIZONS:
                if t + h >= e:
                    continue
                x_t = X_n[:, t].copy()
                y_pred, _ = _free_run_from_state(x_t, h, W, W_back, W_out, C, DECAY_RATE)
                y_true = Y_teach[0, t+1:t+1+h]
                nrmse = _nrmse(y_pred, y_true, 1.0)  # only for relative plotting here
                eval_results.append({'t': int(t), 'delta': int(d), 'h': int(h), 'regime': int(R), 'nrmse': float(nrmse)})

    for R, s in first_start_by_regime.items():
        plt.figure()
        plotted_any = False
        for h in HORIZONS:
            xs, ys = [], []
            for r in eval_results:
                if r['regime'] != R or r['h'] != h:
                    continue
                if (r['t'] - r['delta']) == s:
                    xs.append(r['delta'])
                    ys.append(r['nrmse'])
            if not xs:
                continue
            order = np.argsort(np.array(xs))
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.plot(xs, ys, label=f"h={h}")
            plotted_any = True

        plt.xlabel("Probe offset Δ steps since regime start")
        plt.ylabel("NRMSE")
        plt.title(f"{regime_names.get(R, f'Regime {R}')} : NRMSE vs Δ by horizon")
        if plotted_any:
            plt.legend(title="Horizon")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    if not getattr(args, 'no_plots', False):
        plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ga', action='store_true', help='run GA to tune hyperparameters')
    p.add_argument('--outdir', type=str, default=None, help='base output directory for artifacts; default ./runs')
    p.add_argument('--tag', type=str, default=None, help='optional run tag prefix for the output directory')
    p.add_argument('--jobs', type=int, default=1, help='max worker processes for GA evaluation')
    p.add_argument('--pop', type=int, default=20, help='GA population size')
    p.add_argument('--gens', type=int, default=2, help='GA number of generations')
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 1337, 2025], help='list of seeds to average fitness over')
    p.add_argument('--no-plots', action='store_true', help='suppress matplotlib windows (useful on headless/HPC)')
    p.add_argument('--dask', action='store_true', help='use Dask + Slurm for GA evaluation instead of local processes')
    p.add_argument('--dask-worker-cores', type=int, default=10, help='CPU cores per Dask worker job (<=10 to satisfy policy)')
    p.add_argument('--dask-worker-mem', type=str, default='16GB', help='Memory per Dask worker job (e.g., 16GB)')
    p.add_argument('--dask-worker-walltime', type=str, default='01:00:00', help='Walltime per Dask worker job (HH:MM:SS)')
    p.add_argument('--dask-account', type=str, default='eecs', help='Slurm account for Dask worker jobs')
    p.add_argument('--dask-partition', type=str, default='share', help='Slurm partition for Dask worker jobs')
    p.add_argument('--dask-processes-per-worker', type=int, default=None,
                   help='Dask processes per worker job; default equals --dask-worker-cores so nthreads=1')
    p.add_argument('--dask-timeout', type=float, default=600.0, help='Seconds to wait for Dask workers to start')
    p.add_argument('--dask-chunk-size', type=int, default=None,
                   help='How many genomes each Dask worker evaluates per task; default chooses ≈2×#processes chunks')
    p.add_argument('--dask-preempt-requeue', action='store_true', help='Request --requeue for worker jobs on preemptible partitions')
    args = p.parse_args()

    # If no display or user requests no plots, use non-interactive backend to avoid crashes on HPC
    if args.no_plots or os.environ.get('DISPLAY', '') == '':
        import matplotlib
        matplotlib.use('Agg')

    if args.ga:
        run_ga(args)
    else:
        run_baseline(args)
