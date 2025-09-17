import concurrent.futures
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as _NS
from typing import Dict, List, Sequence, Tuple

import numpy as np
import psutil

from dataset import generate_dataset
from single_reservoir_core import build_reservoir, evaluate, teacher_forced_states, train_readout


DEFAULT_HORIZONS = (5, 10, 50)
DEFAULT_OFFSETS = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400)


def genome_to_dict(g: _NS) -> Dict[str, float]:
    return {
        'C': float(g.C),
        'DECAY_RATE': float(g.DECAY_RATE),
        'target_rho': float(g.target_rho),
        'p_nonzero': float(g.p_nonzero),
        'w_scale': float(g.w_scale),
        'p_in': float(g.p_in),
        'w_in_scale': float(g.w_in_scale),
        'wback_scale': float(g.wback_scale),
        'bias_value': float(g.bias_value),
        'log10_alpha': float(g.log10_alpha),
    }


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def resolve_outdir(base: str | None, tag: str | None) -> Path:
    base_dir = Path(base) if base else Path('runs')
    slurm_id = os.environ.get('SLURM_JOB_ID')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    parts: List[str] = [tag] if tag else []
    if slurm_id:
        parts.append(f"slurm{slurm_id}")
    if slurm_array_id:
        parts.append(f"arr{slurm_array_id}")
    parts.append(timestamp())
    outdir = base_dir.joinpath(*[p for p in parts if p])
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def write_json(path: Path, obj) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def report_mem() -> None:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    print(
        f"[MEM] System total: {vm.total / 1e9:.2f} GB, available: {vm.available / 1e9:.2f} GB; "
        f"Process RSS: {rss / 1e9:.2f} GB"
    )


def default_genome(C: float, decay_rate: float) -> _NS:
    return _NS(
        C=C,
        DECAY_RATE=decay_rate,
        target_rho=0.9,
        p_nonzero=0.0125,
        w_scale=0.4,
        p_in=0.5,
        w_in_scale=0.14,
        wback_scale=0.56,
        bias_value=0.0,
        log10_alpha=math.log10(1e-6),
    )


def clip_and_repair(g: _NS) -> _NS:
    g.C = float(np.clip(g.C, 0.001, 1.0))
    g.DECAY_RATE = float(np.clip(g.DECAY_RATE, 0.1, 1.0))
    eff = g.C * g.DECAY_RATE
    eff = float(np.clip(eff, 0.02, 0.9))
    g.DECAY_RATE = eff / g.C

    g.target_rho = float(np.clip(g.target_rho, 0.5, 1.1))
    g.p_nonzero = float(np.clip(g.p_nonzero, 1e-4, 5e-3))
    g.w_scale = float(np.clip(g.w_scale, 0.05, 1.0))
    g.p_in = float(np.clip(g.p_in, 0.1, 0.9))
    g.w_in_scale = float(np.clip(g.w_in_scale, 0.01, 0.5))
    g.wback_scale = float(np.clip(g.wback_scale, 0.05, 1.5))
    g.bias_value = float(np.clip(g.bias_value, -1.0, 1.0))
    g.log10_alpha = float(np.clip(g.log10_alpha, -10.0, -1.0))
    return g


def mutate(g: _NS, rng: np.random.Generator) -> _NS:
    gg = _NS(**g.__dict__)
    gg.C += rng.normal(0, 0.1)
    gg.DECAY_RATE += rng.normal(0, 0.1)
    gg.target_rho += rng.normal(0, 0.05)
    gg.p_nonzero += rng.normal(0, 5e-4)
    gg.w_scale += rng.normal(0, 0.05)
    gg.p_in += rng.normal(0, 0.05)
    gg.w_in_scale += rng.normal(0, 0.02)
    gg.wback_scale += rng.normal(0, 0.1)
    gg.bias_value += rng.normal(0, 0.05)
    gg.log10_alpha += rng.normal(0, 0.5)
    return clip_and_repair(gg)


def crossover(a: _NS, b: _NS, rng: np.random.Generator) -> _NS:
    child = {}
    for k in a.__dict__.keys():
        child[k] = a.__dict__[k] if rng.random() < 0.5 else b.__dict__[k]
    return clip_and_repair(_NS(**child))


def eval_genome(genome: _NS,
                config: _NS,
                base_seed: int = 42,
                horizons: Sequence[int] = DEFAULT_HORIZONS,
                offsets: Sequence[int] = DEFAULT_OFFSETS) -> Tuple[float, Dict]:
    rng = np.random.default_rng(base_seed)
    Y_teach, Y_labels = generate_dataset(total_points=config.H)
    Y_teach = Y_teach.reshape(1, config.H)

    reservoir = build_reservoir(
        rng,
        N=config.N,
        K=0,
        L=config.L,
        C=genome.C,
        decay_rate=genome.DECAY_RATE,
        spectral_radius=genome.target_rho,
        w_scale=genome.w_scale,
        wback_scale=genome.wback_scale,
        p_nonzero=genome.p_nonzero,
        p_in=genome.p_in,
        w_in_scale=genome.w_in_scale,
        bias_value=genome.bias_value,
    )

    states = teacher_forced_states(reservoir, Y_teach)
    alpha = 10 ** genome.log10_alpha
    W_out, train_end = train_readout(states, Y_teach, Y_labels, alpha, discard=config.N_DISCARD)

    fitness, aux = evaluate(
        reservoir,
        states,
        Y_teach,
        Y_labels,
        train_end,
        horizons=horizons,
        probe_offsets=offsets,
        W_out=W_out,
    )
    return fitness, aux


def eval_genome_multi_seed(genome: _NS, seeds: Sequence[int], config: _NS) -> float:
    vals = [eval_genome(genome, config, base_seed=s)[0] for s in seeds]
    return float(np.mean(np.array(vals, dtype=float)))


def start_dask_cluster(args, outdir: Path):
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
    print(f"[Dask] Scheduler reachable at {address}; expected workers: {jobs * processes}")
    return cluster, client, jobs * processes, desired_jobs


def evaluate_chunk(genomes: Sequence[_NS], seeds: Sequence[int], config: _NS) -> List[float]:
    if not genomes:
        return []
    return [eval_genome_multi_seed(genome, seeds, config) for genome in genomes]


def dask_evaluate_population(client,
                             population: Sequence[_NS],
                             seeds: Sequence[int],
                             chunk_size: int,
                             config: _NS) -> List[float]:
    if not population:
        return []
    chunk_size = max(1, int(chunk_size))
    futures = []
    slices: List[Tuple[int, int]] = []
    for start in range(0, len(population), chunk_size):
        end = min(start + chunk_size, len(population))
        chunk = population[start:end]
        fut = client.submit(evaluate_chunk, chunk, seeds, config, pure=False)
        futures.append(fut)
        slices.append((start, end))

    gathered = client.gather(futures)
    fitnesses = [1e6] * len(population)
    for (start, end), values in zip(slices, gathered):
        if len(values) != (end - start):
            raise RuntimeError("Unexpected chunk result length from Dask worker")
        fitnesses[start:end] = [float(v) for v in values]
    return fitnesses


def run_ga(args, config: _NS) -> Tuple[_NS, float, List[float]]:
    print("running GA")
    pop = int(getattr(args, 'pop', 20))
    gens = int(getattr(args, 'gens', 2))
    jobs = int(getattr(args, 'jobs', 1))
    seeds = [int(s) for s in getattr(args, 'seeds', [42, 1337, 2025])]

    outdir = resolve_outdir(getattr(args, 'outdir', None), getattr(args, 'tag', None))
    (outdir / 'artifacts').mkdir(exist_ok=True)

    cfg_snapshot = {
        'pop': pop,
        'gens': gens,
        'jobs': jobs,
        'seeds': seeds,
        'N': config.N,
        'H': config.H,
        'timestamp': timestamp(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    write_json(outdir / 'config.json', cfg_snapshot)

    num_workers = os.cpu_count() if jobs > 1 else 1
    rng = np.random.default_rng(42)

    population = []
    base = clip_and_repair(default_genome(config.C, config.DECAY_RATE))
    for i in range(pop):
        g = mutate(base, rng) if i > 0 else _NS(**base.__dict__)
        population.append(g)

    best: Tuple[_NS, float] | None = None
    history: List[float] = []

    log_path = outdir / 'ga_log.tsv'
    with open(log_path, 'w') as logf:
        logf.write('gen\tbest_fitness\n')

    dask_cluster = None
    dask_client = None
    expected_workers = None
    desired_jobs = None
    try:
        if getattr(args, 'dask', False):
            dask_cluster, dask_client, expected_workers, desired_jobs = start_dask_cluster(args, outdir)

        for gen in range(gens):
            if dask_client is not None:
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
                    total_procs = expected_workers if expected_workers else 1
                    denom = max(1, 2 * total_procs)
                    chunk_size = max(1, len(population) // denom)
                else:
                    chunk_size = max(1, int(chunk_cfg))

                fitnesses = dask_evaluate_population(dask_client, population, seeds, chunk_size, config)
            else:
                fitnesses: List[float] = []
                if jobs > 1:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, num_workers)) as executor:
                        jobs_list = []
                        for genome in population:
                            jobs_list.append([executor.submit(eval_genome, genome, config, base_seed=s) for s in seeds])
                        for genome_jobs in jobs_list:
                            fs = [job.result()[0] for job in genome_jobs]
                            fitnesses.append(float(np.mean(fs)))
                else:
                    for genome in population:
                        fs = [eval_genome(genome, config, base_seed=s)[0] for s in seeds]
                        fitnesses.append(float(np.mean(fs)))

            order = np.argsort(np.array(fitnesses))
            best_idx = int(order[0])
            best = (population[best_idx], float(fitnesses[best_idx]))
            history.append(best[1])

            with open(log_path, 'a') as logf:
                logf.write(f"{gen}\t{best[1]:.8f}\n")
            write_json(outdir / 'best_genome.json', {'fitness': best[1], 'genome': genome_to_dict(best[0])})

            print(f"gen {gen:03d} best_fitness {best[1]:.6f}")

            def tournament():
                idxs = rng.choice(len(population), size=3, replace=False)
                best_local = min(idxs, key=lambda i: fitnesses[i])
                return population[best_local]

            new_pop = []
            new_pop.append(population[int(order[0])])
            if pop > 1:
                new_pop.append(population[int(order[1])])
            while len(new_pop) < pop:
                a = tournament()
                b = tournament()
                child = mutate(crossover(a, b, rng), rng)
                new_pop.append(child)
            population = new_pop
    finally:
        if dask_client is not None:
            dask_client.close()
        if dask_cluster is not None:
            dask_cluster.close()

    if best is None:
        raise RuntimeError("GA completed without evaluating any genome")

    best_genome, best_fitness = best
    print("best genome:", best_genome)
    print("best fitness:", best_fitness)
    report_mem()

    write_json(outdir / 'best_genome_final.json', {'fitness': best_fitness, 'genome': genome_to_dict(best_genome)})
    np.save(outdir / 'fitness_history.npy', np.array(history, dtype=float))

    return best_genome, best_fitness, history

