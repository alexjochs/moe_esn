import argparse
import concurrent.futures
import json
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
DEFAULT_OFFSETS = (0, 50, 100, 300,)

GA_CONFIG = _NS(
    N=1500,
    L=1,
    H=15_000,
    N_DISCARD=100,
    C=0.05,
    DECAY_RATE=1.0,
)


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


def resolve_outdir(base: str, tag: str) -> Path:
    base_dir = Path(base)
    slurm_id = os.environ.get('SLURM_JOB_ID')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    parts: List[str] = [tag]
    if slurm_id:
        parts.append(f"slurm{slurm_id}")
    if slurm_array_id:
        parts.append(f"arr{slurm_array_id}")
    parts.append(timestamp())
    outdir = base_dir.joinpath(*parts)
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
        target_rho=0.6072229533959054,
        p_nonzero=0.0002780440231267247,
        w_scale=0.08948540567520748,
        p_in=0.43097525871856673,
        w_in_scale=0.15284045686064893,
        wback_scale=1.2666342453436854,
        bias_value=0.0,
        log10_alpha=-6.014296756825457,
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
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client

    jobs = int(args.jobs)
    queue = args.dask_partition
    account = args.dask_account
    worker_cores = int(args.dask_worker_cores)
    worker_mem = args.dask_worker_mem
    worker_walltime = args.dask_worker_walltime
    preempt_requeue = args.dask_preempt_requeue
    processes = int(args.dask_processes_per_worker)
    timeout = float(args.dask_timeout)

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
    pop = int(args.pop)
    gens = int(args.gens)
    jobs = int(args.jobs)
    seeds = [int(s) for s in args.seeds]

    outdir = resolve_outdir(args.outdir, args.tag)
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
        dask_cluster, dask_client, expected_workers, desired_jobs = start_dask_cluster(args, outdir)
        for gen in range(gens):
            try:
                info = dask_client.scheduler_info()
                n_workers = len(info.get('workers', {}))
                if expected_workers is not None and n_workers < expected_workers:
                    missing = expected_workers - n_workers
                    print(
                        f"[Dask] Detected {missing} missing workers (have {n_workers}/{expected_workers}); rescaling"
                    )
                    dask_cluster.scale(jobs=desired_jobs)
                    dask_client.wait_for_workers(
                        n_workers=expected_workers,
                        timeout=float(args.dask_timeout),
                    )
            except Exception as exc:
                print(f"[Dask] Warning: could not refresh worker pool: {exc}")

            total_procs = expected_workers if expected_workers else 1
            denom = max(1, 2 * total_procs)
            chunk_size = max(1, len(population) // denom)

            fitnesses = dask_evaluate_population(dask_client, population, seeds, chunk_size, config)

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genetic algorithm tuner for single ESN reservoir")
    parser.add_argument('--outdir', type=str, required=True,
                        help='base output directory for artifacts')
    parser.add_argument('--tag', type=str, required=True,
                        help='run tag prefix for output directory naming')
    parser.add_argument('--jobs', type=int, required=True,
                        help='maximum worker processes for GA evaluation')
    parser.add_argument('--pop', type=int, required=True, help='GA population size')
    parser.add_argument('--gens', type=int, required=True, help='GA generations to run')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='list of seeds used for averaging fitness')
    parser.add_argument('--dask-worker-cores', type=int, required=True,
                        help='CPU cores per Dask worker job')
    parser.add_argument('--dask-worker-mem', type=str, required=True,
                        help='memory per Dask worker job (e.g., 16GB)')
    parser.add_argument('--dask-worker-walltime', type=str, required=True,
                        help='walltime per Dask worker job')
    parser.add_argument('--dask-account', type=str, required=True,
                        help='Slurm account for Dask worker jobs')
    parser.add_argument('--dask-partition', type=str, required=True,
                        help='Slurm partition for Dask worker jobs')
    parser.add_argument('--dask-processes-per-worker', type=int, required=True,
                        help='Dask processes per worker job')
    parser.add_argument('--dask-timeout', type=float, required=True,
                        help='seconds to wait for Dask workers to start')
    parser.add_argument('--dask-preempt-requeue', action='store_true',
                        help='request --requeue for worker jobs on preemptible partitions')
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    run_ga(cli_args, GA_CONFIG)
