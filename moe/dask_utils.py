import os
from pathlib import Path


def start_dask_client(run_dir: Path,
                      jobs: int,
                      worker_cores: int,
                      worker_mem: str,
                      worker_walltime: str,
                      partition: str,
                      account: str,
                      processes_per_job: int,
                      requeue: bool):
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

        processes = max(1, processes_per_job)
        threads_for_blas = max(1, worker_cores // processes)

        job_script_prologue = [
            f'export OMP_NUM_THREADS={threads_for_blas}',
            f'export OPENBLAS_NUM_THREADS={threads_for_blas}',
            f'export MKL_NUM_THREADS={threads_for_blas}',
            f'export VECLIB_MAXIMUM_THREADS={threads_for_blas}',
            f'export NUMEXPR_NUM_THREADS={threads_for_blas}',
        ]
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            job_script_prologue.insert(0, f"source {venv_path}/bin/activate")

        job_extra = ['--requeue'] if requeue else []

        cluster = SLURMCluster(
            queue=partition,
            account=account,
            processes=processes,
            cores=worker_cores,
            memory=worker_mem,
            walltime=worker_walltime,
            local_directory=str(tmp_dir),
            log_directory=str(log_dir),
            job_script_prologue=job_script_prologue,
            job_extra_directives=job_extra,
            worker_extra_args=["--resources", "reservoir_eval=1"],
        )
        target_jobs = max(1, jobs)
        cluster.scale(jobs=target_jobs)
        client = Client(cluster)
        try:
            expected_workers = max(1, target_jobs * processes_per_job)
            min_required = min(
                expected_workers,
                max(1, int(os.environ.get("ESN_DASK_MIN_CONNECTED_WORKERS", 10))),
            )
            client.wait_for_workers(n_workers=min_required, timeout=600)
            connected = len(client.scheduler_info().get("workers", {}))
            if connected < min_required:
                raise RuntimeError(
                    f"Connected {connected} workers, fewer than required minimum {min_required}."
                )
            if connected < expected_workers:
                print(
                    f"[Dask] Connected {connected}/{expected_workers} workers; remaining workers will attach as resources become available."
                )
        except Exception:
            cluster.close()
            raise
    else:
        total_workers = max(1, min(jobs * processes_per_job, os.cpu_count() or 1))
        cluster = LocalCluster(
            n_workers=total_workers,
            threads_per_worker=max(1, worker_cores // max(1, processes_per_job)),
            resources={"reservoir_eval": 1},
        )
        client = Client(cluster)

    return client, cluster


__all__ = ["start_dask_client"]
