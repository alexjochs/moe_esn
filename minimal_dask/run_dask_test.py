#!/usr/bin/env python
"""Small Dask-on-Slurm smoke test.

Launches a SLURMCluster via dask-jobqueue, requests a handful of workers, runs
an extremely small workload, and reports basic scheduler/worker information.
"""
from __future__ import annotations

import socket
from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def main() -> None:
    workdir = Path.cwd()
    venv_path = workdir / ".venv"
    if not venv_path.exists():
        raise RuntimeError(
            f"Expected virtualenv at {venv_path}; run setup_venv.sh before sbatch."
        )

    queue = "share"
    account = "eecs"
    jobs = 2
    cores_per_job = 2
    mem_per_job = "4GB"
    walltime = "00:15:00"
    processes = 1
    expected_workers = jobs * processes
    wait_timeout = 300
    local_dir = Path("/tmp")
    log_dir = workdir / "dask_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== Dask Slurm smoke test starting ===")
    print(f"Submitting jobs with account={account} queue={queue}")
    print(f"jobs={jobs} cores_per_job={cores_per_job} mem_per_job={mem_per_job}")

    cluster = SLURMCluster(
        queue=queue,
        account=account,
        processes=processes,
        cores=cores_per_job,
        memory=mem_per_job,
        walltime=walltime,
        python=str(venv_path / "bin" / "python"),
        local_directory=str(local_dir),
        log_directory=str(log_dir),
        job_script_prologue=[f"source {venv_path}/bin/activate"],
    )

    try:
        cluster.scale(jobs=jobs)
        print(f"Cluster scheduler: {cluster.scheduler_address}")

        with Client(cluster, timeout=wait_timeout) as client:
            client.wait_for_workers(expected_workers)
            print("Connected to cluster; scheduler info snippet:")
            info = client.scheduler_info()
            workers = info.get("workers", {})
            for idx, (addr, details) in enumerate(workers.items(), start=1):
                host = details.get("host", "?")
                nthreads = details.get("nthreads", "?")
                memory = details.get("memory_limit", "?")
                print(f"  worker {idx}: {addr} host={host} threads={nthreads} mem={memory}")

            def identify(node: int) -> str:
                return f"hello from {socket.gethostname()} handling item {node}"

            futures = client.map(identify, range(4))
            results = client.gather(futures)
            print("Results:")
            for line in results:
                print(f"  {line}")
    finally:
        print("Closing cluster...")
        cluster.close()
        print("Done.")


if __name__ == "__main__":
    main()
