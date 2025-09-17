"""
Creates a single resevoir, instituting an example with a fixed silicon budget,
constrained by the number of total RRAM devices. This serves as a purely theoretical model,
using the leaky-integrate method proposed by jaeger for continous time series work.

Refactor: baseline logic moved into functions; added optional GA optimizer (`--ga`).
"""

import argparse
import concurrent.futures
import math
import os
import sys
from types import SimpleNamespace as _NS
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

from dataset import Regime, generate_dataset
from ga_single_reservoir import default_genome, run_ga
from single_reservoir_core import (
    build_reservoir,
    cycle_starts,
    evaluate,
    free_run_from_state,
    nrmse,
    regime_start_indices_in_range,
    segment_end,
    teacher_forced_states,
    train_readout,
)

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
C = 0.05
DECAY_RATE = 1.0

# Dimensions
K = 1  # number of input "units"
N = 1500  # number of network units (fixed per user request)
L = 1  # number of output units

GA_CONFIG = _NS(N=N, L=L, H=H, N_DISCARD=N_DISCARD, C=C, DECAY_RATE=DECAY_RATE)

def run_baseline(args):
    rng = np.random.default_rng(42)
    genome = default_genome(C, DECAY_RATE)
    reservoir = build_reservoir(
        rng,
        N=N,
        K=0,
        L=L,
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

    Y_teach, Y_labels = generate_dataset(total_points=H)
    Y_teach = Y_teach.reshape(1, H)

    states = teacher_forced_states(reservoir, Y_teach)
    W_out, train_end = train_readout(states, Y_teach, Y_labels, alpha=1e-6, discard=N_DISCARD)

    horizons = (5, 10, 50)
    probe_offsets = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400)
    fitness, aux = evaluate(
        reservoir,
        states,
        Y_teach,
        Y_labels,
        train_end,
        horizons=horizons,
        probe_offsets=probe_offsets,
        W_out=W_out,
    )

    cycle_idx = cycle_starts(Y_labels)
    n_cycles = len(cycle_idx)
    n_train_cycles = max(1, int(math.floor(0.7 * n_cycles)))
    train_end_cycle = cycle_idx[n_train_cycles] if n_train_cycles < n_cycles else int(0.7 * H)
    test_start = train_end_cycle
    test_end = H

    print("Baseline fitness:", fitness)
    print("Probes computed:", aux.get('probes', 0))
    _report_mem()

    if getattr(args, 'no_plots', False):
        return

    starts = regime_start_indices_in_range(Y_labels, test_start, test_end)
    first_start_by_regime: Dict[int, int] = {}
    for s, regime_id in starts:
        first_start_by_regime.setdefault(int(regime_id), int(s))

    regime_names = {
        int(Regime.MACKEY_GLASS): "Mackey-Glass",
        int(Regime.LORENZ): "Lorenz",
        int(Regime.ROSSLER): "Rössler",
    }

    plot_records = []
    for s, regime_id in starts:
        seg_end = segment_end(Y_labels, s)
        for delta in probe_offsets:
            t = s + delta
            if t >= seg_end - 1:
                continue
            for horizon in horizons:
                if t + horizon >= seg_end:
                    continue
                x_t = states[:, t].copy()
                y_pred, _ = free_run_from_state(reservoir, x_t, W_out, horizon)
                y_true = Y_teach[0, t + 1:t + 1 + horizon]
                err = nrmse(y_pred, y_true, 1.0)
                plot_records.append({
                    't': int(t),
                    'delta': int(delta),
                    'h': int(horizon),
                    'regime': int(regime_id),
                    'nrmse': float(err),
                })

    for regime_id, start_idx in first_start_by_regime.items():
        plt.figure()
        plotted_any = False
        for horizon in horizons:
            xs: List[int] = []
            ys: List[float] = []
            for rec in plot_records:
                if rec['regime'] != regime_id or rec['h'] != horizon:
                    continue
                if (rec['t'] - rec['delta']) == start_idx:
                    xs.append(rec['delta'])
                    ys.append(rec['nrmse'])
            if not xs:
                continue
            order = np.argsort(np.array(xs))
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.plot(xs, ys, label=f"h={horizon}")
            plotted_any = True

        plt.xlabel("Probe offset Δ steps since regime start")
        plt.ylabel("NRMSE")
        plt.title(f"{regime_names.get(regime_id, f'Regime {regime_id}')} : NRMSE vs Δ by horizon")
        if plotted_any:
            plt.legend(title="Horizon")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

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
        run_ga(args, GA_CONFIG)
    else:
        run_baseline(args)
