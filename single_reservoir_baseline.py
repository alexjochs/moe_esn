"""
Creates a single reservoir baseline experiment using a fixed silicon budget and
the leaky-integrate method proposed by Jaeger for continuous time series work.

The genetic algorithm entry point now lives in `ga_single_reservoir.py`. This
module focuses solely on running and optionally plotting the baseline setup.

best genome after 120 generations, population 60:
namespace(C=0.046624243040634034, DECAY_RATE=1.0, target_rho=0.5737850795875341, 
p_nonzero=0.00047618637769709894, w_scale=0.6284040028678406, p_in=0.1807399789443335,
w_in_scale=0.18772886222512142, wback_scale=1.5, bias_value=0.2252049194491637,
log10_alpha=-5.419977986835365)
"""

import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import psutil

from dataset import Regime, generate_dataset
from ga_single_reservoir import default_genome
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

def run_baseline():
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
    run_baseline()
