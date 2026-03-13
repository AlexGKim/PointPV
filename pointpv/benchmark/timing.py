"""
Timing utilities for benchmarking MLF vs. McDonald RG likelihood.

Records wall-clock time per likelihood evaluation at varying N and
produces scaling plots.

Inputs
------
method : callable
    Function that accepts (u, C) and returns logL (float).
N_values : list[int]
    Galaxy counts to benchmark.
n_repeats : int
    Number of repeat evaluations per N (for stable timing).

Outputs
-------
dict
    'N': array of N values
    'mean_time': mean wall-clock time per evaluation [s]
    'std_time': standard deviation [s]
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def time_evaluations(
    method: Callable[[np.ndarray, np.ndarray], float],
    u_list: list[np.ndarray],
    C_list: list[np.ndarray],
    n_repeats: int = 3,
) -> dict[str, np.ndarray]:
    """
    Time a likelihood method at multiple N values.

    Parameters
    ----------
    method :
        Callable(u, C) → logL.
    u_list :
        List of velocity vectors, one per N.
    C_list :
        List of covariance matrices, one per N.
    n_repeats :
        Number of timing repeats per N.

    Returns
    -------
    dict with 'N', 'mean_time', 'std_time', 'logL'.
    """
    N_values = [len(u) for u in u_list]
    mean_times = []
    std_times = []
    logL_values = []

    for u, C in zip(u_list, C_list):
        times = []
        last_logL = 0.0
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            last_logL = method(u, C)
            times.append(time.perf_counter() - t0)
        mean_times.append(np.mean(times))
        std_times.append(np.std(times))
        logL_values.append(last_logL)

    return {
        "N": np.array(N_values),
        "mean_time": np.array(mean_times),
        "std_time": np.array(std_times),
        "logL": np.array(logL_values),
    }


def print_timing_table(results: dict[str, np.ndarray], label: str = "") -> None:
    """Print a formatted timing table."""
    if label:
        print(f"\n=== {label} ===")
    print(f"{'N':>8}  {'mean_time (s)':>14}  {'std_time (s)':>12}  {'logL':>14}")
    print("-" * 55)
    for i, N in enumerate(results["N"]):
        print(
            f"{N:>8d}  "
            f"{results['mean_time'][i]:>14.4f}  "
            f"{results['std_time'][i]:>12.4f}  "
            f"{results['logL'][i]:>14.4f}"
        )


def plot_scaling(
    results_dict: dict[str, dict[str, np.ndarray]],
    output_path: str | None = None,
) -> None:
    """
    Plot wall-clock time vs. N for multiple methods.

    Parameters
    ----------
    results_dict :
        Dict mapping method label → timing results dict (from time_evaluations).
    output_path :
        If given, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (label, res), color in zip(results_dict.items(), colors):
        N = res["N"]
        ax.errorbar(
            N, res["mean_time"], yerr=res["std_time"],
            label=label, color=color, marker="o", capsize=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (galaxies)")
    ax.set_ylabel("Wall time per evaluation (s)")
    ax.set_title("MLF vs. McDonald RG: Scaling with N")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
