#!/usr/bin/env python
"""
Hybrid RG+MLF benchmark: find optimal stop_size for RG coarse-graining.

RG compresses N points level-by-level.  Rather than always going to 1 node,
the hybrid strategy stops at N_stop nodes and hands off to MLF (Cholesky):

    total_logL = rg_coarsen_all(..., stop_size=N_stop)   # partial logL + C_stop, u_stop
                 + mlf.log_likelihood(u_stop, C_stop)

For each N, this script sweeps N_stop across powers of 2 from 1 up to N,
times T_RG(N→N_stop) + T_MLF(N_stop) for each value, and finds the minimum.
The crossover is hardware- and library-dependent (BLAS, cache).

Output
------
- Table: N_stop | T_RG (s) | T_MLF (s) | T_total (s) | speedup vs pure MLF
- Plot: T_total vs N_stop (log x-axis), pure-MLF and pure-RG reference lines

Usage
-----
    python scripts/benchmark_hybrid.py --sizes 500 1000
    python scripts/benchmark_hybrid.py --sizes 500 --schur-tol 1.0
    python scripts/benchmark_hybrid.py --sizes 200 500 1000 --n-repeats 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Problem construction (reused from benchmark_scaling.py)
# ---------------------------------------------------------------------------

def _make_problem(
    n: int,
    seed: int = 42,
    length_scale: float = 50.0,
    sigma_v: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic PV problem: exponential kernel covariance."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 500, size=(n, 3))
    r = cdist(pos, pos)
    C = sigma_v ** 2 * np.exp(-r / length_scale)
    C += np.eye(n) * (sigma_v * 0.05) ** 2
    u = rng.standard_normal(n) * sigma_v * 0.1
    return u, C, pos


# ---------------------------------------------------------------------------
# Core benchmark for one N
# ---------------------------------------------------------------------------

def benchmark_hybrid_n(
    n: int,
    n_repeats: int,
    schur_tol: float,
) -> dict:
    """
    Sweep stop_size for a problem of size N and return timing results.

    Returns
    -------
    dict with keys:
        n, t_tree,
        stop_sizes: list[int],
        t_rg: list[float],         # best T_RG per stop_size
        t_mlf_stop: list[float],   # best T_MLF(stop) per stop_size
        t_total: list[float],      # t_rg + t_mlf_stop
        logL_hybrid: list[float],  # total logL at each stop_size (accuracy check)
        t_mlf_full: float,         # pure MLF(N) best time
        logL_mlf: float,           # pure MLF logL (reference)
        t_rg_full: float,          # pure RG(N→1) best time
        logL_rg_full: float,       # pure RG logL (reference)
    """
    from pointpv.rg.tree import build_tree
    from pointpv.rg.coarsen import rg_coarsen_all
    from pointpv.likelihood.mlf import log_likelihood as mlf_logL

    print(f"  N={n}: constructing problem ...", flush=True)
    u, C, pos = _make_problem(n, seed=42 + n)

    print(f"  N={n}: building tree ...", flush=True)
    t0 = time.perf_counter()
    tree = build_tree(pos)
    t_tree = time.perf_counter() - t0
    print(f"  N={n}: tree built in {t_tree:.3f}s", flush=True)

    # --- Pure MLF reference ---
    print(f"  N={n}: timing pure MLF ({n_repeats} repeat(s)) ...", flush=True)
    times_mlf = []
    logL_mlf = float("nan")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        logL_mlf = mlf_logL(u, C)
        times_mlf.append(time.perf_counter() - t0)
    t_mlf_full = min(times_mlf)
    print(f"  N={n}: MLF  best={t_mlf_full:.4f}s  logL={logL_mlf:.4f}", flush=True)

    # --- Pure RG reference (stop_size=1) ---
    rg_kwargs: dict = {"schur_tol": schur_tol}
    label_rg = f"RG-schur={schur_tol}" if schur_tol > 0 else "RG-dense"
    print(f"  N={n}: timing pure {label_rg} ({n_repeats} repeat(s)) ...", flush=True)
    times_rg_full = []
    logL_rg_full = float("nan")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        logL_rg_full = float(rg_coarsen_all(u, C, tree, stop_size=1, **rg_kwargs))  # type: ignore[arg-type]
        times_rg_full.append(time.perf_counter() - t0)
    t_rg_full = min(times_rg_full)
    print(f"  N={n}: {label_rg}  best={t_rg_full:.4f}s  logL={logL_rg_full:.4f}", flush=True)

    # --- Sweep stop_size values (powers of 2 from 1 to N) ---
    # Always include 1 and N; powers of 2 in between
    stop_sizes: list[int] = []
    s = 1
    while s <= n:
        stop_sizes.append(s)
        s *= 2
    if stop_sizes[-1] < n:
        stop_sizes.append(n)

    t_rg_list: list[float] = []
    t_mlf_stop_list: list[float] = []
    t_total_list: list[float] = []
    logL_hybrid_list: list[float] = []

    for stop in stop_sizes:
        if stop == 1:
            # Same as pure RG — no MLF on top
            t_rg_list.append(t_rg_full)
            t_mlf_stop_list.append(0.0)
            t_total_list.append(t_rg_full)
            logL_hybrid_list.append(logL_rg_full)
            continue

        if stop >= n:
            # No RG at all — same as pure MLF
            t_rg_list.append(0.0)
            t_mlf_stop_list.append(t_mlf_full)
            t_total_list.append(t_mlf_full)
            logL_hybrid_list.append(logL_mlf)
            continue

        print(f"  N={n}: timing hybrid stop_size={stop} ({n_repeats} repeat(s)) ...",
              flush=True)
        times_rg_s: list[float] = []
        times_mlf_s: list[float] = []
        logL_h = float("nan")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = rg_coarsen_all(u, C, tree, stop_size=stop, **rg_kwargs)
            t_rg_s = time.perf_counter() - t0
            times_rg_s.append(t_rg_s)

            # result is (partial_logL, C_stop, u_stop) when stop_size > 1
            partial_logL, C_stop, u_stop = result  # type: ignore[misc]

            t0 = time.perf_counter()
            logL_h = partial_logL + mlf_logL(u_stop, C_stop)
            times_mlf_s.append(time.perf_counter() - t0)

        best_rg = min(times_rg_s)
        best_mlf = min(times_mlf_s)
        t_rg_list.append(best_rg)
        t_mlf_stop_list.append(best_mlf)
        t_total_list.append(best_rg + best_mlf)
        logL_hybrid_list.append(logL_h)
        print(
            f"  N={n}: stop={stop:5d}  T_RG={best_rg:.4f}s  "
            f"T_MLF={best_mlf:.4f}s  T_total={best_rg+best_mlf:.4f}s  "
            f"|ΔlogL|={abs(logL_h - logL_mlf):.2e}",
            flush=True,
        )

    return {
        "n": n,
        "t_tree": t_tree,
        "stop_sizes": stop_sizes,
        "t_rg": t_rg_list,
        "t_mlf_stop": t_mlf_stop_list,
        "t_total": t_total_list,
        "logL_hybrid": logL_hybrid_list,
        "t_mlf_full": t_mlf_full,
        "logL_mlf": logL_mlf,
        "t_rg_full": t_rg_full,
        "logL_rg_full": logL_rg_full,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(res: dict) -> None:
    """Print timing table for one N."""
    n = res["n"]
    t_mlf = res["t_mlf_full"]
    stop_sizes = res["stop_sizes"]
    t_total = res["t_total"]
    logL_hybrid = res["logL_hybrid"]
    logL_ref = res["logL_mlf"]

    opt_idx = int(np.argmin(t_total))
    opt_stop = stop_sizes[opt_idx]
    speedup = t_mlf / t_total[opt_idx] if t_total[opt_idx] > 0 else float("nan")

    print(f"\n=== N={n}: hybrid stop_size sweep ===")
    print(f"  Pure MLF:      {t_mlf:.4f}s  (logL={logL_ref:.4f})")
    print(f"  Pure RG→1:     {res['t_rg_full']:.4f}s  (|ΔlogL|={abs(res['logL_rg_full']-logL_ref):.2e})")
    print(f"  Optimal stop:  N_stop*={opt_stop}  T*={t_total[opt_idx]:.4f}s  speedup={speedup:.2f}x\n")

    print(f"  {'stop_size':>10}  {'T_RG (s)':>10}  {'T_MLF (s)':>10}  {'T_total (s)':>12}  {'speedup':>8}  {'|ΔlogL|':>10}")
    print("  " + "-" * 70)
    for i, stop in enumerate(stop_sizes):
        sp = t_mlf / t_total[i] if t_total[i] > 0 else float("nan")
        delta = abs(logL_hybrid[i] - logL_ref)
        marker = " <-- optimal" if i == opt_idx else ""
        print(
            f"  {stop:>10d}  {res['t_rg'][i]:>10.4f}  {res['t_mlf_stop'][i]:>10.4f}"
            f"  {t_total[i]:>12.4f}  {sp:>8.2f}x  {delta:>10.2e}{marker}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results: list[dict], output_path: str) -> None:
    """One panel per N: T_total vs stop_size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = len(all_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

    for ax, res in zip(axes[0], all_results):
        n = res["n"]
        stops = np.array(res["stop_sizes"])
        t_total = np.array(res["t_total"])
        t_mlf = res["t_mlf_full"]
        t_rg = res["t_rg_full"]

        opt_idx = int(np.argmin(t_total))

        ax.plot(stops, t_total, "o-", color="#2ca02c", lw=2, label="RG+MLF hybrid")
        ax.axhline(t_mlf, color="#1f77b4", lw=1.5, linestyle="--", label=f"Pure MLF ({t_mlf:.3f}s)")
        ax.axhline(t_rg,  color="#ff7f0e", lw=1.5, linestyle=":",  label=f"Pure RG→1 ({t_rg:.3f}s)")
        ax.axvline(stops[opt_idx], color="#2ca02c", lw=0.8, linestyle=":", alpha=0.6)
        ax.scatter([stops[opt_idx]], [t_total[opt_idx]], s=100, color="#2ca02c",
                   zorder=5, label=f"Optimal N_stop*={stops[opt_idx]}")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("stop_size (N_stop)", fontsize=11)
        ax.set_ylabel("Best total time (s)", fontsize=11)
        ax.set_title(f"N={n}: hybrid RG+MLF vs stop_size", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid RG+MLF benchmark: sweep stop_size to find optimal handoff point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[500],
        help="Catalog sizes to benchmark",
    )
    p.add_argument(
        "--schur-tol", type=float, default=0.0, metavar="TOL",
        help="schur_tol for RG (0=dense, >0 skips small rank-1 updates)",
    )
    p.add_argument(
        "--n-repeats", type=int, default=3,
        help="Timing repeats per (stop_size, method) combination",
    )
    p.add_argument(
        "--output-dir", default="figs",
        help="Directory for output figure",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mode = f"schur_tol={args.schur_tol}" if args.schur_tol > 0 else "dense"
    print("=== benchmark_hybrid.py ===")
    print(f"  sizes      = {args.sizes}")
    print(f"  RG mode    = {mode}")
    print(f"  n-repeats  = {args.n_repeats}")
    print(f"  output-dir = {args.output_dir}")

    all_results: list[dict] = []
    for n in args.sizes:
        print(f"\n--- N={n} ---")
        res = benchmark_hybrid_n(
            n,
            n_repeats=args.n_repeats,
            schur_tol=args.schur_tol,
        )
        all_results.append(res)
        print_table(res)

    tag = f"_schur{args.schur_tol}" if args.schur_tol > 0 else "_dense"
    out_png = os.path.join(args.output_dir, f"benchmark_hybrid{tag}.png")
    plot_results(all_results, out_png)


if __name__ == "__main__":
    main()
