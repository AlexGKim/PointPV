#!/usr/bin/env python
"""
Runtime scaling benchmark: MLF (Cholesky) vs. McDonald RG likelihood.

Measures wall-clock time per likelihood evaluation at multiple catalog sizes,
compares log-likelihood values between methods, and produces a two-panel
scaling + accuracy figure.

Three baseline methods are always timed:
  MLF       — direct Cholesky, O(N^3)
  RG-dense  — McDonald RG, dense path (schur_tol=0), O(N^3)
  RG-schur  — McDonald RG, Schur approximation only (schur_tol=1, sparse_tol=0)

Additionally, RG-sparse variants are timed at each requested sparse_tol value
(all use schur_tol=1.0). The sparse_tol threshold (in (km/s)^2) controls how
aggressively off-diagonal entries are zeroed after each coarsening level; higher
values produce sparser matrices and faster computation at the cost of accuracy.

Usage
-----
    python scripts/benchmark_scaling.py
    python scripts/benchmark_scaling.py --sizes 100 500 1000 2000
    python scripts/benchmark_scaling.py --sizes 100 1000 10000 --n-repeats 1
    python scripts/benchmark_scaling.py --sparse-tols 0.1 1 10 100 1000
    python scripts/benchmark_scaling.py --skip-mlf-large 5000

Notes
-----
* For N >= 2000, set --n-repeats 1 to keep runtime manageable.
* MLF at N=10000 requires ~800 MB RAM and can take 30-120 s.
  Use --skip-mlf-large N to omit MLF for N >= that threshold.
* For a full fsigma8-grid accuracy check at one N, see validate_fsigma8.py.
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
# Problem construction
# ---------------------------------------------------------------------------

def _make_problem(n: int, seed: int = 42, length_scale: float = 50.0,
                  sigma_v: float = 300.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic PV problem with N galaxies.

    Uses scipy.spatial.distance.cdist to avoid the (N, N, 3) intermediate
    array that numpy broadcasting would require (~2.4 GB at N=10000).
    """
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 500, size=(n, 3))
    r = cdist(pos, pos)
    C = sigma_v ** 2 * np.exp(-r / length_scale)
    C += np.eye(n) * (sigma_v * 0.05) ** 2
    u = rng.standard_normal(n) * sigma_v * 0.1
    return u, C, pos


# ---------------------------------------------------------------------------
# Method list builder
# ---------------------------------------------------------------------------

def _build_methods(
    tree,
    sparse_tols: list[float],
    skip_mlf: bool,
) -> list[tuple[str, object]]:
    """Return (label, callable) pairs for all methods to benchmark."""
    from pointpv.likelihood.mlf import log_likelihood as mlf_logL
    from pointpv.likelihood.rg import log_likelihood as rg_logL

    methods: list[tuple[str, object]] = []

    if not skip_mlf:
        methods.append(("MLF", lambda u, C: mlf_logL(u, C)))

    methods.append(
        ("RG-dense", lambda u, C: rg_logL(u, C, tree=tree, verbose=False))
    )
    methods.append((
        "RG-schur",
        lambda u, C: rg_logL(u, C, tree=tree, schur_tol=1.0, verbose=False),
    ))

    for stol in sparse_tols:
        label = f"RG-sparse={_fmt_tol(stol)}"
        # capture stol by default argument to avoid closure-over-loop-variable
        methods.append((
            label,
            lambda u, C, _s=stol: rg_logL(
                u, C, tree=tree, schur_tol=1.0, sparse_tol=_s, verbose=False
            ),
        ))

    return methods


def _fmt_tol(v: float) -> str:
    """Format a sparse_tol value concisely: 1.0 → '1', 0.1 → '0.1', 1000.0 → '1000'."""
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_n(
    n: int,
    n_repeats: int,
    sparse_tols: list[float],
    skip_mlf: bool,
) -> dict:
    """
    Build a problem of size N, time all methods, and record logL values.

    Returns
    -------
    dict with keys: n, t_tree, methods (dict: label → {times, logL}).
    """
    from pointpv.rg.tree import build_tree

    print(f"  N={n}: constructing problem ...", flush=True)
    u, C, pos = _make_problem(n, seed=42 + n)

    print(f"  N={n}: building tree ...", flush=True)
    t0 = time.perf_counter()
    tree = build_tree(pos)
    t_tree = time.perf_counter() - t0
    print(f"  N={n}: tree built in {t_tree:.3f}s", flush=True)

    method_list = _build_methods(tree, sparse_tols, skip_mlf)

    results: dict = {"n": n, "t_tree": t_tree, "methods": {}}
    for label, fn in method_list:
        print(f"  N={n}: timing {label} ({n_repeats} repeat(s)) ...", flush=True)
        times: list[float] = []
        logL = float("nan")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            logL = fn(u, C)
            times.append(time.perf_counter() - t0)
        results["methods"][label] = {"times": times, "logL": logL}
        print(f"  N={n}: {label}  best={min(times):.4f}s  logL={logL:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_tables(all_results: list[dict]) -> None:
    """Print timing and accuracy tables."""
    # Collect all method labels in encounter order
    labels: list[str] = []
    for res in all_results:
        for lbl in res["methods"]:
            if lbl not in labels:
                labels.append(lbl)

    # --- timing table ---
    col_w = 12
    header = f"{'N':>8}" + "".join(f"  {lbl:>{col_w}}" for lbl in labels)
    print("\n=== Best wall time per evaluation (s) ===")
    print(header)
    print("-" * len(header))
    for res in all_results:
        row = f"{res['n']:>8d}"
        for lbl in labels:
            if lbl in res["methods"]:
                t = min(res["methods"][lbl]["times"])
                row += f"  {t:>{col_w}.4f}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)

    # --- accuracy table ---
    # |ΔlogL| relative to MLF (if present); otherwise relative to RG-dense
    ref_label = "MLF" if any("MLF" in r["methods"] for r in all_results) else "RG-dense"
    compare_labels = [l for l in labels if l != ref_label]

    print(f"\n=== |ΔlogL| vs {ref_label} ===")
    header2 = f"{'N':>8}  {ref_label + ' logL':>16}" + "".join(
        f"  {lbl:>{col_w}}" for lbl in compare_labels
    )
    print(header2)
    print("-" * len(header2))
    for res in all_results:
        n = res["n"]
        m = res["methods"]
        ref_logL = m[ref_label]["logL"] if ref_label in m else float("nan")
        row = f"{n:>8d}  {ref_logL:>16.4f}"
        for lbl in compare_labels:
            if lbl in m and not np.isnan(ref_logL):
                delta = abs(m[lbl]["logL"] - ref_logL)
                row += f"  {delta:>{col_w}.3e}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    all_results: list[dict],
    sparse_tols: list[float],
    output_path: str,
) -> None:
    """Two-panel figure: runtime scaling (top) and |ΔlogL| accuracy (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    Ns = np.array([r["n"] for r in all_results])

    def _best(res: dict, label: str) -> float:
        if label not in res["methods"]:
            return float("nan")
        return min(res["methods"][label]["times"])

    def _logL(res: dict, label: str) -> float:
        if label not in res["methods"]:
            return float("nan")
        return res["methods"][label]["logL"]

    t_mlf   = np.array([_best(r, "MLF")      for r in all_results])
    t_dense = np.array([_best(r, "RG-dense")  for r in all_results])
    t_schur = np.array([_best(r, "RG-schur")  for r in all_results])

    logL_mlf   = np.array([_logL(r, "MLF")     for r in all_results])
    logL_dense = np.array([_logL(r, "RG-dense") for r in all_results])
    logL_schur = np.array([_logL(r, "RG-schur") for r in all_results])

    has_mlf = not np.all(np.isnan(t_mlf))
    ref_logL = logL_mlf if has_mlf else logL_dense

    # Colors for sparse variants: evenly spaced from a green colormap
    n_sparse = len(sparse_tols)
    sparse_colors = [cm.YlOrRd(0.25 + 0.65 * i / max(n_sparse - 1, 1))
                     for i in range(n_sparse)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # ---- top panel: runtime ----
    if has_mlf:
        ax1.plot(Ns, t_mlf,   "o-",  color="#1f77b4", lw=2, label="MLF (Cholesky, O(N³))")
    ax1.plot(Ns, t_dense, "s--", color="#aec7e8", lw=1.5, label="RG-dense (O(N³))")
    ax1.plot(Ns, t_schur, "D-.", color="#ff7f0e", lw=1.5, label="RG-schur (schur_tol=1)")

    for i, stol in enumerate(sparse_tols):
        label = f"RG-sparse={_fmt_tol(stol)}"
        t_sp = np.array([_best(r, label) for r in all_results])
        ax1.plot(Ns, t_sp, "v-", color=sparse_colors[i], lw=1.5,
                 label=f"RG-sparse, tol={_fmt_tol(stol)} (km/s)²")

    # Reference lines anchored to first valid point
    N_ref = np.geomspace(Ns[0], Ns[-1], 300)
    anchor = Ns[0]
    if has_mlf and not np.isnan(t_mlf[0]):
        ax1.plot(N_ref, t_mlf[0] * (N_ref / anchor) ** 3,
                 color="#1f77b4", alpha=0.25, lw=1.2, linestyle=":")
    if not np.isnan(t_schur[0]):
        ax1.plot(N_ref, t_schur[0] * (N_ref / anchor) * np.log2(N_ref / anchor + 2),
                 color="#ff7f0e", alpha=0.25, lw=1.2, linestyle=":",
                 label=r"$N\log N$ ref")
        ax1.plot(N_ref, t_schur[0] * (N_ref / anchor) ** 3,
                 color="#ff7f0e", alpha=0.25, lw=1.2, linestyle="--")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (galaxies)", fontsize=11)
    ax1.set_ylabel("Best wall time per evaluation (s)", fontsize=11)
    ax1.set_title("Likelihood runtime scaling: MLF vs. McDonald RG variants", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left")

    # ---- bottom panel: accuracy ----
    ref_label = "MLF" if has_mlf else "RG-dense"
    ax2.plot(Ns, np.abs(logL_dense - ref_logL), "s--", color="#aec7e8", lw=1.5,
             label="RG-dense")
    ax2.plot(Ns, np.abs(logL_schur - ref_logL), "D-.", color="#ff7f0e", lw=1.5,
             label="RG-schur (schur_tol=1)")

    for i, stol in enumerate(sparse_tols):
        lbl = f"RG-sparse={_fmt_tol(stol)}"
        logL_sp = np.array([_logL(r, lbl) for r in all_results])
        delta = np.abs(logL_sp - ref_logL)
        # Replace zeros with NaN-safe small number for log plot
        delta = np.where(delta == 0, 1e-15, delta)
        ax2.plot(Ns, delta, "v-", color=sparse_colors[i], lw=1.5,
                 label=f"RG-sparse, tol={_fmt_tol(stol)} (km/s)²")

    ax2.axhline(1e-6, color="gray", lw=0.8, linestyle=":", alpha=0.7)
    ax2.text(Ns[-1] * 0.98, 1.5e-6, "1e-6", ha="right", va="bottom",
             color="gray", fontsize=8)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N (galaxies)", fontsize=11)
    ax2.set_ylabel(rf"$|\log L_{{\rm RG}} - \log L_{{\rm {ref_label}}}|$", fontsize=11)
    ax2.set_title("Likelihood accuracy vs. N (single evaluation, fixed fsigma8)", fontsize=12)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Runtime scaling benchmark: MLF vs McDonald RG variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 1000, 10000],
        help="Catalog sizes to benchmark",
    )
    p.add_argument(
        "--sparse-tols", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0],
        metavar="TOL",
        help="sparse_tol values (km/s)² for RG-sparse variants (all use schur_tol=1)",
    )
    p.add_argument(
        "--n-repeats", type=int, default=None,
        help="Timing repeats per method per N.  Default: 3 for N<2000, 1 for N>=2000.",
    )
    p.add_argument(
        "--skip-mlf-large", type=int, default=None, metavar="N",
        help="Omit MLF for N >= this value (MLF is O(N^3) and very slow at large N).",
    )
    p.add_argument(
        "--output-dir", default="figs",
        help="Directory for output figure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== benchmark_scaling.py ===")
    print(f"  sizes       = {args.sizes}")
    print(f"  sparse-tols = {args.sparse_tols}  (km/s)²")
    print(f"  skip-mlf-large = {args.skip_mlf_large}")
    print(f"  output-dir  = {args.output_dir}")

    if any(n >= 5000 for n in args.sizes):
        print(
            "\nWarning: large N requested."
            "\n  MLF (Cholesky) at N=10000 requires ~800 MB RAM and ~30-120 s."
            "\n  Use --skip-mlf-large 5000 to omit MLF for large N."
        )

    all_results: list[dict] = []
    for n in args.sizes:
        n_rep = args.n_repeats if args.n_repeats is not None else (1 if n >= 2000 else 3)
        skip_mlf = args.skip_mlf_large is not None and n >= args.skip_mlf_large
        tag = f"MLF skipped: N >= {args.skip_mlf_large}" if skip_mlf else f"n_repeats={n_rep}"
        print(f"\n--- N={n} ({tag}) ---")
        res = benchmark_n(n, n_rep, args.sparse_tols, skip_mlf=skip_mlf)
        all_results.append(res)

    print_tables(all_results)

    out_png = os.path.join(args.output_dir, "benchmark_scaling.png")
    plot_results(all_results, args.sparse_tols, out_png)


if __name__ == "__main__":
    main()
