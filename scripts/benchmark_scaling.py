#!/usr/bin/env python
"""
Runtime scaling benchmark: MLF (Cholesky) vs. McDonald RG likelihood.

Measures wall-clock time per likelihood evaluation at multiple catalog sizes,
compares log-likelihood values between methods, and produces a two-panel
scaling + accuracy figure.

Fixed baselines always timed:
  MLF       — direct Cholesky, O(N^3)
  RG-dense  — McDonald RG, dense path (schur_tol=0, sparse_tol=0), O(N^3)

Configurable variants:
  RG-schur=X  — one curve per --schur-tols value (sparse_tol=0, schur_tol=X)
  RG-sparse=X — one curve per --sparse-tols value (schur_tol from
                --sparse-schur-tol, default 1.0; sparse_tol=X)

Usage
-----
    python scripts/benchmark_scaling.py
    python scripts/benchmark_scaling.py --sizes 100 500 1000 2000
    python scripts/benchmark_scaling.py --schur-tols 0.1 0.5 1.0 5.0
    python scripts/benchmark_scaling.py --schur-tols 0.5 1.0 --sparse-tols 1 100
    python scripts/benchmark_scaling.py --sparse-tols --schur-tols 0.1 0.5 1.0
    python scripts/benchmark_scaling.py --skip-mlf-large 5000

Notes
-----
* For N >= 2000, set --n-repeats 1 to keep runtime manageable.
* MLF at N=10000 requires ~800 MB RAM and can take 30-120 s.
* For a full fsigma8-grid accuracy check at one N, see validate_fsigma8.py.
* The sparse path requires schur_tol > 0 (see --sparse-schur-tol).
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
    """Build a synthetic PV problem with N galaxies using cdist (memory-efficient)."""
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

def _fmt_tol(v: float) -> str:
    """Format a tolerance value concisely: 1.0 → '1', 0.1 → '0.1', 1000.0 → '1000'."""
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def _build_methods(
    tree,
    schur_tols: list[float],
    sparse_tols: list[float],
    sparse_schur_tol: float,
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

    for stol in schur_tols:
        label = f"RG-schur={_fmt_tol(stol)}"
        methods.append((
            label,
            lambda u, C, _s=stol: rg_logL(u, C, tree=tree, schur_tol=_s, verbose=False),
        ))

    for ptol in sparse_tols:
        label = f"RG-sparse={_fmt_tol(ptol)}"
        methods.append((
            label,
            lambda u, C, _p=ptol, _s=sparse_schur_tol: rg_logL(
                u, C, tree=tree, schur_tol=_s, sparse_tol=_p, verbose=False
            ),
        ))

    return methods


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_n(
    n: int,
    n_repeats: int,
    schur_tols: list[float],
    sparse_tols: list[float],
    sparse_schur_tol: float,
    skip_mlf: bool,
) -> dict:
    """Build a problem of size N, time all methods, and record logL values."""
    from pointpv.rg.tree import build_tree

    print(f"  N={n}: constructing problem ...", flush=True)
    u, C, pos = _make_problem(n, seed=42 + n)

    print(f"  N={n}: building tree ...", flush=True)
    t0 = time.perf_counter()
    tree = build_tree(pos)
    t_tree = time.perf_counter() - t0
    print(f"  N={n}: tree built in {t_tree:.3f}s", flush=True)

    method_list = _build_methods(tree, schur_tols, sparse_tols, sparse_schur_tol, skip_mlf)

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
    labels: list[str] = []
    for res in all_results:
        for lbl in res["methods"]:
            if lbl not in labels:
                labels.append(lbl)

    col_w = 14

    # --- timing table ---
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
    schur_tols: list[float],
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
    logL_mlf   = np.array([_logL(r, "MLF")     for r in all_results])
    logL_dense = np.array([_logL(r, "RG-dense") for r in all_results])
    has_mlf = not np.all(np.isnan(t_mlf))
    ref_logL = logL_mlf if has_mlf else logL_dense
    ref_label = "MLF" if has_mlf else "RG-dense"

    # Colour ramps: orange family for schur variants, red/yellow for sparse variants
    n_schur  = len(schur_tols)
    n_sparse = len(sparse_tols)
    schur_colors  = [cm.Oranges(0.35 + 0.55 * i / max(n_schur  - 1, 1)) for i in range(n_schur)]
    sparse_colors = [cm.YlOrRd(0.30 + 0.60 * i / max(n_sparse - 1, 1)) for i in range(n_sparse)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # ---- top panel: runtime ----
    if has_mlf:
        ax1.plot(Ns, t_mlf,   "o-",  color="#1f77b4", lw=2, label="MLF (Cholesky, O(N³))")
    ax1.plot(Ns, t_dense, "s--", color="#aec7e8", lw=1.5, label="RG-dense (O(N³))")

    for i, stol in enumerate(schur_tols):
        lbl = f"RG-schur={_fmt_tol(stol)}"
        t_s = np.array([_best(r, lbl) for r in all_results])
        ax1.plot(Ns, t_s, "D-.", color=schur_colors[i], lw=1.5,
                 label=f"RG-schur, tol={_fmt_tol(stol)}")

    for i, ptol in enumerate(sparse_tols):
        lbl = f"RG-sparse={_fmt_tol(ptol)}"
        t_p = np.array([_best(r, lbl) for r in all_results])
        ax1.plot(Ns, t_p, "v-", color=sparse_colors[i], lw=1.5,
                 label=f"RG-sparse, tol={_fmt_tol(ptol)} (km/s)²")

    # Reference lines anchored to N[0]
    N_ref  = np.geomspace(Ns[0], Ns[-1], 300)
    anchor = Ns[0]
    if has_mlf and not np.isnan(t_mlf[0]):
        ax1.plot(N_ref, t_mlf[0] * (N_ref / anchor) ** 3,
                 color="#1f77b4", alpha=0.20, lw=1.2, linestyle=":", label=r"$N^3$ ref")
    # Anchor N log N to the first schur or sparse variant with valid timing
    _t_fast = None
    for stol in schur_tols:
        candidate = np.array([_best(r, f"RG-schur={_fmt_tol(stol)}") for r in all_results])
        if not np.isnan(candidate[0]):
            _t_fast = candidate[0]
            break
    if _t_fast is not None:
        ax1.plot(N_ref, _t_fast * (N_ref / anchor) * np.log2(N_ref / anchor + 2),
                 color="#ff7f0e", alpha=0.20, lw=1.2, linestyle=":", label=r"$N\log N$ ref")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (galaxies)", fontsize=11)
    ax1.set_ylabel("Best wall time per evaluation (s)", fontsize=11)
    ax1.set_title("Likelihood runtime scaling: MLF vs. McDonald RG variants", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left")

    # ---- bottom panel: accuracy ----
    ax2.plot(Ns, np.abs(logL_dense - ref_logL), "s--", color="#aec7e8", lw=1.5,
             label="RG-dense")

    for i, stol in enumerate(schur_tols):
        lbl = f"RG-schur={_fmt_tol(stol)}"
        logL_s = np.array([_logL(r, lbl) for r in all_results])
        delta = np.where(np.abs(logL_s - ref_logL) == 0, 1e-15, np.abs(logL_s - ref_logL))
        ax2.plot(Ns, delta, "D-.", color=schur_colors[i], lw=1.5,
                 label=f"RG-schur, tol={_fmt_tol(stol)}")

    for i, ptol in enumerate(sparse_tols):
        lbl = f"RG-sparse={_fmt_tol(ptol)}"
        logL_p = np.array([_logL(r, lbl) for r in all_results])
        delta = np.where(np.abs(logL_p - ref_logL) == 0, 1e-15, np.abs(logL_p - ref_logL))
        ax2.plot(Ns, delta, "v-", color=sparse_colors[i], lw=1.5,
                 label=f"RG-sparse, tol={_fmt_tol(ptol)} (km/s)²")

    ax2.axhline(1e-6, color="gray", lw=0.8, linestyle=":", alpha=0.7)
    ax2.text(Ns[-1] * 0.98, 1.5e-6, "1e-6", ha="right", va="bottom",
             color="gray", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N (galaxies)", fontsize=11)
    ax2.set_ylabel(rf"$|\log L_\mathrm{{RG}} - \log L_\mathrm{{{ref_label}}}|$", fontsize=11)
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
        "--schur-tols", type=float, nargs="*", default=[1.0], metavar="TOL",
        help="schur_tol values for RG-schur variants (sparse_tol=0 for these curves)",
    )
    p.add_argument(
        "--sparse-tols", type=float, nargs="*", default=[1.0, 10.0, 100.0],
        metavar="TOL",
        help="sparse_tol values (km/s)² for RG-sparse variants; pass no values to suppress",
    )
    p.add_argument(
        "--sparse-schur-tol", type=float, default=1.0, metavar="TOL",
        help="schur_tol used for all RG-sparse variants (must be > 0)",
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

    schur_tols  = args.schur_tols  or []
    sparse_tols = args.sparse_tols or []

    if sparse_tols and args.sparse_schur_tol <= 0:
        print("ERROR: --sparse-schur-tol must be > 0 when --sparse-tols are specified.",
              file=sys.stderr)
        sys.exit(1)

    print("=== benchmark_scaling.py ===")
    print(f"  sizes            = {args.sizes}")
    print(f"  schur-tols       = {schur_tols}")
    print(f"  sparse-tols      = {sparse_tols}  (km/s)²")
    print(f"  sparse-schur-tol = {args.sparse_schur_tol}")
    print(f"  skip-mlf-large   = {args.skip_mlf_large}")
    print(f"  output-dir       = {args.output_dir}")

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
        res = benchmark_n(n, n_rep, schur_tols, sparse_tols, args.sparse_schur_tol,
                          skip_mlf=skip_mlf)
        all_results.append(res)

    print_tables(all_results)

    out_png = os.path.join(args.output_dir, "benchmark_scaling.png")
    plot_results(all_results, schur_tols, sparse_tols, out_png)


if __name__ == "__main__":
    main()
