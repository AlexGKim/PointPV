#!/usr/bin/env python
"""
Science-level fsigma8 recovery validation.

Generates a synthetic catalog, scans both MLF and RG likelihoods over a
fsigma8 grid, compares accuracy and timing, and produces all diagnostic plots.

Usage
-----
    python scripts/validate_fsigma8.py --n 200 --n-grid 40
    python scripts/validate_fsigma8.py --n 50 --flip   # real FLIP covariance (slow)

Exit code
---------
0  if max|ΔlogL| ≤ 1e-4
1  if max|ΔlogL| > 1e-4  (accuracy failure)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so pointpv and scripts are importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_problem(
    n: int,
    seed: int = 42,
    length_scale: float = 50.0,
    sigma_v: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic PV problem with N galaxies.

    Returns u, C, pos where C is an exponential covariance at fsigma8=0.47
    (sigma_v is calibrated so C ∝ fsigma8² * sigma_v_base²).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 500, size=(n, 3))
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    C = sigma_v**2 * np.exp(-r / length_scale)
    C += np.eye(n) * (sigma_v * 0.05) ** 2
    u = rng.standard_normal(n) * sigma_v * 0.1
    return u, C, pos


def _scan_synthetic(
    u: np.ndarray,
    C_ref: np.ndarray,
    pos: np.ndarray,
    fs8_ref: float,
    fs8_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan both MLF and RG over a fsigma8 grid using a scaled synthetic covariance.

    The covariance scales as C(fs8) = (fs8/fs8_ref)^2 * C_ref.
    """
    from pointpv.likelihood.mlf import log_likelihood as mlf_logL
    from pointpv.likelihood.rg import log_likelihood as rg_logL

    logL_mlf = np.empty(len(fs8_values))
    logL_rg = np.empty(len(fs8_values))
    times_mlf = np.empty(len(fs8_values))
    times_rg = np.empty(len(fs8_values))

    for i, fs8 in enumerate(fs8_values):
        C = (fs8 / fs8_ref) ** 2 * C_ref

        t0 = time.perf_counter()
        logL_mlf[i] = mlf_logL(u, C)
        times_mlf[i] = time.perf_counter() - t0

        t0 = time.perf_counter()
        logL_rg[i] = rg_logL(u, C, pos, verbose=False)
        times_rg[i] = time.perf_counter() - t0

    return logL_mlf, logL_rg, times_mlf, times_rg


def _scan_flip(
    catalog: dict,
    pos: np.ndarray,
    fs8_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan using real FLIP/CAMB covariance pipeline."""
    from pointpv.covariance.velocity import build_covariance
    from pointpv.likelihood.mlf import log_likelihood as mlf_logL
    from pointpv.likelihood.rg import log_likelihood as rg_logL
    from pointpv.mock.catalog import eta_to_velocity

    u = eta_to_velocity(catalog["eta"], catalog["z_obs"])
    logL_mlf = np.empty(len(fs8_values))
    logL_rg = np.empty(len(fs8_values))
    times_mlf = np.empty(len(fs8_values))
    times_rg = np.empty(len(fs8_values))

    for i, fs8 in enumerate(fs8_values):
        C = build_covariance(catalog, fsigma8=fs8)

        t0 = time.perf_counter()
        logL_mlf[i] = mlf_logL(u, C)
        times_mlf[i] = time.perf_counter() - t0

        t0 = time.perf_counter()
        logL_rg[i] = rg_logL(u, C, pos, verbose=False)
        times_rg[i] = time.perf_counter() - t0

    return logL_mlf, logL_rg, times_mlf, times_rg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Science-level fsigma8 recovery validation")
    p.add_argument("--n", type=int, default=200, help="Catalog size (default 200)")
    p.add_argument("--n-grid", type=int, default=40, help="fsigma8 grid points (default 40)")
    p.add_argument(
        "--fs8-min", type=float, default=0.2, help="fsigma8 grid lower bound (default 0.2)"
    )
    p.add_argument(
        "--fs8-max", type=float, default=0.8, help="fsigma8 grid upper bound (default 0.8)"
    )
    p.add_argument(
        "--fs8-truth", type=float, default=0.47, help="True fsigma8 (default 0.47)"
    )
    p.add_argument(
        "--flip", action="store_true", help="Use real FLIP/CAMB covariance (slow)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir", default="figs", help="Output directory for figures (default: figs/)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fs8_grid = np.linspace(args.fs8_min, args.fs8_max, args.n_grid)
    fs8_step = fs8_grid[1] - fs8_grid[0]

    print(f"=== validate_fsigma8.py ===")
    print(f"  N={args.n}  grid={args.n_grid} pts [{args.fs8_min:.2f}, {args.fs8_max:.2f}]"
          f"  fs8_truth={args.fs8_truth}  flip={'yes' if args.flip else 'no'}")

    # --- generate catalog ---
    if args.flip:
        from generate_mock import generate_synthetic_catalog
        print(f"\nGenerating synthetic catalog for FLIP (N={args.n}) ...")
        catalog = generate_synthetic_catalog(n=args.n, seed=args.seed, use_mag_limit=False)
        pos = catalog["pos"]
        print("Running FLIP/CAMB scan (may take a few minutes) ...")
        logL_mlf, logL_rg, times_mlf, times_rg = _scan_flip(catalog, pos, fs8_grid)
        # Also plot catalog diagnostics
        from plot_catalog import plot_nz, plot_sky
        plot_nz(
            catalog["z_obs"], stem="validate", outdir=args.output_dir,
            m_lim=20.0, M_star=-21.5, alpha=-1.1, M_faint=-17.0,
        )
        plot_sky(catalog["z_obs"], catalog["ra"], catalog["dec"],
                 stem="validate", outdir=args.output_dir)
    else:
        print(f"\nGenerating synthetic catalog with exponential covariance (N={args.n}) ...")
        sigma_v = 300.0
        u, C_ref, pos = _make_synthetic_problem(args.n, seed=args.seed, sigma_v=sigma_v)

        # Fake catalog for plot_nz / plot_sky
        rng = np.random.default_rng(args.seed)
        H0, c_kms = 67.36, 2.998e5
        r_mpc = np.linalg.norm(pos, axis=1)
        z_obs = r_mpc * H0 / c_kms
        ra = rng.uniform(0, 360, args.n)
        dec = np.rad2deg(np.arcsin(rng.uniform(-1, 1, args.n)))
        catalog_meta = {"z_obs": z_obs, "ra": ra, "dec": dec}

        from plot_catalog import plot_nz, plot_sky
        plot_nz(
            z_obs, stem="validate", outdir=args.output_dir,
            m_lim=20.0, M_star=-21.5, alpha=-1.1, M_faint=-17.0,
        )
        plot_sky(z_obs, ra, dec, stem="validate", outdir=args.output_dir)

        print(f"Scanning {args.n_grid} fsigma8 points ...")
        logL_mlf, logL_rg, times_mlf, times_rg = _scan_synthetic(
            u, C_ref, pos, args.fs8_truth, fs8_grid
        )

    # --- accuracy report ---
    from pointpv.benchmark.accuracy import compare_logL, print_accuracy_report, plot_comparison
    comparison = compare_logL(fs8_grid, logL_mlf, logL_rg)
    print_accuracy_report(comparison)

    # Report best-fit and speedup
    fs8_best_mlf = comparison["fsigma8_mlf"]
    fs8_best_rg = comparison["fsigma8_rg"]
    mean_t_mlf = float(np.mean(times_mlf))
    mean_t_rg = float(np.mean(times_rg))
    speedup = mean_t_mlf / max(mean_t_rg, 1e-9)

    print(f"\n  Best-fit fsigma8 — MLF: {fs8_best_mlf:.4f}  RG: {fs8_best_rg:.4f}")
    print(f"  Mean eval time  — MLF: {mean_t_mlf:.4f}s  RG: {mean_t_rg:.4f}s"
          f"  speedup: {speedup:.1f}×")

    # --- accuracy plot ---
    compare_png = os.path.join(args.output_dir, "validate_compare.png")
    plot_comparison(fs8_grid, logL_mlf, logL_rg, output_path=compare_png)
    print(f"\nSaved {compare_png}")

    # --- scaling plot (single-N timing) ---
    from pointpv.benchmark.timing import plot_scaling
    results_timing = {
        "MLF": {
            "N": np.array([args.n]),
            "mean_time": np.array([mean_t_mlf]),
            "std_time": np.array([0.0]),
            "logL": np.array([logL_mlf[np.argmax(logL_mlf)]]),
        },
        "RG": {
            "N": np.array([args.n]),
            "mean_time": np.array([mean_t_rg]),
            "std_time": np.array([0.0]),
            "logL": np.array([logL_rg[np.argmax(logL_rg)]]),
        },
    }
    scaling_png = os.path.join(args.output_dir, "validate_scaling.png")
    plot_scaling(results_timing, output_path=scaling_png)
    print(f"Saved {scaling_png}")

    # --- exit code based on accuracy ---
    max_diff = comparison["max_abs_diff"]
    threshold = 1e-4
    if max_diff > threshold:
        print(f"\nFAILURE: max|ΔlogL| = {max_diff:.2e} exceeds threshold {threshold:.0e}",
              file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nPASSED: max|ΔlogL| = {max_diff:.2e} ≤ {threshold:.0e}")


if __name__ == "__main__":
    main()
