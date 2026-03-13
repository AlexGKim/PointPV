#!/usr/bin/env python
"""
Run the McDonald RG log-likelihood scan over fsigma8.

Usage
-----
    python scripts/run_rg.py --n 1000 --backend scipy
    python scripts/run_rg.py --catalog data/mock_1000.npz --output results/rg_1000.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RG method fsigma8 scan")
    p.add_argument("--catalog", default=None,
                   help="Path to .npz catalog.  Default: data/mock_{n}.npz")
    p.add_argument("--n", type=int, default=1000,
                   help="Catalog size (used to locate default catalog path)")
    p.add_argument("--output", default=None,
                   help="Output .npz path.  Default: results/rg_{n}.npz")
    p.add_argument("--backend", default=None,
                   choices=["scipy", "petsc"],
                   help="Sparse backend (overrides POINTPV_BACKEND)")
    p.add_argument("--fs8-min", type=float, default=0.2)
    p.add_argument("--fs8-max", type=float, default=0.8)
    p.add_argument("--n-grid", type=int, default=20,
                   help="Number of fsigma8 grid points")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate a synthetic catalog on the fly (no file needed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schur-tol", type=float, default=0.0,
                   help="Schur update cutoff for speed/accuracy tradeoff (0=exact)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend is not None:
        os.environ["POINTPV_BACKEND"] = args.backend

    # ---- load or generate catalog ----
    if args.synthetic:
        from scripts.generate_mock import generate_synthetic_catalog
        print(f"Generating synthetic catalog N={args.n} ...")
        catalog = generate_synthetic_catalog(n=args.n, seed=args.seed)
    else:
        catalog_path = args.catalog or f"data/mock_{args.n}.npz"
        if not Path(catalog_path).exists():
            print(f"ERROR: catalog not found at {catalog_path}.\n"
                  "Run scripts/generate_mock.py first, or pass --synthetic.",
                  file=sys.stderr)
            sys.exit(1)
        from pointpv.mock.catalog import load_catalog
        catalog = load_catalog(catalog_path)
        print(f"Loaded catalog ({len(catalog['ra'])} objects) from {catalog_path}")

    # ---- extract positions ----
    if "pos" not in catalog:
        # Reconstruct comoving positions from (ra, dec, z_obs)
        H0 = 67.36
        c_km_s = 2.998e5
        z = catalog["z_obs"]
        r = c_km_s * z / H0
        ra_rad = np.deg2rad(catalog["ra"])
        dec_rad = np.deg2rad(catalog["dec"])
        cos_dec = np.cos(dec_rad)
        positions = np.column_stack([
            r * cos_dec * np.cos(ra_rad),
            r * cos_dec * np.sin(ra_rad),
            r * np.sin(dec_rad),
        ])
    else:
        positions = catalog["pos"]

    # ---- build velocity vector ----
    from pointpv.mock.catalog import eta_to_velocity
    u = eta_to_velocity(catalog["eta"], catalog["z_obs"])

    # ---- run scan ----
    from pointpv.likelihood.rg import scan_fsigma8
    fsigma8_values = np.linspace(args.fs8_min, args.fs8_max, args.n_grid)
    print(f"Running RG scan over {args.n_grid} fsigma8 values ...")
    results = scan_fsigma8(u, catalog, positions, fsigma8_values=fsigma8_values,
                           schur_tol=args.schur_tol, verbose=not args.quiet)

    best_idx = int(np.argmax(results["logL"]))
    best_fs8 = results["fsigma8"][best_idx]
    print(f"\nBest-fit fsigma8 = {best_fs8:.4f}  "
          f"(logL = {results['logL'][best_idx]:.4f})")
    print(f"Mean time per eval: {results['time_per_eval'].mean():.3f}s")

    # ---- save results ----
    output = args.output
    if output is None:
        os.makedirs("results", exist_ok=True)
        output = f"results/rg_{args.n}.npz"
    np.savez(output, method="rg", n=args.n, **results)
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
