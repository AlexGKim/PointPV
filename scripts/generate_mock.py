#!/usr/bin/env python
"""
Generate a mock peculiar velocity catalog from AbacusSummit light cone data.

Usage
-----
    python scripts/generate_mock.py --n 1000 --output data/mock_1000.npz
    python scripts/generate_mock.py --n 10000 --lightcone /path/to/abacus/lightcone

For local development without the full light cone, pass --synthetic to
generate a Gaussian random catalog without reading any ASDF files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mock PV catalog")
    p.add_argument("--n", type=int, default=1000, help="Catalog size")
    p.add_argument(
        "--lightcone",
        default=None,
        help="Path to AbacusSummit light cone directory. "
             "If omitted, uses $ABACUS_LIGHTCONE env var.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Default: data/mock_{N}.npz",
    )
    p.add_argument(
        "--sigma-eta",
        type=float,
        default=0.2,
        help="Log-distance-ratio scatter (default 0.2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic Gaussian catalog (no abacusutils needed)",
    )
    p.add_argument(
        "--z-max",
        type=float,
        default=0.8,
        help="Maximum redshift cutoff (default 0.8)",
    )
    return p.parse_args()


def generate_synthetic_catalog(
    n: int,
    sigma_eta: float = 0.2,
    seed: int = 42,
    z_min: float = 0.01,
    z_max: float = 0.3,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic mock catalog with Gaussian random velocities.

    Positions are drawn from a random distribution in a survey volume.
    True velocities are zero (noise-only catalog for unit testing).

    Parameters
    ----------
    n : int
        Number of objects.
    sigma_eta : float
        Log-distance-ratio scatter.
    seed : int
    z_min, z_max : float
        Redshift range for uniform distribution.

    Returns
    -------
    dict
        Keys: 'ra', 'dec', 'z_obs', 'eta', 'sigma_eta', 'v_r_true', 'pos'
    """
    rng = np.random.default_rng(seed)

    ra = rng.uniform(0.0, 360.0, n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    z_obs = rng.uniform(z_min, z_max, n)

    # Comoving distance: approximate r ≈ c*z/H0 for small z
    H0 = 67.36
    c_km_s = 2.998e5
    r_com = c_km_s * z_obs / H0  # Mpc/h (rough)

    # Cartesian positions
    cos_dec = np.cos(np.deg2rad(dec))
    x = r_com * cos_dec * np.cos(np.deg2rad(ra))
    y = r_com * cos_dec * np.sin(np.deg2rad(ra))
    z_cart = r_com * np.sin(np.deg2rad(dec))
    pos = np.column_stack([x, y, z_cart])

    eta = rng.normal(0.0, sigma_eta, n)
    v_r_true = np.zeros(n)

    return {
        "ra": ra,
        "dec": dec,
        "z_obs": z_obs,
        "eta": eta,
        "sigma_eta": np.full(n, sigma_eta),
        "v_r_true": v_r_true,
        "pos": pos,
    }


def main() -> None:
    args = parse_args()

    output = args.output
    if output is None:
        os.makedirs("data", exist_ok=True)
        output = f"data/mock_{args.n}.npz"

    if args.synthetic:
        print(f"Generating synthetic catalog with N={args.n}...")
        catalog = generate_synthetic_catalog(
            n=args.n,
            sigma_eta=args.sigma_eta,
            seed=args.seed,
        )
    else:
        lightcone_path = args.lightcone or os.environ.get("ABACUS_LIGHTCONE")
        if lightcone_path is None:
            print(
                "ERROR: Provide --lightcone path or set $ABACUS_LIGHTCONE.\n"
                "Use --synthetic for a test catalog without abacusutils.",
                file=sys.stderr,
            )
            sys.exit(1)

        from pointpv.mock.lightcone import load_lightcone
        from pointpv.mock.catalog import build_catalog

        print(f"Loading AbacusSummit light cone from {lightcone_path} ...")
        halos = load_lightcone(lightcone_path, z_max=args.z_max)
        print(f"  Loaded {len(halos['ra'])} halos after z < {args.z_max} cut.")

        print(f"Building catalog with N={args.n} ...")
        catalog = build_catalog(
            halos,
            n_target=args.n,
            sigma_eta=args.sigma_eta,
            seed=args.seed,
        )

    np.savez(output, **catalog)
    print(f"Saved catalog ({args.n} objects) to {output}")


if __name__ == "__main__":
    main()
