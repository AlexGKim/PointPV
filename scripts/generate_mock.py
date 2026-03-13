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
        default=0.1,
        help="Maximum redshift cutoff (default 0.1)",
    )
    p.add_argument(
        "--m-lim",
        type=float,
        default=20.0,
        help="Apparent magnitude limit (default 20.0)",
    )
    p.add_argument(
        "--no-mag-limit",
        action="store_true",
        help="Disable magnitude limit; draw z uniformly (old behaviour)",
    )
    return p.parse_args()


def _sample_schechter(
    rng: np.random.Generator,
    n: int,
    M_star: float,
    alpha: float,
    M_faint: float,
    M_bright: float = -25.0,
) -> np.ndarray:
    """
    Sample absolute magnitudes from a Schechter luminosity function via
    rejection sampling.

    φ(M) ∝ 10^{-0.4(α+1)(M-M*)} exp(-10^{-0.4(M-M*)})
    """
    results = np.empty(n)
    filled = 0
    # Estimate φ_max: evaluate on a fine grid and take the maximum
    M_grid = np.linspace(M_bright, M_faint, 2000)
    u_grid = 10.0 ** (-0.4 * (M_grid - M_star))
    phi_grid = u_grid ** (alpha + 1.0) * np.exp(-u_grid)
    phi_max = phi_grid.max()

    batch = max(n * 10, 10000)
    while filled < n:
        M_try = rng.uniform(M_bright, M_faint, batch)
        u_try = 10.0 ** (-0.4 * (M_try - M_star))
        phi_try = u_try ** (alpha + 1.0) * np.exp(-u_try)
        accept = rng.uniform(0.0, phi_max, batch) < phi_try
        M_acc = M_try[accept]
        take = min(len(M_acc), n - filled)
        results[filled : filled + take] = M_acc[:take]
        filled += take
    return results


def generate_synthetic_catalog(
    n: int,
    sigma_eta: float = 0.2,
    seed: int = 42,
    z_min: float = 0.01,
    z_max: float = 0.1,
    m_lim: float = 20.0,
    M_star: float = -21.5,
    alpha: float = -1.1,
    M_faint: float = -17.0,
    oversample: int = 10,
    use_mag_limit: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic mock catalog with Gaussian random velocities.

    Positions are drawn with a magnitude-limited selection function by default.
    True velocities are zero (noise-only catalog for unit testing).

    Parameters
    ----------
    n : int
        Number of objects.
    sigma_eta : float
        Log-distance-ratio scatter.
    seed : int
    z_min, z_max : float
        Redshift range.
    m_lim : float
        Apparent magnitude limit (used when use_mag_limit=True).
    M_star, alpha, M_faint : float
        Schechter LF parameters.
    oversample : int
        Draw oversample * n candidates before applying magnitude cut.
    use_mag_limit : bool
        If False, skip magnitude selection and draw z uniformly (old behaviour).

    Returns
    -------
    dict
        Keys: 'ra', 'dec', 'z_obs', 'eta', 'sigma_eta', 'v_r_true', 'pos'
    """
    rng = np.random.default_rng(seed)

    H0 = 67.36
    c_km_s = 2.998e5

    if use_mag_limit:
        n_draw = oversample * n
        ra = rng.uniform(0.0, 360.0, n_draw)
        dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_draw)))
        z_obs = rng.uniform(z_min, z_max, n_draw)

        # Luminosity distance in Mpc (flat, small-z approx)
        r_com_mpc = c_km_s * z_obs / H0          # Mpc
        d_L = r_com_mpc * (1.0 + z_obs)          # Mpc

        # Sample absolute magnitudes from Schechter LF
        M_abs = _sample_schechter(rng, n_draw, M_star, alpha, M_faint)

        # Apparent magnitude
        m = M_abs + 5.0 * np.log10(d_L) + 25.0

        mask = m < m_lim
        n_pass = mask.sum()
        if n_pass < n:
            raise RuntimeError(
                f"Only {n_pass} objects pass m < {m_lim} from {n_draw} draws. "
                f"Increase oversample (current={oversample}) or raise z_max."
            )

        idx = np.where(mask)[0][:n]
        ra = ra[idx]
        dec = dec[idx]
        z_obs = z_obs[idx]
        r_com = c_km_s * z_obs / H0  # Mpc (same H0, no h factor needed for pos)
    else:
        ra = rng.uniform(0.0, 360.0, n)
        dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
        z_obs = rng.uniform(z_min, z_max, n)
        r_com = c_km_s * z_obs / H0

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
        use_mag = not args.no_mag_limit
        print(
            f"Generating synthetic catalog with N={args.n}, "
            f"z_max={args.z_max}, "
            + (f"m_lim={args.m_lim}" if use_mag else "no magnitude limit")
            + " ..."
        )
        catalog = generate_synthetic_catalog(
            n=args.n,
            sigma_eta=args.sigma_eta,
            seed=args.seed,
            z_max=args.z_max,
            m_lim=args.m_lim,
            use_mag_limit=use_mag,
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
