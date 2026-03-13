#!/usr/bin/env python
"""
Diagnostic plots for a mock PV catalog.

Usage
-----
    python scripts/plot_catalog.py --catalog data/mock_1000.npz [--output figs/]

Produces:
    nz_<stem>.pdf   — n(z) histogram with analytic dN/dz reference
    sky_<stem>.pdf  — Mollweide sky map coloured by redshift
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot mock PV catalog diagnostics")
    p.add_argument("--catalog", required=True, help="Path to .npz catalog file")
    p.add_argument(
        "--output",
        default="figs/",
        help="Output directory for PDF figures (default: figs/)",
    )
    p.add_argument(
        "--m-lim",
        type=float,
        default=20.0,
        help="Apparent magnitude limit used when generating the catalog (default 20.0)",
    )
    p.add_argument(
        "--M-star",
        type=float,
        default=-21.5,
        help="Schechter M* (default -21.5)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=-1.1,
        help="Schechter faint-end slope (default -1.1)",
    )
    p.add_argument(
        "--M-faint",
        type=float,
        default=-17.0,
        help="Faint truncation of LF (default -17.0)",
    )
    p.add_argument("--bins", type=int, default=20, help="Number of z histogram bins")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Analytic dN/dz
# ---------------------------------------------------------------------------

def _schechter_integral(
    M_lim: float,
    M_star: float,
    alpha: float,
    M_faint: float,
    M_bright: float = -25.0,
) -> float:
    """Integrate Schechter LF from M_bright to min(M_lim, M_faint) numerically."""
    M_top = min(M_lim, M_faint)
    if M_top <= M_bright:
        return 0.0
    M_grid = np.linspace(M_bright, M_top, 2000)
    u = 10.0 ** (-0.4 * (M_grid - M_star))
    phi = u ** (alpha + 1.0) * np.exp(-u)
    return float(np.trapezoid(phi, M_grid))


def analytic_dndz(
    z_arr: np.ndarray,
    m_lim: float,
    M_star: float,
    alpha: float,
    M_faint: float,
    H0: float = 67.36,
    c_km_s: float = 2.998e5,
) -> np.ndarray:
    """
    Unnormalised dN/dz ∝ (dV/dz) × ∫ φ(M) dM  [M < M_lim(z)]

    For flat cosmology, small-z:
        dV/dz ∝ r_com² * dr_com/dz = (c*z/H0)² * (c/H0)
    """
    dndz = np.zeros_like(z_arr)
    for i, z in enumerate(z_arr):
        if z <= 0:
            continue
        r_com = c_km_s * z / H0          # Mpc
        d_L = r_com * (1.0 + z)          # Mpc
        M_lim_z = m_lim - 5.0 * np.log10(d_L) - 25.0
        phi_int = _schechter_integral(M_lim_z, M_star, alpha, M_faint)
        dvdz = r_com ** 2  # (c/H0) is a common prefactor, absorbed into norm
        dndz[i] = dvdz * phi_int
    return dndz


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_nz(
    z: np.ndarray,
    stem: str,
    outdir: str,
    m_lim: float,
    M_star: float,
    alpha: float,
    M_faint: float,
    n_bins: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    counts, edges = np.histogram(z, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    ax.bar(centers, counts, width=bin_width, alpha=0.7, label="catalog")

    # Analytic reference curve
    z_fine = np.linspace(edges[0], edges[-1], 400)
    dndz = analytic_dndz(z_fine, m_lim, M_star, alpha, M_faint)
    # Normalise to match histogram area
    norm = np.trapezoid(dndz, z_fine)
    if norm > 0:
        scale = counts.sum() * bin_width / norm
        ax.plot(z_fine, dndz * scale, "r-", lw=1.5, label="analytic dN/dz")

    ax.set_xlabel("z")
    ax.set_ylabel("N")
    ax.set_title(f"n(z) — {stem}")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(outdir, f"nz_{stem}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sky(z: np.ndarray, ra: np.ndarray, dec: np.ndarray, stem: str, outdir: str) -> None:
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="mollweide")

    # Mollweide expects longitude in [-π, π], latitude in [-π/2, π/2]
    lon = np.deg2rad(ra)
    lon[lon > np.pi] -= 2 * np.pi
    lat = np.deg2rad(dec)

    sc = ax.scatter(lon, lat, c=z, s=1, cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, label="z", shrink=0.6)
    ax.set_title(f"Sky distribution — {stem}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(outdir, f"sky_{stem}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cat_path = Path(args.catalog)
    if not cat_path.exists():
        print(f"ERROR: catalog file not found: {cat_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(cat_path)
    z = data["z_obs"]
    ra = data["ra"]
    dec = data["dec"]

    stem = cat_path.stem  # e.g. "mock_1000"

    os.makedirs(args.output, exist_ok=True)

    plot_nz(
        z, stem, args.output,
        m_lim=args.m_lim,
        M_star=args.M_star,
        alpha=args.alpha,
        M_faint=args.M_faint,
        n_bins=args.bins,
    )
    plot_sky(z, ra, dec, stem, args.output)


if __name__ == "__main__":
    main()
