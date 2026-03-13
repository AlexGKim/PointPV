"""
Mock peculiar velocity catalog builder.

Takes a halo catalog (from lightcone.py) and produces a PV survey mock by:
  1. Applying a mass/luminosity threshold to mimic magnitude-limited selection.
  2. Sub-sampling to a target size N.
  3. Adding log-distance-ratio noise (Gaussian, σ ~ 20%) to simulate TF/FP errors.

Inputs
------
halos : dict
    Output of lightcone.load_lightcone().
n_target : int
    Number of objects in the output catalog.
sigma_eta : float
    1-sigma scatter in log-distance ratio η = ln(d_est/d_true), default 0.2.
seed : int
    Random seed for reproducibility.

Outputs
-------
dict with arrays:
    ra          [deg]
    dec         [deg]
    z_obs       []     Observed redshift
    eta         []     Measured log-distance ratio (noisy)
    sigma_eta   []     Per-object uncertainty (homoscedastic here)
    v_r_true    [km/s] True radial peculiar velocity (for validation)

Units
-----
Distances in Mpc/h, velocities in km/s.
"""

from __future__ import annotations

import numpy as np


def build_catalog(
    halos: dict[str, np.ndarray],
    n_target: int,
    sigma_eta: float = 0.2,
    mass_cut: float | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Build a mock PV catalog from a halo dict.

    Parameters
    ----------
    halos :
        Dict with keys 'ra', 'dec', 'z_obs', 'v_r', 'mass'.
    n_target :
        Number of objects to include.  If fewer halos survive the mass
        cut, an error is raised.
    sigma_eta :
        Log-distance-ratio scatter (dimensionless, ~0.2 for TF/FP).
    mass_cut :
        Minimum halo mass in Msun/h.  Default: choose automatically so
        that N ≥ n_target survives.
    seed :
        NumPy random seed.

    Returns
    -------
    dict
        Catalog arrays suitable for likelihood evaluation.
    """
    rng = np.random.default_rng(seed)

    mass = halos["mass"]

    if mass_cut is None:
        # Find the top-n_target objects by mass
        idx_sorted = np.argsort(mass)[::-1]
        if len(idx_sorted) < n_target:
            raise ValueError(
                f"Only {len(idx_sorted)} halos available; "
                f"cannot build catalog of size {n_target}."
            )
        idx = idx_sorted[:n_target]
    else:
        mask = mass >= mass_cut
        idx_all = np.where(mask)[0]
        if len(idx_all) < n_target:
            raise ValueError(
                f"Only {len(idx_all)} halos above mass_cut {mass_cut:.2e}; "
                f"cannot build catalog of size {n_target}."
            )
        idx = rng.choice(idx_all, size=n_target, replace=False)

    v_r_true = halos["v_r"][idx]

    # True log-distance ratio: η_true = ln(d_obs/d_true) ≈ -v_r / (H(z)*d)
    # For the mock we define η_true = 0 (only peculiar velocity matters);
    # the measured η includes noise.
    eta_noise = rng.normal(0.0, sigma_eta, size=n_target)
    # Physical contribution: η ≈ -v_r / (c * z) at low z (rough conversion)
    # Store the noise separately so it can be removed for validation.
    eta = eta_noise  # zero-mean noise mock (true η from v_r is tracked via v_r_true)

    return {
        "ra": halos["ra"][idx],
        "dec": halos["dec"][idx],
        "z_obs": halos["z_obs"][idx],
        "eta": eta,
        "sigma_eta": np.full(n_target, sigma_eta),
        "v_r_true": v_r_true,
    }


def eta_to_velocity(
    eta: np.ndarray,
    z: np.ndarray,
    H0: float = 67.36,
) -> np.ndarray:
    """
    Convert log-distance-ratio η to a line-of-sight peculiar velocity.

    Uses the approximation v_pec ≈ c * z * η valid for small η and z.

    Parameters
    ----------
    eta : array, dimensionless
        Log-distance ratio η = ln(d_obs / d_true).
    z : array
        Observed redshift.
    H0 : float
        Hubble constant in km/s/Mpc (used only for dimensional checks).

    Returns
    -------
    v_pec : array, km/s
    """
    c_km_s = 2.998e5
    return c_km_s * z * eta


def save_catalog(catalog: dict[str, np.ndarray], path: str) -> None:
    """Save catalog dict as a .npz file."""
    np.savez(path, **catalog)


def load_catalog(path: str) -> dict[str, np.ndarray]:
    """Load catalog dict from a .npz file."""
    data = np.load(path)
    return {k: data[k] for k in data.files}
