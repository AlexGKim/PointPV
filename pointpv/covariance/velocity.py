"""
Velocity-velocity covariance matrix builder.

Wraps the FLIP package to compute the N×N galaxy–galaxy velocity covariance
matrix C_{ij}(fsigma8, cosmology) for a peculiar velocity survey.

The covariance between two galaxies at positions r_i, r_j separated by angle θ
is given by the line-of-sight projection of the matter velocity power spectrum:

    C_{ij} = fsigma8² * ∫ dk/(2π²) P_lin(k) W(k, r_i, r_j, θ)

where W is the window function encoding the projection geometry and
observational noise is added on the diagonal.

Inputs
------
catalog : dict
    Keys: 'ra' [deg], 'dec' [deg], 'z_obs', 'sigma_eta' [dimensionless].
fsigma8 : float
    Growth rate times sigma_8 (the single free parameter).
cosmology : dict or astropy Cosmology
    Background cosmology (H0, Omega_m, …).

Outputs
-------
C : (N, N) ndarray, (km/s)²
    Full velocity-velocity covariance matrix, including noise diagonal.

Units
-----
All matrix elements in (km/s)².
"""

from __future__ import annotations

import numpy as np


def build_covariance(
    catalog: dict[str, np.ndarray],
    fsigma8: float,
    cosmology: dict | None = None,
    cutoff_mpc: float | None = None,
) -> np.ndarray:
    """
    Build the velocity-velocity covariance matrix via FLIP.

    Parameters
    ----------
    catalog :
        Must contain 'ra' [deg], 'dec' [deg], 'z_obs', 'sigma_eta'.
    fsigma8 :
        Linear growth rate × sigma_8.
    cosmology :
        Dict with keys 'H0', 'Omega_m', 'Omega_b', 'n_s', 'sigma8'.
        Defaults to AbacusSummit base cosmology.
    cutoff_mpc :
        Set C_ij = 0 for pairs separated by more than this distance [Mpc/h].
        None means keep all entries (dense matrix).

    Returns
    -------
    C : (N, N) ndarray, (km/s)²
    """
    if cosmology is None:
        cosmology = _abacussummit_cosmology()

    try:
        C = _flip_covariance(catalog, fsigma8, cosmology)
    except ImportError:
        # Fall back to analytic approximation for unit testing without FLIP
        C = _analytic_covariance(catalog, fsigma8, cosmology)

    if cutoff_mpc is not None:
        C = _apply_cutoff(C, catalog, cutoff_mpc)

    return C


def _abacussummit_cosmology() -> dict:
    """Return AbacusSummit base cosmology parameters."""
    return {
        "H0": 67.36,
        "Omega_m": 0.3152,
        "Omega_b": 0.0493,
        "n_s": 0.9649,
        "sigma8": 0.8111,
    }


def _flip_covariance(
    catalog: dict[str, np.ndarray],
    fsigma8: float,
    cosmology: dict,
) -> np.ndarray:
    """
    Compute covariance matrix using FLIP.

    Raises ImportError if FLIP is not installed.
    """
    import flip  # noqa: F401 — will raise ImportError if missing

    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cos = cosmology
    astropy_cosmo = FlatLambdaCDM(H0=cos["H0"], Om0=cos["Omega_m"])

    # Convert (RA, Dec, z) to comoving Cartesian positions
    ra = catalog["ra"]
    dec = catalog["dec"]
    z = catalog["z_obs"]
    sigma_eta = catalog["sigma_eta"]

    from astropy.coordinates import SkyCoord
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    r_com = astropy_cosmo.comoving_distance(z).to(u.Mpc / u.littleh, equivalencies=u.with_H0(astropy_cosmo.H0)).value

    # FLIP API (subject to FLIP version — verify against installed version)
    cov_model = flip.CovMatrix(
        coords=coords,
        redshifts=z,
        comoving_distances=r_com,
        sigma_v=sigma_eta * 2.998e5 * z,  # approximate velocity error
        power_spectrum="linear",
        cosmology=astropy_cosmo,
        fsigma8=fsigma8,
    )
    return cov_model.matrix  # (N, N) ndarray in (km/s)²


def _analytic_covariance(
    catalog: dict[str, np.ndarray],
    fsigma8: float,
    cosmology: dict,
    sigma_v_fid: float = 300.0,
) -> np.ndarray:
    """
    Analytic approximation to the velocity-velocity covariance.

    Uses a Gaussian velocity correlation function as a stand-in for the
    full power-spectrum integral.  Intended only for unit tests when FLIP
    is not available.

    C_ij = fsigma8² * σ_v² * exp(-r_ij² / (2 * r_corr²))  (i ≠ j)
    C_ii = fsigma8² * σ_v² + σ_noise²

    Parameters
    ----------
    sigma_v_fid :
        Fiducial velocity dispersion at fsigma8=1 [km/s].

    Returns
    -------
    C : (N, N) ndarray, (km/s)²
    """
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    cos = cosmology
    h = cos["H0"] / 100.0
    astropy_cosmo = FlatLambdaCDM(H0=cos["H0"], Om0=cos["Omega_m"])

    ra = catalog["ra"]
    dec = catalog["dec"]
    z = catalog["z_obs"]
    sigma_eta = catalog["sigma_eta"]
    N = len(ra)

    # Comoving positions (Mpc/h)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    r_com = astropy_cosmo.comoving_distance(z).value * h  # Mpc/h

    x = r_com * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = r_com * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z_cart = r_com * np.sin(np.deg2rad(dec))
    pos = np.column_stack([x, y, z_cart])  # (N, 3) Mpc/h

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r_pair = np.linalg.norm(diff, axis=-1)  # (N, N) Mpc/h

    r_corr = 50.0  # Mpc/h — rough correlation length of velocity field
    sigma_sq = (fsigma8 * sigma_v_fid) ** 2

    C = sigma_sq * np.exp(-r_pair**2 / (2 * r_corr**2))

    # Noise diagonal: σ_η → σ_v via σ_v = c * z * σ_η
    c_km_s = 2.998e5
    sigma_v_noise = c_km_s * z * sigma_eta
    np.fill_diagonal(C, C.diagonal() + sigma_v_noise**2)

    return C


def _apply_cutoff(
    C: np.ndarray,
    catalog: dict[str, np.ndarray],
    cutoff_mpc: float,
) -> np.ndarray:
    """
    Zero out off-diagonal entries for pairs separated by more than cutoff_mpc.

    This enforces sparsity for the RG algorithm.

    Parameters
    ----------
    C : (N, N) ndarray
    catalog : dict with 'ra' [deg], 'dec' [deg], 'z_obs'
    cutoff_mpc : separation threshold in Mpc/h

    Returns
    -------
    C : (N, N) ndarray with zeros for distant pairs
    """
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cos = _abacussummit_cosmology()
    h = cos["H0"] / 100.0
    astropy_cosmo = FlatLambdaCDM(H0=cos["H0"], Om0=cos["Omega_m"])

    ra = catalog["ra"]
    dec = catalog["dec"]
    z = catalog["z_obs"]
    r_com = astropy_cosmo.comoving_distance(z).value * h

    x = r_com * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = r_com * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z_cart = r_com * np.sin(np.deg2rad(dec))
    pos = np.column_stack([x, y, z_cart])

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r_pair = np.linalg.norm(diff, axis=-1)

    mask = r_pair > cutoff_mpc
    C_cut = C.copy()
    C_cut[mask] = 0.0
    return C_cut
