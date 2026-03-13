"""
AbacusSummit light cone reader.

Reads halo positions, redshifts, velocities, and masses from AbacusSummit
ASDF light cone files using abacusutils.

Inputs
------
path : str or Path
    Directory containing AbacusSummit light cone ASDF files.
z_max : float
    Maximum redshift cutoff (default 0.8).

Outputs
-------
dict with arrays:
    ra          [deg]  Right ascension
    dec         [deg]  Declination
    z_obs       []     Observed redshift (peculiar velocity included)
    v_r         [km/s] True radial peculiar velocity
    mass        [Msun/h] Halo mass
    pos         [Mpc/h] Comoving (x, y, z) positions

Units
-----
Distances in Mpc/h, velocities in km/s.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def load_lightcone(
    path: str | Path,
    z_max: float = 0.8,
    fields: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load AbacusSummit light cone halos using abacusutils.

    Parameters
    ----------
    path :
        Directory with AbacusSummit light cone ASDF files.
    z_max :
        Discard halos above this redshift.
    fields :
        ASDF fields to read; default is a minimal useful set.

    Returns
    -------
    dict
        Arrays keyed by field name.  Always includes 'ra', 'dec',
        'z_obs', 'v_r', 'mass', 'pos'.
    """
    try:
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    except ImportError as e:
        raise ImportError(
            "abacusutils is required to read AbacusSummit files. "
            "Install with: pip install abacusutils"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Light cone path does not exist: {path}")

    default_fields = ["pos", "vel", "N", "npstartA"]
    fields = fields or default_fields

    cat = CompaSOHaloCatalog(str(path), fields=fields, cleaned=False)
    halos = cat.halos

    # Positions in Mpc/h, velocities in km/s
    pos = np.array(halos["pos"])   # shape (N, 3)
    vel = np.array(halos["vel"])   # shape (N, 3)
    mass_counts = np.array(halos["N"])  # particle count

    particle_mass = cat.header.get("ParticleMassHMsun", 2e9)  # Msun/h per particle
    mass = mass_counts * particle_mass

    ra, dec, z_obs, v_r = _cartesian_to_radecz(pos, vel, cat.header)

    mask = z_obs <= z_max
    return {
        "ra": ra[mask],
        "dec": dec[mask],
        "z_obs": z_obs[mask],
        "v_r": v_r[mask],
        "mass": mass[mask],
        "pos": pos[mask],
    }


def _cartesian_to_radecz(
    pos: np.ndarray,
    vel: np.ndarray,
    header: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert comoving Cartesian positions to (RA, Dec, z_obs, v_r).

    The observer is assumed to be at the origin.  The Hubble flow
    redshift is computed from the comoving distance using the
    cosmological parameters in the AbacusSummit header.

    Parameters
    ----------
    pos : (N, 3) array, Mpc/h
    vel : (N, 3) array, km/s
    header : AbacusSummit header dict (must contain 'H0', 'Omega_M')

    Returns
    -------
    ra, dec : degrees
    z_obs   : observed redshift (Hubble + peculiar)
    v_r     : radial peculiar velocity, km/s (positive = receding)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    H0 = header.get("H0", 67.36)
    omega_m = header.get("Omega_M", 0.3152)
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

    # Comoving distance in Mpc/h → Mpc
    h = H0 / 100.0
    r_com = np.linalg.norm(pos, axis=1) / h  # Mpc

    # Hubble flow redshift from comoving distance (vectorised approximation)
    # For z < 0.8 the linear approximation z ≈ H0*r/c is ~1% accurate;
    # use astropy for correctness.
    from astropy.coordinates import Distance
    dist_obj = Distance(r_com, unit=u.Mpc)
    z_hubble = dist_obj.compute_z(cosmology=cosmo)

    # Radial unit vector
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)  # (N, 3)
    v_r = np.einsum("ij,ij->i", vel, r_hat)  # km/s

    # Observed redshift: z_obs ≈ z_hubble + v_r/c (first-order)
    c_km_s = 2.998e5
    z_obs = z_hubble + v_r / c_km_s * (1 + z_hubble)

    # RA, Dec
    coords = SkyCoord(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        unit="Mpc", representation_type="cartesian",
    ).icrs
    ra = coords.ra.deg
    dec = coords.dec.deg

    return ra, dec, z_obs, v_r
