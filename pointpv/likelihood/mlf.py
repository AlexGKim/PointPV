"""
Baseline Maximum Likelihood Function (MLF) log-likelihood.

Evaluates the Gaussian log-likelihood via dense Cholesky decomposition:

    L(θ) = -½ [log|C(θ)| + uᵀ C(θ)⁻¹ u]

where u is the N-vector of peculiar velocity measurements and C(θ) is the
N×N velocity-velocity covariance matrix parameterised by fsigma8.

On a laptop (POINTPV_BACKEND=scipy or None): uses scipy.linalg.cho_factor.
On Perlmutter  (POINTPV_BACKEND=cupy):          uses cupy.linalg.cholesky.

Inputs
------
u : (N,) ndarray, km/s
    Observed peculiar velocities (or log-distance ratios, appropriately scaled).
C : (N, N) ndarray, (km/s)²
    Full covariance matrix.

Outputs
-------
logL : float
    Log-likelihood (no additive constant: omits -N/2 * log(2π)).

Units
-----
Velocities in km/s; covariance in (km/s)².
"""

from __future__ import annotations

import os
import time

import numpy as np


def log_likelihood(u: np.ndarray, C: np.ndarray) -> float:
    """
    Evaluate the Gaussian log-likelihood via Cholesky decomposition.

    Parameters
    ----------
    u : (N,) array, km/s
    C : (N, N) array, (km/s)²

    Returns
    -------
    logL : float
        -½ [log|C| + uᵀ C⁻¹ u]
    """
    backend = os.environ.get("POINTPV_BACKEND", "scipy").lower()

    if backend == "cupy":
        return _cholesky_cupy(u, C)
    else:
        return _cholesky_scipy(u, C)


def _cholesky_scipy(u: np.ndarray, C: np.ndarray) -> float:
    """CPU Cholesky log-likelihood via scipy."""
    from scipy.linalg import cho_factor, cho_solve

    L_fac, lower = cho_factor(C, lower=True)

    # log|C| = 2 * Σ log(diag(L))
    log_det = 2.0 * np.sum(np.log(np.diag(L_fac)))

    # u^T C^{-1} u
    alpha = cho_solve((L_fac, lower), u)
    quad = float(np.dot(u, alpha))

    return -0.5 * (log_det + quad)


def _cholesky_cupy(u: np.ndarray, C: np.ndarray) -> float:
    """GPU Cholesky log-likelihood via CuPy."""
    import cupy as cp
    from cupy.linalg import cholesky
    from cupyx.scipy.linalg import solve_triangular

    C_gpu = cp.asarray(C, dtype=cp.float64)
    u_gpu = cp.asarray(u, dtype=cp.float64)

    L = cholesky(C_gpu)

    # log|C| = 2 * Σ log(diag(L))
    log_det = float(2.0 * cp.sum(cp.log(cp.diag(L))))

    # Solve L z = u, then u^T C^{-1} u = z^T z
    z = solve_triangular(L, u_gpu, lower=True)
    quad = float(cp.dot(z, z))

    return -0.5 * (log_det + quad)


def scan_fsigma8(
    u: np.ndarray,
    catalog: dict[str, np.ndarray],
    fsigma8_values: np.ndarray | None = None,
    cosmology: dict | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan log-likelihood over a grid of fsigma8 values.

    Parameters
    ----------
    u : (N,) array
        Observed peculiar velocities [km/s].
    catalog : dict
        Catalog dict (passed to covariance builder).
    fsigma8_values : (M,) array
        Grid of fsigma8 to evaluate.  Default: 20 points in [0.2, 0.8].
    cosmology : dict, optional
        Background cosmology.
    verbose : bool
        Print timing info.

    Returns
    -------
    dict
        'fsigma8': grid values
        'logL': log-likelihood at each grid point
        'time_per_eval': wall-clock seconds per evaluation
    """
    from pointpv.covariance.velocity import build_covariance

    if fsigma8_values is None:
        fsigma8_values = np.linspace(0.2, 0.8, 20)

    logL_values = np.empty(len(fsigma8_values))
    times = []

    for i, fs8 in enumerate(fsigma8_values):
        t0 = time.perf_counter()
        C = build_covariance(catalog, fs8, cosmology=cosmology)
        logL_values[i] = log_likelihood(u, C)
        dt = time.perf_counter() - t0
        times.append(dt)
        if verbose:
            print(f"  fsigma8={fs8:.3f}  logL={logL_values[i]:.4f}  t={dt:.3f}s")

    return {
        "fsigma8": fsigma8_values,
        "logL": logL_values,
        "time_per_eval": np.array(times),
    }
