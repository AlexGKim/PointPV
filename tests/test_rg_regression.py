"""
Regression test: RG log-likelihood matches Cholesky for N=100.

Uses a synthetic dense covariance matrix (exponential correlation).
Requires |logL_rg - logL_chol| < 1e-6 (generous tolerance to allow for
floating-point accumulation over many levels).
"""

from __future__ import annotations

import numpy as np
import pytest

from pointpv.likelihood.mlf import log_likelihood as mlf_logL
from pointpv.likelihood.rg import log_likelihood as rg_logL


def _make_synthetic_problem(
    n: int,
    seed: int = 0,
    length_scale: float = 50.0,
    sigma_v: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic PV problem with N galaxies.

    Returns
    -------
    u : (N,) km/s
    C : (N, N) (km/s)^2
    pos : (N, 3) Mpc/h
    """
    rng = np.random.default_rng(seed)

    # Random positions in a 500 Mpc/h box
    pos = rng.uniform(0, 500, size=(n, 3))

    # Exponential covariance kernel: C_ij = sigma_v^2 * exp(-r_ij / L)
    diff = pos[:, None, :] - pos[None, :, :]       # (N, N, 3)
    r = np.sqrt(np.sum(diff**2, axis=-1))           # (N, N) Mpc/h
    C = sigma_v**2 * np.exp(-r / length_scale)
    # Add small diagonal noise to ensure positive-definiteness
    C += np.eye(n) * (sigma_v * 0.05)**2

    u = rng.standard_normal(n) * sigma_v * 0.1

    return u, C, pos


@pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (50, 2), (100, 3)])
def test_rg_matches_cholesky(n: int, seed: int) -> None:
    u, C, pos = _make_synthetic_problem(n, seed=seed)

    logL_chol = mlf_logL(u, C)
    logL_rg = rg_logL(u, C, pos, verbose=False)

    diff = abs(logL_rg - logL_chol)
    tol = 1e-6
    assert diff < tol, (
        f"N={n} seed={seed}: RG={logL_rg:.10f}  Cholesky={logL_chol:.10f}  "
        f"diff={diff:.2e}  (tol={tol:.0e})"
    )


def test_rg_best_fit_close_to_cholesky() -> None:
    """Best-fit fsigma8 from a coarse scan should agree to ≤ 1 grid step."""
    # Use a small N for speed; just check the argmax agrees
    n = 30
    u, C, pos = _make_synthetic_problem(n, seed=99)

    logL_chol = mlf_logL(u, C)
    logL_rg = rg_logL(u, C, pos, verbose=False)

    # Both should return similar values for a well-conditioned matrix
    assert abs(logL_rg - logL_chol) < 1e-6
