"""
Unit test: rg_step_n2 matches scipy Cholesky for N=2.

For N=2, one exact RG step should reproduce the Gaussian log-likelihood
to machine precision.
"""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from pointpv.rg.coarsen import rg_step_n2
from pointpv.likelihood.mlf import log_likelihood


def _scipy_logL(u: np.ndarray, C: np.ndarray) -> float:
    L_fac, lower = cho_factor(C, lower=True)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L_fac))))
    alpha = cho_solve((L_fac, lower), u)
    quad = float(np.dot(u, alpha))
    return -0.5 * (log_det + quad)


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_rg_step_n2_matches_cholesky(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # Random 2x2 SPD matrix
    A = rng.standard_normal((2, 2))
    C = A @ A.T + np.eye(2) * 0.5  # guaranteed SPD

    u = rng.standard_normal(2) * 100.0  # km/s

    logL_rg = rg_step_n2(u, C)
    logL_chol = _scipy_logL(u, C)

    assert abs(logL_rg - logL_chol) < 1e-10, (
        f"seed={seed}: RG={logL_rg:.12f}  Cholesky={logL_chol:.12f}  "
        f"diff={abs(logL_rg - logL_chol):.2e}"
    )


def test_rg_step_n2_via_mlf() -> None:
    """Also check against the mlf.log_likelihood wrapper."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((2, 2))
    C = A @ A.T + np.eye(2) * 1.0
    u = rng.standard_normal(2) * 50.0

    logL_rg = rg_step_n2(u, C)
    logL_mlf = log_likelihood(u, C)

    assert abs(logL_rg - logL_mlf) < 1e-10, (
        f"rg_step_n2={logL_rg:.12f}  mlf={logL_mlf:.12f}  "
        f"diff={abs(logL_rg - logL_mlf):.2e}"
    )
