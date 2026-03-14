"""
Large-N correctness tests: RG log-likelihood must match Cholesky for N up to 1000.

These are marked slow because N=1000 takes ~1s per evaluation.
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
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 500, size=(n, 3))
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    C = sigma_v**2 * np.exp(-r / length_scale)
    C += np.eye(n) * (sigma_v * 0.05) ** 2
    u = rng.standard_normal(n) * sigma_v * 0.1
    return u, C, pos


@pytest.mark.slow
@pytest.mark.parametrize("n,seed", [(200, 10), (500, 11), (1000, 12)])
def test_rg_matches_cholesky_large_n(n: int, seed: int) -> None:
    u, C, pos = _make_synthetic_problem(n, seed=seed)

    logL_mlf = mlf_logL(u, C)
    logL_rg = rg_logL(u, C, pos, verbose=False)

    diff = abs(logL_rg - logL_mlf)
    tol = 1e-6
    assert diff < tol, (
        f"N={n} seed={seed}: RG={logL_rg:.10f}  MLF={logL_mlf:.10f}  "
        f"diff={diff:.2e}  (tol={tol:.0e})"
    )
