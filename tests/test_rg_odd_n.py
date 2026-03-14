"""
Odd-N correctness tests: exercises singleton pass-through in the RG tree.

When N is odd, one node is unpaired at each level where the active count is odd.
The singleton is passed through unchanged.  The final log-likelihood must still
match Cholesky to within 1e-6.
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


@pytest.mark.parametrize("n,seed", [(3, 20), (7, 21), (11, 22), (33, 23), (101, 24)])
def test_rg_matches_cholesky_odd_n(n: int, seed: int) -> None:
    u, C, pos = _make_synthetic_problem(n, seed=seed)

    logL_mlf = mlf_logL(u, C)
    logL_rg = rg_logL(u, C, pos, verbose=False)

    diff = abs(logL_rg - logL_mlf)
    tol = 1e-6
    assert diff < tol, (
        f"N={n} seed={seed}: RG={logL_rg:.10f}  MLF={logL_mlf:.10f}  "
        f"diff={diff:.2e}  (tol={tol:.0e})"
    )
