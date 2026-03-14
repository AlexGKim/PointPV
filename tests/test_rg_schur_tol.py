"""
schur_tol accuracy tradeoff tests.

Verifies that the Schur sparsity cutoff introduces only a controlled error
in the log-likelihood relative to the exact (schur_tol=0) result.

Per CLAUDE.md: schur_tol=1.0 gives |ΔlogL| ~ 1e-6 for a physical covariance.
Here we use a synthetic exponential covariance and require |ΔlogL| < 1e-5
at schur_tol=1.0 to give a modest safety margin.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def exact_logL_n200() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    n = 200
    u, C, pos = _make_synthetic_problem(n, seed=30)
    logL_exact = rg_logL(u, C, pos, schur_tol=0.0)
    return logL_exact, u, C, pos


@pytest.mark.parametrize(
    "schur_tol,max_err",
    [
        (0.1, 1e-4),
        (0.5, 5e-4),
        (1.0, 1e-5),   # CLAUDE.md claim: ~1e-6 for physical cov; use 1e-5 margin
        (2.0, 1e-3),
        (5.0, 5e-3),
    ],
)
def test_schur_tol_accuracy(
    exact_logL_n200: tuple,
    schur_tol: float,
    max_err: float,
) -> None:
    logL_exact, u, C, pos = exact_logL_n200
    logL_approx = rg_logL(u, C, pos, schur_tol=schur_tol)
    err = abs(logL_approx - logL_exact)
    assert err < max_err, (
        f"schur_tol={schur_tol}: |ΔlogL|={err:.2e}  (limit={max_err:.0e})"
    )


def test_schur_tol_1_tight(exact_logL_n200: tuple) -> None:
    """schur_tol=1.0 should achieve |ΔlogL| < 1e-5 (CLAUDE.md claim)."""
    logL_exact, u, C, pos = exact_logL_n200
    logL_approx = rg_logL(u, C, pos, schur_tol=1.0)
    err = abs(logL_approx - logL_exact)
    assert err < 1e-5, f"|ΔlogL|={err:.2e} exceeds 1e-5 at schur_tol=1.0"


def test_schur_tol_zero_is_exact(exact_logL_n200: tuple) -> None:
    """schur_tol=0 (default) must be bit-identical between calls."""
    logL_exact, u, C, pos = exact_logL_n200
    logL_repeat = rg_logL(u, C, pos, schur_tol=0.0)
    assert logL_exact == logL_repeat, "schur_tol=0 is not deterministic"
