"""
End-to-end test using the real FLIP/CAMB covariance pipeline.

Skipped automatically if camb or flip are not importable.

Verifies:
  1. The covariance matrix is symmetric positive-definite.
  2. RG and MLF log-likelihoods agree to 1e-6 at fsigma8=0.47.
  3. Best-fit fsigma8 (coarse 10-point scan) agrees within one grid step.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Skip entire module if camb / flip are not installed
pytest.importorskip("camb", reason="camb not installed")

try:
    sys.path.insert(0, "/Users/akim/Projects/flip")
    from flip.covariance.covariance import CovMatrix  # noqa: F401
    _HAS_FLIP = True
except ImportError:
    _HAS_FLIP = False

pytestmark = [
    pytest.mark.flip,
    pytest.mark.skipif(not _HAS_FLIP, reason="FLIP not available"),
]

# Deferred imports so the skip above fires before any ImportError
from pointpv.covariance.velocity import build_covariance  # noqa: E402
from pointpv.likelihood.mlf import log_likelihood as mlf_logL  # noqa: E402
from pointpv.likelihood.rg import log_likelihood as rg_logL  # noqa: E402


def _make_catalog(n: int = 50, seed: int = 99) -> dict:
    """Build a small synthetic catalog for covariance tests."""
    sys.path.insert(0, str(__file__).rsplit("/tests", 1)[0])
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from generate_mock import generate_synthetic_catalog
    return generate_synthetic_catalog(n=n, seed=seed, use_mag_limit=False)


@pytest.fixture(scope="module")
def flip_covariance():
    catalog = _make_catalog(n=50)
    fs8 = 0.47
    C = build_covariance(catalog, fsigma8=fs8)
    return catalog, C


def test_covariance_spd(flip_covariance) -> None:
    """Covariance matrix must be symmetric positive-definite."""
    _, C = flip_covariance
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.min() > 0, f"Covariance has non-positive eigenvalue: {eigvals.min():.3e}"


def test_rg_matches_mlf_flip(flip_covariance) -> None:
    """RG and MLF log-likelihoods must agree to 1e-6 with FLIP covariance."""
    catalog, C = flip_covariance
    rng = np.random.default_rng(101)
    u = rng.standard_normal(len(catalog["ra"])) * 300.0

    pos = catalog["pos"]

    logL_mlf = mlf_logL(u, C)
    logL_rg = rg_logL(u, C, pos, verbose=False)

    diff = abs(logL_rg - logL_mlf)
    tol = 1e-6
    assert diff < tol, (
        f"FLIP cov: RG={logL_rg:.10f}  MLF={logL_mlf:.10f}  diff={diff:.2e}"
    )


def test_best_fit_fsigma8_agrees(flip_covariance) -> None:
    """Best-fit fsigma8 from a 10-point scan agrees within one grid step."""
    catalog, _ = flip_covariance
    rng = np.random.default_rng(102)
    u = rng.standard_normal(len(catalog["ra"])) * 300.0
    pos = catalog["pos"]

    fs8_grid = np.linspace(0.3, 0.7, 10)
    step = fs8_grid[1] - fs8_grid[0]

    logL_mlf = np.array([mlf_logL(u, build_covariance(catalog, fs8)) for fs8 in fs8_grid])
    logL_rg = np.array([rg_logL(u, build_covariance(catalog, fs8), pos) for fs8 in fs8_grid])

    fs8_best_mlf = fs8_grid[np.argmax(logL_mlf)]
    fs8_best_rg = fs8_grid[np.argmax(logL_rg)]

    assert abs(fs8_best_rg - fs8_best_mlf) <= step + 1e-9, (
        f"Best-fit fsigma8: MLF={fs8_best_mlf:.3f}  RG={fs8_best_rg:.3f}  "
        f"diff={abs(fs8_best_rg-fs8_best_mlf):.3f} > step={step:.3f}"
    )
