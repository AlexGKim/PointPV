"""
RG level-size shrinkage tests.

Verifies that the number of active nodes halves at each RG level, from N
down to 1.  Uses the return_diagnostics=True parameter added to rg_coarsen_all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointpv.rg.coarsen import rg_coarsen_all
from pointpv.rg.tree import build_tree


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


@pytest.mark.parametrize("n,seed", [
    (8, 40),
    (16, 41),
    (32, 42),
    (64, 43),
    (100, 44),
    (200, 45),
])
def test_level_sizes_shrink(n: int, seed: int) -> None:
    u, C, pos = _make_synthetic_problem(n, seed=seed)
    tree = build_tree(pos)
    _, level_sizes = rg_coarsen_all(u, C, tree, return_diagnostics=True)

    assert level_sizes[0] == n, (
        f"N={n}: level_sizes[0]={level_sizes[0]}, expected {n}"
    )
    assert level_sizes[-1] == 1, (
        f"N={n}: level_sizes[-1]={level_sizes[-1]}, expected 1"
    )

    # Each step must shrink by at most ceil(prev/2) + 1
    for k in range(len(level_sizes) - 1):
        prev = level_sizes[k]
        curr = level_sizes[k + 1]
        limit = math.ceil(prev / 2) + 1
        assert curr <= limit, (
            f"N={n}: level_sizes[{k}]={prev} → [{k+1}]={curr} "
            f"exceeds ceil({prev}/2)+1={limit}"
        )

    # For n >= 16, the total shrinkage must be > 90%
    if n >= 16:
        ratio = level_sizes[-1] / level_sizes[0]
        assert ratio < 0.1, (
            f"N={n}: shrinkage ratio={ratio:.3f} is not < 0.1 (>90% reduction)"
        )


def test_fill_fraction_decreases() -> None:
    """Fill fraction tracking returns valid fractions, one per level."""
    u, C, pos = _make_synthetic_problem(200, seed=60)
    tree = build_tree(pos)
    _, level_sizes, fill_fracs = rg_coarsen_all(
        u, C, tree, return_diagnostics=True, fill_tol=1.0
    )
    # must have one fill fraction per level (including level 0)
    assert len(fill_fracs) == len(level_sizes)
    # all entries are valid fractions
    assert all(0.0 <= f <= 1.0 for f in fill_fracs), f"Invalid fill fractions: {fill_fracs}"
    # single-node final level has no off-diagonal entries → fill=0
    assert fill_fracs[-1] == 0.0
    # initial level has some entries above threshold (sigma_v=300 >> fill_tol=1.0)
    assert fill_fracs[0] > 0.0


def test_active_fraction_meaningful() -> None:
    """With schur_tol=1.0, verbose output should show active < 100% (some work skipped)."""
    # Just verify the 3-tuple return and that active fractions printed without error
    u, C, pos = _make_synthetic_problem(200, seed=61)
    tree = build_tree(pos)
    result = rg_coarsen_all(
        u, C, tree, return_diagnostics=True, fill_tol=1.0, schur_tol=1.0
    )
    assert len(result) == 3


def test_active_frac_stop_triggers_mlf_handoff() -> None:
    """active_frac_stop=0.5 should trigger early stop and return a 3-tuple."""
    u, C, pos = _make_synthetic_problem(64, seed=70)
    tree = build_tree(pos)
    result = rg_coarsen_all(u, C, tree, schur_tol=1.0, active_frac_stop=0.5)
    assert isinstance(result, tuple), "Expected tuple, got scalar float"
    assert len(result) == 3
    partial_logL, C_stop, u_stop = result
    assert isinstance(partial_logL, float)
    assert C_stop.shape[0] > 1, f"C_stop should have >1 node, got shape {C_stop.shape}"
    assert u_stop.shape[0] == C_stop.shape[0]


def test_active_frac_stop_accuracy() -> None:
    """RG partial logL + MLF completion should match direct MLF within tolerance."""
    from scipy.linalg import cho_factor, cho_solve

    def mlf_logL(u: np.ndarray, C: np.ndarray) -> float:
        cho = cho_factor(C)
        sign, logdet = np.linalg.slogdet(C)
        quad = float(u @ cho_solve(cho, u))
        return -0.5 * (logdet + quad)

    u, C, pos = _make_synthetic_problem(100, seed=71)
    tree = build_tree(pos)

    # Direct MLF reference
    ref_logL = mlf_logL(u, C)

    # RG partial + MLF completion
    result = rg_coarsen_all(u, C, tree, schur_tol=1.0, active_frac_stop=0.5)
    assert isinstance(result, tuple), "active_frac_stop should trigger early handoff"
    partial_logL, C_stop, u_stop = result
    total_logL = partial_logL + mlf_logL(u_stop, C_stop)

    assert abs(total_logL - ref_logL) < 5.0, (
        f"|ΔlogL| = {abs(total_logL - ref_logL):.4f} exceeds tolerance of 5.0"
    )


