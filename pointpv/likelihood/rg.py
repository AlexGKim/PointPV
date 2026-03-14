"""
McDonald (2019) Renormalization Group log-likelihood.

Evaluates the same Gaussian log-likelihood as mlf.py, but using the RG
coarse-graining algorithm of McDonald (2019, PhysRevD.100.043511).

At each level of the hierarchy, pairs of adjacent galaxies are merged by
integrating out their "difference mode."  Each merge step contributes a
scalar to log|det C| and updates the covariance of the surviving "sum mode."
The total work is O(N log N) (or O(N) with a spatial cutoff on correlations).

Inputs
------
u : (N,) ndarray, km/s
    Observed peculiar velocities.
C : (N, N) ndarray, (km/s)²
    Full covariance matrix (or a sparse approximation).
positions : (N, 3) ndarray, Mpc/h
    Comoving positions, used to build the KD-tree pairing.

Outputs
-------
logL : float
    -½ [log|C| + uᵀ C⁻¹ u]  (matches mlf.log_likelihood to machine precision
    for dense matrices; small errors possible with sparsity cutoff).

Reference
---------
McDonald (2019), PhysRevD.100.043511, §III.
"""

from __future__ import annotations

import time

import numpy as np

from pointpv.rg.tree import build_tree, RGTree
from pointpv.rg.coarsen import rg_coarsen_all


def log_likelihood(
    u: np.ndarray,
    C: np.ndarray,
    positions: np.ndarray | None = None,
    max_leaf_size: int = 2,
    verbose: bool = False,
    schur_tol: float = 0.0,
    tree: RGTree | None = None,
) -> float:
    """
    McDonald RG log-likelihood.

    Parameters
    ----------
    u : (N,) array, km/s
    C : (N, N) array, (km/s)²
    positions : (N, 3) array, Mpc/h, optional
        Used to build the hierarchical pairing tree.  Required when
        ``tree`` is not supplied.
    max_leaf_size :
        Stop recursing when a node has ≤ this many members; evaluate
        remaining small blocks with direct Cholesky.
    verbose :
        Print per-level timing.
    tree : RGTree, optional
        Pre-built pairing tree.  When provided, ``build_tree`` is skipped,
        avoiding redundant work in repeated evaluations (e.g. fsigma8 scans).

    Returns
    -------
    logL : float
    """
    if tree is None and positions is None:
        raise ValueError("Either positions or tree must be supplied.")

    t0 = time.perf_counter()

    # Build the KD-tree pairing (or reuse the supplied one)
    if tree is None:
        tree = build_tree(positions)
    if verbose:
        print(f"  [RG] tree built in {time.perf_counter()-t0:.3f}s, depth={tree.depth}")

    # Run the RG coarse-graining
    logL = rg_coarsen_all(u, C, tree, verbose=verbose, schur_tol=schur_tol)

    if verbose:
        print(f"  [RG] total wall time: {time.perf_counter()-t0:.3f}s")

    return logL


def scan_fsigma8(
    u: np.ndarray,
    catalog: dict[str, np.ndarray],
    positions: np.ndarray,
    fsigma8_values: np.ndarray | None = None,
    cosmology: dict | None = None,
    schur_tol: float = 0.0,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan log-likelihood over a grid of fsigma8 values using the RG method.

    Parameters
    ----------
    u : (N,) array, km/s
    catalog : dict
        Catalog dict (passed to covariance builder).
    positions : (N, 3) array, Mpc/h
        Comoving positions for KD-tree construction.
    fsigma8_values : (M,) array, optional
        Default: 20 points in [0.2, 0.8].
    cosmology : dict, optional
    verbose : bool

    Returns
    -------
    dict
        'fsigma8', 'logL', 'time_per_eval', 'tree_build_time'

        ``tree_build_time`` is the one-off cost of building the KD-tree pairing
        (paid once, not included in each entry of ``time_per_eval``).
    """
    from pointpv.covariance.velocity import build_covariance

    if fsigma8_values is None:
        fsigma8_values = np.linspace(0.2, 0.8, 20)

    # Build the pairing tree once — positions don't change between evaluations.
    t_tree0 = time.perf_counter()
    shared_tree = build_tree(positions)
    tree_build_time = time.perf_counter() - t_tree0
    if verbose:
        print(f"  [RG] tree built in {tree_build_time:.4f}s (shared across all evaluations)")

    logL_values = np.empty(len(fsigma8_values))
    times = []

    for i, fs8 in enumerate(fsigma8_values):
        t0 = time.perf_counter()
        C = build_covariance(catalog, fs8, cosmology=cosmology)
        logL_values[i] = log_likelihood(u, C, tree=shared_tree, verbose=False, schur_tol=schur_tol)
        dt = time.perf_counter() - t0
        times.append(dt)
        if verbose:
            print(f"  fsigma8={fs8:.3f}  logL={logL_values[i]:.4f}  t={dt:.3f}s")

    return {
        "fsigma8": fsigma8_values,
        "logL": logL_values,
        "time_per_eval": np.array(times),
        "tree_build_time": tree_build_time,
    }
