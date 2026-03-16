"""
McDonald RG coarse-graining step.

At each level of the RG hierarchy, pairs of adjacent nodes (i, j) are
merged by integrating out their "difference mode" d = x_i - x_j.

For a Gaussian model with current sum-mode mean vectors m_i, m_j and
covariance sub-blocks C_ii, C_ij, C_jj, the contribution of the
difference mode to the log-likelihood is:

    ΔlogL_ij = -½ [log|C_ii + C_jj - 2*C_ij| + d^T (C_ii+C_jj-2C_ij)^{-1} d]

where d = u_i - u_j is the observed difference.

The surviving sum mode s = x_i + x_j has:
    u_s       = u_i + u_j
    C_ss_new  = C_ii + C_jj + 2*C_ij
                  - (C_ii - C_ij) (C_ii + C_jj - 2*C_ij)^{-1} (C_ii - C_ij)^T

(See McDonald 2019, PhysRevD.100.043511, eq. 9–12, adapted for per-pair
scalar/block sub-matrices.)

For the N=2 special case (one pair of scalars), the full log-likelihood
is recovered exactly in a single step.

Inputs
------
u : (N,) ndarray, km/s
    Observed peculiar velocities at the current level.
C : (N, N) ndarray, (km/s)²
    Current inter-node covariance matrix.
tree : RGTree
    Hierarchical pairing structure.

Outputs
-------
logL : float
    -½ [log|C| + uᵀ C⁻¹ u]

Units
-----
Velocities in km/s; covariance in (km/s)².
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from pointpv.rg.tree import RGTree, RGNode
from pointpv.rg.sparse_ops import get_solver, dense_to_sparse

_SCHUR_BROADCAST_THRESHOLD = 0.5  # use broadcast when active fraction > this


def rg_coarsen_all(
    u: np.ndarray,
    C: np.ndarray,
    tree: RGTree,
    verbose: bool = False,
    schur_tol: float = 0.0,
    return_diagnostics: bool = False,
    fill_tol: float = 0.0,
    sparse_tol: float = 0.0,
    stop_size: int = 1,
) -> "float | tuple":
    """
    Run all RG coarsening levels and return the log-likelihood.

    Parameters
    ----------
    u : (N,) array, km/s
        Observed velocities at the leaf level.
    C : (N, N) array, (km/s)²
        Full covariance matrix at the leaf level.
    tree : RGTree
        Hierarchical pairing tree from rg.tree.build_tree.
    verbose : bool
        Print per-level timing.
    schur_tol : float
        Skip rank-1 Schur update for rows where |diff_col[i]| ≤ schur_tol.
    return_diagnostics : bool
        If True, return (logL, level_sizes) where level_sizes[k] is the
        number of active nodes at the start of level k (before merging).
        level_sizes[0] == N and level_sizes[-1] == 1.
        When fill_tol > 0, returns a 3-tuple (logL, level_sizes, fill_fractions).
    fill_tol : float
        When > 0, compute fill fraction (fraction of off-diagonal entries with
        |C[i,j]| >= fill_tol) at the start of each level.  Appended to the
        3-tuple return when return_diagnostics=True.
    sparse_tol : float
        When > 0, store C_cur as scipy.sparse.csc_matrix and zero out
        off-diagonal entries with |C[i,j]| < sparse_tol after each level.
        Requires schur_tol > 0 (the full dense Schur update with schur_tol=0
        would negate the sparsity).  Together they give O(k²) work per pair
        where k = nnz per column instead of O(N²).
    stop_size : int
        Stop compression early when the active node count drops to <= stop_size
        and return the compressed (partial_logL, C_stop, u_stop) for the caller
        to finish with MLF (Cholesky).  Default 1 = full compression (existing
        behaviour, returns scalar logL).  stop_size >= N means do no RG at all.

    Returns
    -------
    logL : float
        When stop_size==1 and return_diagnostics=False (default).
    (logL, level_sizes) : (float, list[int])
        When stop_size==1 and return_diagnostics=True and fill_tol==0.
    (logL, level_sizes, fill_fractions) : (float, list[int], list[float])
        When stop_size==1 and return_diagnostics=True and fill_tol>0.
    (partial_logL, C_stop, u_stop) : (float, ndarray, ndarray)
        When stop_size > 1 and return_diagnostics=False.  The caller should add
        mlf.log_likelihood(u_stop, C_stop) to get the full log-likelihood.
    (partial_logL, C_stop, u_stop, level_sizes[, fill_fractions]) : tuple
        When stop_size > 1 and return_diagnostics=True.
    """
    if sparse_tol > 0.0 and schur_tol == 0.0:
        raise ValueError(
            "sparse_tol requires schur_tol > 0. "
            "The full Schur update (schur_tol=0) is dense and would defeat sparsity."
        )

    logL_acc = 0.0

    level_nodes = tree.levels[0]
    u_cur = u.copy()
    C_cur: "np.ndarray | sp.csc_matrix" = C.copy()

    solver = get_solver() if sparse_tol > 0.0 else None

    if sparse_tol > 0.0:
        C_cur = dense_to_sparse(C_cur).tocsc()

    level_sizes: list[int] = [len(u_cur)]
    fill_fractions: list[float] = []

    # Compute fill fraction at level 0 (before any coarsening)
    if fill_tol > 0.0:
        C_dense0 = C_cur.toarray() if sp.issparse(C_cur) else C_cur
        mask0 = ~np.eye(len(u_cur), dtype=bool)
        fill_fractions.append(float(np.mean(np.abs(C_dense0[mask0]) >= fill_tol)))

    # Map from node object id to current local array index
    node_to_local: dict[int, int] = {id(n): i for i, n in enumerate(level_nodes)}

    for level_idx in range(1, tree.depth + 1):
        # Hybrid mode: stop early and return compressed state for MLF handoff
        if stop_size > 1 and len(u_cur) <= stop_size:
            break

        t0 = time.perf_counter() if (verbose or fill_tol > 0.0) else 0.0

        # fill fraction at start of this level = entry computed at end of previous level
        fill = fill_fractions[level_idx - 1] if fill_tol > 0.0 else 0.0

        next_nodes = tree.levels[level_idx]

        # Collect pairs and singletons for this level
        pair_updates: list[tuple[int, int, RGNode]] = []
        singleton_updates: list[tuple[int, RGNode]] = []
        for node in next_nodes:
            if len(node.children) == 2:
                ci, cj = node.children
                pair_updates.append((node_to_local[id(ci)], node_to_local[id(cj)], node))
            else:
                ci = node.children[0]
                singleton_updates.append((node_to_local[id(ci)], node))

        # Process each pair: integrate out difference mode, update u_cur and C_cur in-place.
        # Processing one pair at a time ensures cross-pair Schur corrections are applied.
        # We accumulate the pair's logL contribution and replace (li, lj) rows/cols with
        # the new sum-mode row, then drop the lj row at the end.
        #
        # To avoid index-shifting during deletion we collect deletions and apply at the end.
        cols_to_delete: list[int] = []
        new_local: dict[int, int] = {}
        level_active: list[float] = []

        for li, lj, node in pair_updates:
            c_ii = float(C_cur[li, li])
            c_jj = float(C_cur[lj, lj])
            c_ij = float(C_cur[li, lj])
            c_dd = max(c_ii + c_jj - 2 * c_ij, 1e-12)
            d = u_cur[li] - u_cur[lj]
            logL_acc += -0.5 * (np.log(abs(c_dd)) + d**2 / c_dd) + np.log(2.0)

            if sp.issparse(C_cur):
                # Fully sparse path: no densification, sparse rank-1 outer product.
                # diff_sp is (N,1); schur_tol threshold applied to stored values only.
                diff_sp = (C_cur.getcol(li) - C_cur.getcol(lj)).tocsr()
                diff_sp.data[np.abs(diff_sp.data) <= schur_tol] = 0
                diff_sp.eliminate_zeros()
                if fill_tol > 0.0:
                    level_active.append(diff_sp.nnz / len(u_cur))
                if diff_sp.nnz > 0:
                    # Sparse rank-1 Schur correction: (N,1) @ (1,N) → (N,N) sparse
                    diff_col_sp = diff_sp.tocsc()
                    correction = (diff_col_sp @ diff_col_sp.T / c_dd).tocsc()
                    C_cur = (C_cur - correction).tocsc()
                    # u update via backend-agnostic SpMV: diff_col_sp (N,1) @ [d/c_dd]
                    u_cur -= solver.matvec(diff_col_sp, np.array([d / c_dd]))
            else:
                # Dense path: unchanged.
                diff_col = C_cur[:, li] - C_cur[:, lj]
                if schur_tol > 0.0:
                    active = np.abs(diff_col) > schur_tol
                    k = int(active.sum())
                    if fill_tol > 0.0:
                        level_active.append(float(k / len(diff_col)))
                    if k == 0:
                        pass
                    elif k >= len(u_cur) * _SCHUR_BROADCAST_THRESHOLD:
                        # High active fraction: masked broadcast avoids copy+scatter overhead
                        dc_full = diff_col * active
                        C_cur -= np.outer(dc_full, dc_full) / c_dd
                        u_cur -= (dc_full / c_dd) * d
                    else:
                        # Low active fraction: fancy indexing saves O(N²) work
                        dc = diff_col[active]
                        C_cur[np.ix_(active, active)] -= np.outer(dc, dc) / c_dd
                        u_cur[active] -= (dc / c_dd) * d
                else:
                    scale = diff_col / c_dd
                    C_cur -= scale[:, np.newaxis] * diff_col[np.newaxis, :]
                    u_cur -= scale * d

            # Sum-mode row/col accumulation
            u_cur[li] += u_cur[lj]
            if sp.issparse(C_cur):
                C_cur = C_cur.tolil()
                C_cur[li, :] += C_cur[lj, :]
                C_cur[:, li] += C_cur[:, lj]
                C_cur = C_cur.tocsc()
            else:
                C_cur[li, :] += C_cur[lj, :]
                C_cur[:, li] += C_cur[:, lj]
            new_local[id(node)] = li
            cols_to_delete.append(lj)

        # Singletons pass through unchanged
        for li, node in singleton_updates:
            new_local[id(node)] = li

        # Delete merged-away columns/rows (sort descending to preserve indices)
        if cols_to_delete:
            keep = sorted(set(range(len(u_cur))) - set(cols_to_delete))
            u_cur = u_cur[keep]
            if sp.issparse(C_cur):
                C_cur = C_cur[keep, :][:, keep]
            else:
                C_cur = C_cur[np.ix_(keep, keep)]
            # Remap indices
            old_to_new = {old: new for new, old in enumerate(keep)}
            new_local = {k: old_to_new[v] for k, v in new_local.items()}

        # Re-sparsify: zero out entries that fell below sparse_tol after Schur updates
        if sparse_tol > 0.0 and sp.issparse(C_cur) and len(u_cur) > 1:
            C_cur = C_cur.tocsr()
            C_cur.data[np.abs(C_cur.data) < sparse_tol] = 0.0
            C_cur.eliminate_zeros()
            C_cur = C_cur.tocsc()

        node_to_local = new_local

        level_sizes.append(len(u_cur))

        # Compute fill fraction of C_cur after deletions (= start of next level)
        if fill_tol > 0.0 and len(u_cur) > 1:
            C_dense = C_cur.toarray() if sp.issparse(C_cur) else C_cur
            mask = ~np.eye(len(u_cur), dtype=bool)
            fill_fractions.append(float(np.mean(np.abs(C_dense[mask]) >= fill_tol)))
        elif fill_tol > 0.0:
            fill_fractions.append(0.0)  # single node: no off-diagonal

        if verbose:
            msg = (
                f"  [RG] level {level_idx}: {len(next_nodes)} nodes"
            )
            if fill_tol > 0.0:
                msg += f", fill={fill * 100:.1f}%"
                if schur_tol > 0.0 and level_active:
                    mean_active = float(np.mean(level_active))
                    msg += f", active={mean_active * 100:.1f}%"
            msg += f", t={time.perf_counter()-t0:.4f}s"
            print(msg)

    # Hybrid mode: stopped early — return compressed matrix and vector for MLF handoff
    if stop_size > 1 and len(u_cur) > 1:
        C_out = C_cur.toarray() if sp.issparse(C_cur) else C_cur.copy()
        if return_diagnostics:
            if fill_tol > 0.0:
                return float(logL_acc), C_out, u_cur.copy(), level_sizes, fill_fractions
            return float(logL_acc), C_out, u_cur.copy(), level_sizes
        return float(logL_acc), C_out, u_cur.copy()

    # Full compression to a single node (default behaviour)
    assert len(u_cur) == 1, f"Expected 1 final node, got {len(u_cur)}"
    c_final = float(C_cur[0, 0])
    logL_acc += -0.5 * (np.log(abs(c_final)) + u_cur[0]**2 / c_final)

    logL = float(logL_acc)
    if return_diagnostics:
        if fill_tol > 0.0:
            return logL, level_sizes, fill_fractions
        return logL, level_sizes
    return logL


def rg_step_n2(u: np.ndarray, C: np.ndarray) -> float:
    """
    Exact RG log-likelihood for N=2 (one pair).

    Used for unit testing.  Should match scipy Cholesky exactly.

    Parameters
    ----------
    u : (2,) array, km/s
    C : (2, 2) array, (km/s)²

    Returns
    -------
    logL : float
    """
    c11, c12, c22 = C[0, 0], C[0, 1], C[1, 1]
    u1, u2 = u[0], u[1]

    # Integrate out difference mode d = u1 - u2
    c_dd = c11 + c22 - 2 * c12
    c_sd = c11 - c22  # covariance between sum and difference modes
    d = u1 - u2
    log_det_d = float(np.log(abs(c_dd)))
    quad_d = float(d**2 / c_dd)

    # Remaining sum mode s = u1 + u2, conditioned on d
    c_ss_new = c11 + c22 + 2 * c12 - c_sd**2 / c_dd  # Schur complement
    mu_s = c_sd / c_dd * d  # conditional mean of s given d
    u_s_tilde = (u1 + u2) - mu_s
    log_det_s = float(np.log(abs(c_ss_new)))
    quad_s = float(u_s_tilde**2 / c_ss_new)

    # +log(2): Jacobian of (u1,u2) -> (d,s) transformation
    return -0.5 * (log_det_d + quad_d + log_det_s + quad_s) + float(np.log(2))
