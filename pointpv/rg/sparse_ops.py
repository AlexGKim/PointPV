"""
Sparse matrix backend abstraction for the McDonald RG algorithm.

Provides a unified SparseSolver interface with two backends:
  - ScipySolver : scipy.sparse (CPU, laptop development)
  - PETScSolver : petsc4py with cuSPARSE GPU backend (NERSC Perlmutter)

Select at runtime via the POINTPV_BACKEND environment variable:
    export POINTPV_BACKEND=scipy   # default
    export POINTPV_BACKEND=petsc   # GPU/MPI on Perlmutter

The RG coarse-graining in coarsen.py uses this module when sparse_tol > 0:
  - dense_to_sparse() converts the initial covariance to a CSC sparse matrix
  - get_solver() selects the backend (scipy or petsc) once per call
  - solver.matvec() performs the per-pair u-update SpMV (diff_col @ [d/c_dd])
    in a backend-agnostic way, enabling GPU acceleration on Perlmutter via PETSc.

Interface
---------
solver = get_solver()
solver.matvec(A, x)   → A @ x  (sparse × dense vector)
solver.solve(A, b)    → A⁻¹ b  (sparse direct or iterative solve)
solver.logdet(A)      → log|det A|  (for sparse positive-definite A)

Units
-----
No unit assumptions; inputs/outputs in whatever units the caller uses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_solver() -> "SparseSolver":
    """
    Return the appropriate SparseSolver for the current environment.

    Reads the POINTPV_BACKEND environment variable.  Defaults to ScipySolver.
    """
    backend = os.environ.get("POINTPV_BACKEND", "scipy").lower()
    if backend == "petsc":
        return PETScSolver()
    elif backend == "scipy":
        return ScipySolver()
    else:
        raise ValueError(
            f"Unknown POINTPV_BACKEND={backend!r}. "
            "Valid choices: 'scipy', 'petsc'."
        )


class SparseSolver(ABC):
    """Abstract base class for sparse matrix operations."""

    @abstractmethod
    def matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        """
        Sparse matrix–vector product A @ x.

        Parameters
        ----------
        A : sparse matrix, (M, N)
        x : (N,) array

        Returns
        -------
        (M,) array
        """

    @abstractmethod
    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """
        Solve A x = b for x.

        Parameters
        ----------
        A : sparse matrix, (N, N), positive-definite
        b : (N,) array

        Returns
        -------
        x : (N,) array
        """

    @abstractmethod
    def logdet(self, A: sp.spmatrix) -> float:
        """
        Compute log|det A| for a sparse positive-definite matrix.

        Parameters
        ----------
        A : sparse matrix, (N, N), positive-definite symmetric

        Returns
        -------
        float
        """


class ScipySolver(SparseSolver):
    """
    CPU sparse solver using scipy.sparse.

    Uses SuperLU (via spsolve) for direct solves and a log-determinant
    via LU decomposition diagonal products.
    """

    def matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        return A @ x

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        return spla.spsolve(A, b)

    def logdet(self, A: sp.spmatrix) -> float:
        """
        Compute log|det A| via sparse LU decomposition.

        Uses scipy.sparse.linalg.splu which factors A = P L U.
        log|det A| = Σ log|diag(U)|.
        """
        lu = spla.splu(A.tocsc())
        diagU = lu.U.diagonal()
        return float(np.sum(np.log(np.abs(diagU))))


class PETScSolver(SparseSolver):
    """
    GPU/MPI sparse solver using petsc4py.

    Uses PETSc MATSEQAIJCUSPARSE matrices and KSP solvers for
    cuSPARSE-accelerated sparse operations on NVIDIA GPUs.

    This backend is intended for NERSC Perlmutter A100 GPU nodes.
    Requires petsc4py compiled with CUDA support.
    """

    def __init__(self) -> None:
        try:
            import petsc4py
            from petsc4py import PETSc
            self._PETSc = PETSc
        except ImportError as e:
            raise ImportError(
                "petsc4py is required for the PETSc backend. "
                "On Perlmutter: conda install petsc4py"
            ) from e

    def _scipy_to_petsc(self, A: sp.spmatrix) -> object:
        """Convert a scipy sparse matrix to a PETSc Mat."""
        PETSc = self._PETSc
        A_csr = A.tocsr()
        n, m = A_csr.shape
        mat = PETSc.Mat().createAIJWithArrays(
            size=(n, m),
            csr=(A_csr.indptr, A_csr.indices, A_csr.data.astype(np.float64)),
        )
        mat.setType(PETSc.Mat.Type.SEQAIJCUSPARSE)
        mat.setUp()
        mat.assemble()
        return mat

    def matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        PETSc = self._PETSc
        mat = self._scipy_to_petsc(A)
        x_vec = PETSc.Vec().createSeq(len(x))
        x_vec.setArray(x.astype(np.float64))
        y_vec = mat.createVecLeft()
        mat.mult(x_vec, y_vec)
        return np.array(y_vec.getArray())

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        PETSc = self._PETSc
        mat = self._scipy_to_petsc(A)
        ksp = PETSc.KSP().create()
        ksp.setOperators(mat)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.CHOLESKY)
        ksp.setFromOptions()

        b_vec = PETSc.Vec().createSeq(len(b))
        b_vec.setArray(b.astype(np.float64))
        x_vec = b_vec.duplicate()
        ksp.solve(b_vec, x_vec)
        return np.array(x_vec.getArray())

    def logdet(self, A: sp.spmatrix) -> float:
        # PETSc does not expose log-det directly; fall back to scipy
        from scipy.sparse.linalg import splu
        lu = splu(A.tocsc())
        diagU = lu.U.diagonal()
        return float(np.sum(np.log(np.abs(diagU))))


def dense_to_sparse(
    C: np.ndarray,
    cutoff: float | None = None,
) -> sp.csr_matrix:
    """
    Convert a dense covariance matrix to a scipy sparse CSR matrix.

    Parameters
    ----------
    C : (N, N) ndarray
    cutoff : float, optional
        Zero out entries with |C_ij| < cutoff * max(|C|).

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    C_out = C.copy()
    if cutoff is not None:
        threshold = cutoff * np.max(np.abs(C))
        C_out[np.abs(C_out) < threshold] = 0.0
    return sp.csr_matrix(C_out)
