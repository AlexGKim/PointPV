# MLF Baseline: Maximum Likelihood Function

## Overview

The baseline method evaluates the Gaussian log-likelihood

    L(θ) = -½ [log|C(θ)| + uᵀ C(θ)⁻¹ u]

via a dense Cholesky decomposition.

## Algorithm

1. Build the N×N velocity-velocity covariance matrix C(fsigma8) via FLIP
   (or the analytic fallback in `pointpv/covariance/velocity.py`).
2. Compute the Cholesky factor L such that C = L Lᵀ (scipy.linalg.cho_factor
   on CPU; cupy.linalg.cholesky on GPU).
3. log|C| = 2 Σ log(diag(L)).
4. Solve L z = u; quadratic form = zᵀz = uᵀ C⁻¹ u.
5. L = -½ (log|C| + quadratic).

Cost: O(N³) per evaluation.  For N = 1000, this is ~10⁻¹ s on a laptop;
for N = 10⁴ it is ~100 s.

## Code locations

| Concept                  | File                              |
|--------------------------|-----------------------------------|
| Cholesky log-likelihood  | `pointpv/likelihood/mlf.py`       |
| Covariance builder       | `pointpv/covariance/velocity.py`  |
| CLI scan script          | `scripts/run_baseline.py`         |

## Covariance model

The velocity-velocity covariance is built from linear theory:

    C_ij = (f σ_8)² × ξ_vv(r_ij; cosmology)

where ξ_vv is the velocity correlation function computed via FLIP
(`flip.covariance.build_covariance`).  For development without FLIP,
an analytic isotropic exponential kernel is used:

    C_ij = (f σ_8)² × σ_v² × exp(-r_ij / L)

with σ_v = 300 km/s and L = 100 Mpc/h.

## Reference

Lai et al. (2025), arXiv:2512.03229 — DESI DR1 peculiar velocity analysis
using the MLF method with the FLIP covariance package.
