# McDonald (2019) Renormalization Group Likelihood

## Reference

McDonald (2019), PhysRevD.100.043511, §III.

## Overview

The standard Gaussian log-likelihood for peculiar velocity surveys is

    L(θ) = -½ [log|C(θ)| + uᵀ C(θ)⁻¹ u]

where `u` is the N-vector of line-of-sight velocities and `C(θ)` is the
N×N velocity-velocity covariance matrix parameterised by `θ = fsigma8`.
Direct evaluation via Cholesky decomposition costs O(N³), which becomes
prohibitive for N > ~10⁴.

McDonald (2019) reformulates this as a hierarchical coarse-graining:
at each level of a binary tree, pairs of nearby galaxies are merged by
integrating out their "difference mode," reducing the effective N by ~½
at each step.  The total cost is O(N log N) (or O(N) with sparse C).

## RG Step for One Pair (i, j)

Given current sum-mode quantities `u_i`, `u_j`, `C_ii`, `C_jj`, `C_ij`:

**Difference mode** `d = u_i - u_j`:

    C_dd  = C_ii + C_jj - 2 C_ij          (variance of d)
    ΔlogL = -½ [log C_dd + d² / C_dd]      (contribution to log|C| and quadratic)

**Sum mode** `s = u_i + u_j` (survives to next level):

    u_s   = u_i + u_j
    C_ss  = C_ii + C_jj + 2 C_ij - (C_ij - C_jj)² / C_dd

The last term is the Schur complement that accounts for the information
gained by observing d.

## Cross-covariance propagation

For two merged nodes a = (i,j) and b = (k,l), the new cross-covariance is

    C_ab_new = C_ik + C_il + C_jk + C_jl

This is exact for the sum-mode after integrating out both difference modes.

## Algorithm (rg_coarsen_all)

1. Build a binary KD-tree pairing (greedy nearest-neighbour, `tree.build_tree`).
2. At each level l = 1 … depth:
   a. For each pair node: accumulate ΔlogL, compute new u_s and C_ss.
   b. For each singleton: pass through unchanged.
   c. Rebuild the u and C arrays at half the size.
3. Final single node: add log(C_00) + u_0² / C_00.
4. Return L = -½ (accumulated log_det + accumulated quad).

## Code locations

| Concept                  | File                              |
|--------------------------|-----------------------------------|
| Tree building            | `pointpv/rg/tree.py`              |
| RG coarsening loop       | `pointpv/rg/coarsen.py`           |
| N=2 exact unit           | `pointpv/rg/coarsen.py:rg_step_n2` |
| Top-level likelihood     | `pointpv/likelihood/rg.py`        |

## Accuracy

For a dense (exact) covariance matrix, the RG result equals the Cholesky
result to machine precision (~10⁻¹² absolute difference).  When a spatial
cutoff is applied to C (setting distant pairs to zero), a small systematic
error is introduced, controlled by the cutoff scale.
