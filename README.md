# PointPV

**McDonald 2019 Renormalization Group Likelihood for Peculiar Velocity Surveys**

This project tests whether the McDonald (2019) RG coarse-graining algorithm can replace
the standard maximum-likelihood function (MLF) approach for fitting fsigma8 from large
peculiar velocity surveys, with O(N) scaling vs. O(N³) for the standard method.

## Quick Start

### Laptop (CPU, N ≤ 2,000)

```bash
# Use existing generic env, or: conda env create -f environment_cpu.yml
conda activate generic

# Install extra deps if needed
pip install emcee iminuit camb

# Generate a magnitude-limited synthetic catalog (no lightcone required)
# Default: m_lim=20, z_max=0.1, Schechter LF.  Add --no-mag-limit for uniform z.
python scripts/generate_mock.py --synthetic --n 1000 --output data/mock_1000.npz

# Diagnostic plots (n(z) histogram + Mollweide sky map → figs/)
python scripts/plot_catalog.py --catalog data/mock_1000.npz --output figs/

# Run baseline MLF likelihood scan
python scripts/run_baseline.py --catalog data/mock_1000.npz --backend scipy

# Run McDonald RG likelihood scan
python scripts/run_rg.py --catalog data/mock_1000.npz --backend scipy

# Compare results
python scripts/compare.py --baseline results/baseline_1000.npz --rg results/rg_1000.npz
```

> **Note:** avoid running scripts via `conda run` — the `--n` argument conflicts
> with conda's own `-n`/`--name` flag. Call python directly instead.

### Perlmutter (GPU, N up to 100,000)

```bash
conda env create -f environment_gpu.yml
conda activate pointpv-gpu

sbatch slurm/baseline_job.sh
sbatch slurm/rg_job.sh
```

## Method

### Baseline: MLF (Maximum Likelihood Function)

Evaluates L(θ) = -½ [log|C(θ)| + uᵀ C(θ)⁻¹ u] via dense Cholesky decomposition.
Complexity: O(N³). Uses FLIP for covariance matrix construction, CuPy for GPU Cholesky.

### McDonald RG Likelihood

Reformulates the same Gaussian likelihood via hierarchical coarse-graining:
at each level, pairs of adjacent galaxies are merged by integrating out their
"difference mode." Each step contributes a scalar to log|det C| and updates the
covariance of the "sum mode." Complexity: O(N log N) or O(N) with sparsity cutoff.

Reference: McDonald 2019, PhysRevD.100.043511.

## Project Structure

```
pointpv/
├── mock/           AbacusSummit light cone reader + mock PV catalog
├── covariance/     Velocity-velocity covariance (wraps FLIP)
├── likelihood/     mlf.py (Cholesky baseline) + rg.py (McDonald RG)
├── rg/             Tree building, RG coarse-graining, sparse backends
└── benchmark/      Timing and accuracy utilities
```

## Key Results

See `docs/results.md` (updated as experiments run).

## Testing & Validation

### Fast pytest suite (no FLIP required, < 60 s)

```bash
PY=$(conda info --base)/envs/generic/bin/python
$PY -m pytest tests/ -v -m "not flip"
```

### Full suite including FLIP covariance tests (~2–5 min)

```bash
$PY -m pytest tests/ -v
```

### Test file summary

| File | Covers |
|------|--------|
| `tests/test_rg_n2.py` | N=2 exact closed-form special case |
| `tests/test_rg_regression.py` | RG vs Cholesky for N=10–100 |
| `tests/test_rg_large_n.py` | RG vs Cholesky for N=200, 500, 1000 (`-m slow`) |
| `tests/test_rg_odd_n.py` | Odd-N singleton pass-through (N=3,7,11,33,101) |
| `tests/test_rg_schur_tol.py` | `schur_tol` accuracy tradeoff at N=200 |
| `tests/test_rg_coarsening.py` | Active-node count halves per RG level |
| `tests/test_rg_flip_covariance.py` | SPD check + RG/MLF agreement with real FLIP covariance (`-m flip`) |
| `tests/test_plots.py` | Smoke tests for all plot-generation functions |

### Science validation script

Runs an end-to-end fsigma8 recovery with both methods and generates all
diagnostic plots under `figs/`:

```bash
# Synthetic exponential covariance (fast, ~10 s for N=200)
$PY scripts/validate_fsigma8.py --n 200 --n-grid 40

# Real FLIP/CAMB covariance (slow, requires FLIP)
$PY scripts/validate_fsigma8.py --n 50 --flip
```

Output files produced:
- `figs/validate_compare.png` — log-likelihood curves and MLF/RG difference
- `figs/validate_scaling.png` — wall-clock timing comparison
- `figs/nz_validate.pdf` — n(z) histogram
- `figs/sky_validate.pdf` — Mollweide sky map

## References

- McDonald 2019, PhysRevD.100.043511
- Lai et al. 2025, arXiv:2512.03229 (DESI DR1 peculiar velocities)
- Garrison et al. 2021, MNRAS 508 575 (AbacusSummit)
- FLIP: https://github.com/corentinravoux/flip
