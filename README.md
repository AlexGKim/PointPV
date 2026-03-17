# PointPV

**McDonald 2019 Renormalization Group Likelihood for Peculiar Velocity Surveys**

This project tests whether the McDonald (2019) RG coarse-graining algorithm can replace
the standard maximum-likelihood function (MLF) approach for fitting fsigma8 from large
peculiar velocity surveys, with O(N log N) scaling vs. O(N³) for the standard method.

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
python scripts/run_baseline.py --catalog data/mock_1000.npz

# Run McDonald RG likelihood scan
python scripts/run_rg.py --catalog data/mock_1000.npz

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
covariance of the "sum mode." Complexity: O(N³) dense (default), O(N²) with schur_tol > 0.

Reference: McDonald 2019, PhysRevD.100.043511.

## Project Structure

```
pointpv/
├── mock/           AbacusSummit light cone reader + mock PV catalog
├── covariance/     Velocity-velocity covariance (wraps FLIP)
├── likelihood/     mlf.py (Cholesky baseline) + rg.py (McDonald RG)
├── rg/             Tree building, RG coarse-graining
└── benchmark/      Timing and accuracy utilities
```

## Key Results

See `docs/results.md` (updated as experiments run).

## Testing & Validation

### Fast pytest suite (no FLIP required, < 60 s)

```bash
PY=$(conda info --base)/envs/generic/bin/python
$PY -m pytest tests/ -v -m "not flip and not slow"

# Or via Makefile:
make test-fast
```

### Slow tests (scaling benchmarks, no FLIP)

```bash
$PY -m pytest tests/ -v -m "not flip"
make test-slow
```

### Full suite including FLIP covariance tests (~2–5 min)

```bash
$PY -m pytest tests/ -v
make test-all
```

### Test file summary

| File | Covers |
|------|--------|
| `tests/test_rg_n2.py` | N=2 exact closed-form special case |
| `tests/test_rg_regression.py` | RG vs Cholesky for N=10–100 |
| `tests/test_rg_large_n.py` | RG vs Cholesky for N=200, 500, 1000 (`-m slow`) |
| `tests/test_rg_odd_n.py` | Odd-N singleton pass-through (N=3,7,11,33,101) |
| `tests/test_rg_schur_tol.py` | `schur_tol` accuracy tradeoff at N=200 |
| `tests/test_rg_coarsening.py` | Fill fraction tracking; level-size shrinkage |
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

# Sparsity diagnostic table (fill% per level)
$PY scripts/validate_fsigma8.py --n 200 --n-grid 5 --fill-tol 1.0
```

Output files produced:
- `figs/validate_compare.png` — log-likelihood curves and MLF/RG difference
- `figs/validate_scaling.png` — wall-clock timing comparison
- `figs/nz_validate.pdf` — n(z) histogram
- `figs/sky_validate.pdf` — Mollweide sky map

Or via Makefile: `make validate`

### Runtime scaling benchmark

Measures wall-clock time for MLF, RG-dense, and RG variants across a range of
catalog sizes and produces a two-panel figure: log-log runtime scaling (with
N³ and N log N reference lines) and |ΔlogL| accuracy vs N.

```bash
# Default: N = 100, 1000, 10000  (N=10000 ~800 MB RAM, MLF ~30-120 s)
$PY scripts/benchmark_scaling.py

# Laptop-friendly, synthetic exponential covariance
$PY scripts/benchmark_scaling.py --no-flip --sizes 100 500 1000 2000

# FLIP/CAMB covariance (default), specific schur_tol values
$PY scripts/benchmark_scaling.py --sizes 200 500 1000 --schur-tols 1 100 1000 10000

# Hybrid RG+MLF via active_frac_stop: stop when mean active fraction >= threshold,
# hand off compressed state to Cholesky.  Requires schur_tol > 0.
# Each (schur_tol, active_frac_stop) pair adds a "RG-schur=X+afs=Y" variant.
$PY scripts/benchmark_scaling.py --no-flip --sizes 100 500 1000 \
    --schur-tols 1.0 --active-frac-stops 0.3 0.5 0.7

# Skip MLF for large N (only time RG methods)
$PY scripts/benchmark_scaling.py --skip-mlf-large 5000

# Via Makefile (N=100, 1000)
make benchmark
```

Output: `figs/benchmark_scaling.png`

### Hybrid RG+MLF stop-size benchmark

Finds the optimal point to hand off from RG compression to a final Cholesky
solve. Sweeps `stop_size` (powers of 2 from 1 to N) and plots total time
T_RG(N→stop) + T_MLF(stop) vs stop_size.

```bash
# Dense path (shows flat curve — no benefit for dense)
$PY scripts/benchmark_hybrid.py --sizes 500 1000

# Multiple N to see how optimal stop_size scales
$PY scripts/benchmark_hybrid.py --sizes 200 500 1000 2000
```

Output: `figs/benchmark_hybrid_<mode>.png`

## Status: incomplete items

### Local (laptop)

| Item | Details |
|------|---------|
| No mock catalog yet | `data/` does not exist. Run `python scripts/generate_mock.py --synthetic --n 1000 --output data/mock_1000.npz` before any script. |
| `docs/results.md` is empty | All benchmark tables are placeholders. Populate by running the pipeline end-to-end. |
| FLIP tests not exercised locally | Tests marked `-m flip` are excluded from the fast suite. The covariance construction path (`pointpv/covariance/velocity.py` + FLIP) has not been run in CI. |

### NERSC (Perlmutter)

| Item | Details |
|------|---------|
| `--account=m1234` placeholder | Both `slurm/baseline_job.sh` and `slurm/rg_job.sh` use a dummy account code. Update before submitting. |
| `pointpv-gpu` env not created | Run `conda env create -f environment_gpu.yml` on Perlmutter before submitting jobs. |
| AbacusSummit path unconfirmed | `docs/data.md` lists a likely path but notes it must be confirmed. No end-to-end run on real data yet. |
| CuPy GPU Cholesky path untested | `POINTPV_BACKEND=cupy` in `baseline_job.sh` triggers a CuPy code path in `mlf.py` that has not been run on an A100. |

---

## References

- McDonald 2019, PhysRevD.100.043511
- Lai et al. 2025, arXiv:2512.03229 (DESI DR1 peculiar velocities)
- Garrison et al. 2021, MNRAS 508 575 (AbacusSummit)
- FLIP: https://github.com/corentinravoux/flip
