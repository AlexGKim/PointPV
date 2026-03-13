# PointPV

Test project: McDonald 2019 renormalization group (RG) log-likelihood for fitting
fsigma8 from peculiar velocity surveys, benchmarked against the standard MLF method.

## Key References
- New method: McDonald 2019, PhysRevD.100.043511 — RG coarse-graining of Gaussian likelihoods
- Baseline: Lai et al. 2025, arXiv:2512.03229 — MLF method on DESI DR1 peculiar velocities
- Mock data: AbacusSummit light cone (Garrison et al. 2021, MNRAS 508 575)

## Data
- AbacusSummit light cone lives on NERSC Perlmutter (confirm exact path before use)
- For local development: copy a small subset (N~2000 halos) to pointpv/data/

## Architecture
- `pointpv/mock/`       — AbacusSummit light cone reader + mock PV catalog generation
- `pointpv/covariance/` — velocity-velocity covariance matrix via FLIP package
- `pointpv/likelihood/` — mlf.py (baseline Cholesky) and rg.py (McDonald RG)
- `pointpv/rg/`         — RG tree, coarse-graining step, sparse backend abstraction
- `pointpv/benchmark/`  — timing and accuracy comparison utilities

## Sparse Backend
Set `POINTPV_BACKEND=scipy` (default, CPU laptop dev) or `POINTPV_BACKEND=petsc`
(Perlmutter GPU/MPI) to switch between ScipySolver and PETScSolver in sparse_ops.py.

## Environments
- Laptop:     `generic` conda env works (has numpy, scipy, astropy, camb, pytest)
              conda env create -f environment_cpu.yml creates `pointpv-cpu`
- Perlmutter: conda env create -f environment_gpu.yml  (adds petsc4py, cupy)
- conda is miniforge3; use `$(conda info --base)/envs/generic/bin/python` for the
  direct python path (needed because `conda run` mis-parses `--n` as its own flag)

## FLIP
- Installed at /Users/akim/Projects/flip (not on sys.path by default)
- velocity.py inserts this path at runtime — no manual setup needed
- Uses lai22 velocity model with CAMB for P(k) (not CLASS)
- emcee and iminuit must be installed: pip install emcee iminuit

## RG coarsen_all
- `schur_tol` parameter controls speed/accuracy tradeoff
- schur_tol=0.0 (default): exact, ~560ms for N=1000
- schur_tol=1.0: 7x faster (~80ms), |ΔlogL|~1e-6 for physical covariance

## Conventions
- All covariance matrices are in units of (km/s)²
- Distances in Mpc/h, velocities in km/s
- fsigma8 is the single free parameter; cosmology otherwise fixed to AbacusSummit cosmology
- Log-likelihood is defined as L = -½[log|C| + uᵀC⁻¹u] (no constant term)

## Running benchmarks
    PY=$(conda info --base)/envs/generic/bin/python

    # Magnitude-limited catalog (default: m_lim=20, z_max=0.1, Schechter LF)
    $PY scripts/generate_mock.py --synthetic --n 1000 --output data/mock_1000.npz
    # Uniform-z catalog (add --no-mag-limit for quick tests without selection function)

    # Diagnostic plots → figs/nz_*.pdf, figs/sky_*.pdf
    $PY scripts/plot_catalog.py --catalog data/mock_1000.npz --output figs/

    $PY scripts/run_baseline.py --catalog data/mock_1000.npz
    $PY scripts/run_rg.py --catalog data/mock_1000.npz
    $PY scripts/compare.py --baseline results/baseline_1000.npz --rg results/rg_1000.npz

## NERSC
- Module load: python, cudatoolkit (for GPU runs)
- Submit jobs: sbatch slurm/baseline_job.sh, sbatch slurm/rg_job.sh
- Partition: gpu (A100), nodes: 1–4 for scaling tests
