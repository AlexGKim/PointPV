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
- Laptop:     conda env create -f environment_cpu.yml
- Perlmutter: conda env create -f environment_gpu.yml  (adds petsc4py, cupy)

## Conventions
- All covariance matrices are in units of (km/s)²
- Distances in Mpc/h, velocities in km/s
- fsigma8 is the single free parameter; cosmology otherwise fixed to AbacusSummit cosmology
- Log-likelihood is defined as L = -½[log|C| + uᵀC⁻¹u] (no constant term)

## Running benchmarks
    python scripts/run_baseline.py --n 1000 --backend scipy
    python scripts/run_rg.py --n 1000 --backend scipy
    python scripts/compare.py --n 1000

## NERSC
- Module load: python, cudatoolkit (for GPU runs)
- Submit jobs: sbatch slurm/baseline_job.sh, sbatch slurm/rg_job.sh
- Partition: gpu (A100), nodes: 1–4 for scaling tests
