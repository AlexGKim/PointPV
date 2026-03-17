#!/bin/bash
#SBATCH --job-name=pointpv_gpu_debug
#SBATCH --account=desi_g           # update with your NERSC project code
#SBATCH --partition=gpu
#SBATCH --constraint=gpu&hbm80g  # A100 80 GB nodes only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu_debug_%j.out
#SBATCH --error=logs/gpu_debug_%j.err

# Debug run: synthetic covariance (no FLIP/CAMB), small N, 1 repeat.
# Completes in < 30 minutes on a single A100.
# Purpose: verify GPU code path (CuPy Cholesky, RG-dense-GPU, hybrid handoff)
# before submitting the full benchmark_gpu_a100.sh job.

module load python cudatoolkit
conda activate pointpv-gpu

export POINTPV_BACKEND=cupy

mkdir -p logs figs

srun python scripts/benchmark_scaling.py \
    --sizes 64 256 1024 4096 \
    --schur-tols 5000 10000 30000 \
    --active-frac-stops 0.1 0.2 \
    --n-repeats 1 \
    --gpu \
    --output-dir figs
