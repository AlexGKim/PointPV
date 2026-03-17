#!/bin/bash
#SBATCH --job-name=pointpv_gpu_bench
#SBATCH --account=m1234           # update with your NERSC project code
#SBATCH --constraint=gpu&hbm80g  # A100 80 GB nodes only
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_bench_%j.out
#SBATCH --error=logs/gpu_bench_%j.err

module load python cudatoolkit
conda activate pointpv-gpu

export POINTPV_BACKEND=cupy

PY=$(conda info --base)/envs/pointpv-gpu/bin/python
mkdir -p logs figs

# Default: FLIP/CAMB covariance (real science case).
# Add --no-flip for a quick synthetic-exponential test run.
$PY scripts/benchmark_scaling.py \
    --sizes 1000 5000 10000 30000 60000 \
    --schur-tols 50 100 500 1000 \
    --active-frac-stops 0.3 0.5 0.7 \
    --skip-mlf-large 5000 \
    --n-repeats 3 \
    --gpu \
    --output-dir figs
