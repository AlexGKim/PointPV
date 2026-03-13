#!/bin/bash
#SBATCH --job-name=pointpv_baseline
#SBATCH --account=m1234          # Update with your NERSC project code
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ---- environment ----
module load python cudatoolkit
conda activate pointpv_gpu

export POINTPV_BACKEND=cupy

# ---- parameters (override from sbatch --export or edit here) ----
N=${N:-1000}
FS8_MIN=${FS8_MIN:-0.2}
FS8_MAX=${FS8_MAX:-0.8}
N_GRID=${N_GRID:-40}
CATALOG=${CATALOG:-data/mock_${N}.npz}
OUTPUT=${OUTPUT:-results/baseline_${N}.npz}

mkdir -p logs results

echo "=== PointPV MLF Baseline ==="
echo "N=${N}  backend=${POINTPV_BACKEND}  nodes=1"
date

python scripts/run_baseline.py \
    --n ${N} \
    --catalog ${CATALOG} \
    --output ${OUTPUT} \
    --backend cupy \
    --fs8-min ${FS8_MIN} \
    --fs8-max ${FS8_MAX} \
    --n-grid ${N_GRID}

echo "Done."
date
