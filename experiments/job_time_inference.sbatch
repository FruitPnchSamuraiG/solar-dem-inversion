#!/bin/bash
#SBATCH --job-name=dem_time_inference
#SBATCH --output=logs/eval/time_inference_%j.log
#SBATCH --error=logs/eval/time_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=torch_pr_41_cilvr

# Throughput of the final networks over one 2048 x 2048 DEM grid, on one GPU and
# on the 16 CPU cores of a solver job, for the paper's speed comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/timing
CKPT=output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt

python3 -u experiments/time_inference.py --device cuda --ckpt "$CKPT" --batch 1048576 \
    --out "output/experiments/timing/gpu_${SLURM_JOB_ID}.json"
OMP_NUM_THREADS=16 python3 -u experiments/time_inference.py --device cpu --n_pixels 1048576 \
    --out "output/experiments/timing/cpu16_${SLURM_JOB_ID}.json"
