#!/bin/bash
#SBATCH --job-name=dem_sweep_rs_cpu
#SBATCH --output=logs/sweep/sweep_rs_cpu_%j.log
#SBATCH --error=logs/sweep/sweep_rs_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --partition=cpu_short
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_general

# One task of job_sweep_resample.sbatch on a CPU node, for the small widths:
# the GPU queue is the bottleneck and these models are data-loading bound.
#   sbatch --export=ALL,TASK=10 experiments/job_sweep_resample_cpu.sbatch
# 10 data-loader workers + 22 compute threads.
set -euo pipefail
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22
exec bash experiments/job_sweep_resample.sbatch
