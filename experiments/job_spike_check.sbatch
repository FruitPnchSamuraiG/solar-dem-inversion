#!/bin/bash
#SBATCH --job-name=dem_spike
#SBATCH --output=logs/eval/spike_%j.log
#SBATCH --error=logs/eval/spike_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:59:00
#SBATCH --mail-type=FAIL

# experiments/spike_check.py: every pixel of a staged BP split, each channel against the
# median of its eight neighbors at full resolution, with the final model's self-check.
#   sbatch --export=ALL,PHASE=val experiments/job_spike_check.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
MODEL=${MODEL_PATH:-output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt}
args=(--root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS" --phase "${PHASE:-val}" --model "$MODEL"
      --workers 14 --output "output/experiments/diagnostics/bp_spike_check_${PHASE:-val}${TAG:+_$TAG}.json")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max_blocks "$MAX_BLOCKS"); fi
python3 -u experiments/spike_check.py "${args[@]}"
