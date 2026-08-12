#!/bin/bash
#SBATCH --job-name=dem_fullvis_seq
#SBATCH --output=logs/visuals/fullvis_seq_%j.log
#SBATCH --error=logs/visuals/fullvis_seq_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# One GPU, all curated dates sequentially: 8 dates x {BP, ENet} = 16 exports.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/visuals
mapfile -t DATES < <(grep -v '^#' experiments/visualizer_dates.txt | sed '/^$/d')

for track in bp enet; do
  for timestamp in "${DATES[@]}"; do
    echo "=== track=$track timestamp=$timestamp ==="
    TRACK="$track" TIMESTAMP="$timestamp" bash experiments/job_export_full_disk_visuals.sbatch
  done
done
