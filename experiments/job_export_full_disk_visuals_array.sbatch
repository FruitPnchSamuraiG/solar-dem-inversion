#!/bin/bash
#SBATCH --job-name=dem_fullvis
#SBATCH --output=logs/visuals/fullvis_%A_%a.log
#SBATCH --error=logs/visuals/fullvis_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Eight curated dates x two solver tracks.  Submit with:
#   sbatch --array=0-15%2 experiments/job_export_full_disk_visuals_array.sbatch
# The %2 throttle keeps this visual-export work from monopolising the group GPU quota.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
mapfile -t DATES < <(grep -v '^#' experiments/visualizer_dates.txt | sed '/^$/d')
N=${#DATES[@]}
TASK=${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}
if [ "$TASK" -lt "$N" ]; then
  TRACK=bp
  TIMESTAMP=${DATES[$TASK]}
else
  TRACK=enet
  TIMESTAMP=${DATES[$((TASK - N))]}
fi
echo "task=$TASK track=$TRACK timestamp=$TIMESTAMP"
exec bash experiments/job_export_full_disk_visuals.sbatch
