#!/bin/bash
#SBATCH --job-name=dem_first_mlp
#SBATCH --output=logs/eval/first_mlp_%j.log
#SBATCH --error=logs/eval/first_mlp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --time=01:30:00
#SBATCH --account=torch_pr_41_general

# Recreate the June per-pixel MLP (its checkpoint was not kept) with its
# original settings, then plot it against the ablation patch CNN and BP on the
# ablation's held-out pixels. Small CPU job: 213k params, ~221k pixels.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/first_mlp_repro
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "### git HEAD: $(git rev-parse --short HEAD)"
DIRS=(data/20110906_2217 data/20120603_0000 data/20131113_0908 data/20140910_1731)
python3 -u experiments/train_unsupervised.py --data_dirs "${DIRS[@]}" \
  --crop 1800,1800,256,256 --hidden 256 --epochs 30 --loss_name barrier
python3 -u experiments/plot_first_mlp_vs_patch_cnn.py --data_dirs "${DIRS[@]}" \
  --threads "$SLURM_CPUS_PER_TASK"
