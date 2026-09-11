#!/bin/bash
#SBATCH --job-name=dem_enet_match
#SBATCH --output=logs/scaled/enet_match_%j.log
#SBATCH --error=logs/scaled/enet_match_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:55:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Retrain the production h232 architecture with Samuel's confirmed ENet
# objective. Training remains label-free: the staged DEM is only read for the
# validation agreement printout, never used in the optimization loss.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/matched_enet_alpha0p001

TRAIN_ROOT=${TRAIN_ROOT:-$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS}
echo "Training loss uses alpha=0.001. Validation DEM labels in this older staging"
echo "came from alpha=1, so use val_loss for convergence and the shared test for agreement."
python3 -u experiments/train_scaled.py \
  --root "$TRAIN_ROOT" \
  --variant mlp6 \
  --hidden 232 \
  --loss enet \
  --enet_C -1 \
  --enet_alpha 0.001 \
  --enet_lam 0.5 \
  --epochs 40 \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps 3000 \
  --out_dir output/experiments/matched_enet_alpha0p001
