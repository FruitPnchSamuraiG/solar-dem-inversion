#!/bin/bash
#SBATCH --job-name=dem_sqrt_ff
#SBATCH --output=logs/scaled/sqrt_ff_%j.log
#SBATCH --error=logs/scaled/sqrt_ff_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=02:15:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Production h232 BP label-free model with the supervised model's input path:
# sqrt(AIA) then Fourier features at 12 frequencies (src/model.py:posEncode).
# Everything else matches the width sweep (40 epochs, batch 16, warmup 3000).
# Target: the flare-core ceiling (94 A fit 0.24 vs solver 0.92). The tag gets
# a _sqrt_ff12 suffix, so the production checkpoint cannot be overwritten.
# 2:15 like the sweep rerun that needed it; a checkpoint is saved every epoch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/input_encoding
echo "### git HEAD: $(git rev-parse --short HEAD)"
python3 -u experiments/train_scaled.py \
  --root "${TRAIN_ROOT:-$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS}" \
  --variant mlp6 \
  --hidden 232 \
  --loss barrier \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps 3000 \
  --input_transform sqrt \
  --fourier_freqs "${FOURIER_FREQS:-12}" \
  --out_dir output/experiments/input_encoding
