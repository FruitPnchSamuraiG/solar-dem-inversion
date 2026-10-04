#!/bin/bash
#SBATCH --job-name=dem_sqrt_ff
#SBATCH --output=logs/scaled/sqrt_ff_%j.log
#SBATCH --error=logs/scaled/sqrt_ff_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# The label-free model with the supervised model's input path: sqrt(AIA) then
# Fourier features at 12 frequencies (src/model.py:posEncode). Defaults
# reproduce the first run (19049208): h232 BP, fixed pixels, warmup 3000.
# For the final model (2026-10-04): TRACK=bp|enet HIDDEN=680 RESAMPLE=1 and
# each track's warmup (BP 500, ENet 3000), matching its baseline run; see
# submit_input_encoding_final.sh. Tags get _sqrt_ff12 (and _resample), so no
# baseline checkpoint can be overwritten. A checkpoint is saved every epoch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/input_encoding
echo "### git HEAD: $(git rev-parse --short HEAD)"
case "${TRACK:-bp}" in
  bp)   ROOT=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS; LOSS_ARGS=(--loss barrier) ;;
  enet) ROOT=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS
        LOSS_ARGS=(--loss enet --enet_C -1 --enet_alpha 0.001 --enet_lam 0.5) ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
EXTRA=()
if [ "${RESAMPLE:-0}" = 1 ]; then EXTRA+=(--resample_pixels); fi
python3 -u experiments/train_scaled.py \
  --root "${TRAIN_ROOT:-$ROOT}" \
  --variant mlp6 \
  --hidden "${HIDDEN:-232}" \
  "${LOSS_ARGS[@]}" \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps "${WARMUP:-3000}" \
  --input_transform sqrt \
  --fourier_freqs "${FOURIER_FREQS:-12}" \
  "${EXTRA[@]}" \
  --out_dir output/experiments/input_encoding
