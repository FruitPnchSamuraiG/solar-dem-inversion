#!/bin/bash
#SBATCH --job-name=dem_followup
#SBATCH --output=logs/scaled/followup_%j.log
#SBATCH --error=logs/scaled/followup_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:59:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# One resumable segment of a follow-up run (2026-10-06 meeting suggestions).
# Base recipe = the final ElasticNet model, now for both tracks: mlp6, 336 wide,
# square root + 12 Fourier frequencies, fresh pixels every epoch, 40 epochs.
# Each experiment changes one thing through the environment:
#   TRACK=bp|enet  HIDDEN=336  EPOCHS=40  BRIGHT_GAMMA=0  TAG_SUFFIX=  TF32=0
#   MU=1 (barrier weight; >1 makes the band stiffer, closer to the solver's
#   hard constraint; give it a TAG_SUFFIX such as _mu10)
#   L1_BRIGHT_POWER=0 (barrier L1 weight / max(1, I)**p; tag it too, e.g. _ibr1)
# A segment stops before MAX_HOURS (default 1.7; GPU jobs past 2 h with low
# utilisation get cancelled) and the next segment resumes from <tag>_state.pt.
# Extra segments exit at once when training is already complete.
# Submit through experiments/submit_followup.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/followup
echo "### git HEAD: $(git rev-parse --short HEAD)"
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   ROOT=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS
        LOSS_ARGS=(--loss barrier); WU=500 ;;
  enet) ROOT=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS
        LOSS_ARGS=(--loss enet --enet_C -1 --enet_alpha 0.001 --enet_lam 0.5); WU=3000 ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
EXTRA=()
if [ "${TF32:-0}" = 1 ]; then EXTRA+=(--tf32); fi
python3 -u experiments/train_scaled.py \
  --root "$ROOT" \
  --variant mlp6 \
  --hidden "${HIDDEN:-336}" \
  "${LOSS_ARGS[@]}" \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps "${WARMUP:-$WU}" \
  --input_transform sqrt \
  --fourier_freqs 12 \
  --resample_pixels \
  --bright_gamma "${BRIGHT_GAMMA:-0}" \
  --mu "${MU:-1.0}" \
  --l1_bright_power "${L1_BRIGHT_POWER:-0}" \
  --tag_suffix "${TAG_SUFFIX:-}" \
  --resume \
  --max_hours "${MAX_HOURS:-1.7}" \
  "${EXTRA[@]}" \
  --out_dir output/experiments/followup
