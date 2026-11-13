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
#   INPUT=sqrt FOURIER=12 (INPUT=log1p FOURIER=0 for the log1p input)  CLIP=1
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
# INIT_FROM=<checkpoint>: warm start from its weights (a continuation run)
if [ -n "${INIT_FROM:-}" ]; then EXTRA+=(--init_from "$INIT_FROM"); fi
# CLAMP_MODE=mask keeps deconvolution-clamp pixels with the clamped channels free;
# RELAX_BAND=1 widens the band by the solver's tolLevel (both add a tag suffix)
if [ -n "${CLAMP_MODE:-}" ]; then EXTRA+=(--clamp_mode "$CLAMP_MODE"); fi
if [ "${RELAX_BAND:-0}" = 1 ]; then EXTRA+=(--relax_band); fi
# BAND_FORM=price|huber sets the band weight per pixel and channel from the input, with the
# tolerance lowered from EPS to EPS_END over the first 70% of training (KAPPA: huber slope cap)
# AL_MU>0: method of multipliers, multipliers from frozen copies every AL_EVERY epochs (at most AL_MAX)
if [ -n "${AL_MU:-}" ]; then EXTRA+=(--al_mu "$AL_MU" --al_every "${AL_EVERY:-20}" --al_max "${AL_MAX:-5}"); fi
# AL_RULE=adaptive: mu follows the textbook rule from AL_MU (x10 when violations stall), stop at AL_STOP
# half-widths (99% of solver-feasible validation pixels); AL_CAP caps multipliers at AL_CAP x price scale
if [ -n "${AL_RULE:-}" ]; then EXTRA+=(--al_rule "$AL_RULE" --al_stop "${AL_STOP:-0.1}" --al_cap "${AL_CAP:-0}"
  --al_outer_mode "${AL_OUTER:-fixed}" --al_floor_gain "${AL_FLOOR:-0}" --al_growth "${AL_GROWTH:-10}"); fi
# OPTIMIZER=muon: Muon on the hidden weight matrices (MUON_LR, Adam-matched step), Adam on the rest
if [ -n "${OPTIMIZER:-}" ]; then EXTRA+=(--optimizer "$OPTIMIZER" --muon_lr "${MUON_LR:-3e-4}"); fi
if [ -n "${BAND_FORM:-}" ]; then
  EXTRA+=(--band_form "$BAND_FORM" --eps "${EPS:-0.1}" --eps_end "${EPS_END:-0.001}" --kappa "${KAPPA:-64}")
fi
python3 -u experiments/train_scaled.py \
  --root "$ROOT" \
  --variant mlp6 \
  --hidden "${HIDDEN:-336}" \
  "${LOSS_ARGS[@]}" \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps "${WARMUP:-$WU}" \
  --input_transform "${INPUT:-sqrt}" \
  --fourier_freqs "${FOURIER:-12}" \
  --clip "${CLIP:-1.0}" \
  --resample_pixels \
  --bright_gamma "${BRIGHT_GAMMA:-0}" \
  --mu "${MU:-1.0}" \
  --l1_bright_power "${L1_BRIGHT_POWER:-0}" \
  --tag_suffix "${TAG_SUFFIX:-}" \
  --seed "${SEED:-42}" \
  --lr "${LR:-3e-4}" \
  --resume \
  --max_hours "${MAX_HOURS:-1.7}" \
  "${EXTRA[@]}" \
  --out_dir output/experiments/followup
