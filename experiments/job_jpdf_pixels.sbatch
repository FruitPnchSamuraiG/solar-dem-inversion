#!/bin/bash
#SBATCH --job-name=dem_jpdf
#SBATCH --output=logs/eval/jpdf_%j.log
#SBATCH --error=logs/eval/jpdf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_cilvr

# Census of the JPDF low-end pixels, the deconvolution-clamp pixels and the
# solver's relaxed (tolLevel) pixels: experiments/jpdf_pixels.py. Evaluation
# only, CPU.
#   MODE=shared  one pass over the shared test set: clusters, clamp, isolation,
#                headline metrics under masks (default)
#   MODE=tol     one pass over our own test split: by tolLevel and by position
#                on the Sun
#   TRACK=bp|enet.  Smoke test: add MAX_BLOCKS=10,TAG=smoke to --export.
# Submit the second track with --account=torch_pr_41_courant so both start at once.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
WORKERS=${WORKERS:-16}
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp) LF=output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt
      SUP=eval_and_enet_specs/checkpoints/model_best_bp.pth
      SHARED=/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS
      OWN=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS ;;
  enet) LF=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
        SUP=eval_and_enet_specs/checkpoints/model_best_en.pth
        SHARED=/projects/rps/dff6142/fouheylab/solar_dem/enet_AIA_hofdeconv_full_DS
        OWN=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
# LF_MODEL overrides the label-free checkpoint (e.g. a new training run); use with TAG.
LF=${LF_MODEL:-$LF}
[ -f "$LF" ] || { echo "missing checkpoint: $LF" >&2; exit 3; }
MODE=${MODE:-shared}
case "$MODE" in
  shared) DATA=$SHARED; OUT=output/experiments/diagnostics/${TRACK}_jpdf_pixels${TAG:+_$TAG}.json ;;
  tol)    DATA=$OWN;    OUT=output/experiments/diagnostics/${TRACK}_jpdf_pixels_tol${TAG:+_$TAG}.json ;;
  *) echo "MODE must be shared or tol" >&2; exit 2 ;;
esac
args=(--mode "$MODE" --label-free "$LF" --supervised "$SUP" --reference "$TRACK"
      --data "$DATA" --output "$OUT" --workers "$WORKERS" --threads-per-worker "$THREADS")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
python3 -u experiments/jpdf_pixels.py "${args[@]}"
