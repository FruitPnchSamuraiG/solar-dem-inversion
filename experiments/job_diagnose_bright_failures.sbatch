#!/bin/bash
#SBATCH --job-name=dem_bright_diag
#SBATCH --output=logs/eval/bright_diag_%j.log
#SBATCH --error=logs/eval/bright_diag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --time=03:30:00
#SBATCH --account=torch_pr_41_general

# One track per job: TRACK=bp or TRACK=enet.  CPU only -- the 176k mlp6 is
# tiny, the job is read-bound, and repeated AIA inputs are predicted once.
# Every valid pixel of the shared test set is used; nothing is sampled.
# Smoke test: add MAX_BLOCKS=640 and TAG=smoke to --export.
# Side by side afterwards: diagnose_bright_failures.py --compare LF.json SUP.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics

WORKERS=${WORKERS:-8}
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS ))
(( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

# MODEL_KIND=supervised runs Samuel's checkpoint through the identical analysis.
case "${TRACK:?set TRACK=bp or TRACK=enet}:${MODEL_KIND:-label_free}" in
  bp:label_free) MODEL=output/experiments/sweep/scaled_mlp6_barrier_h232.pt ;;
  enet:label_free) MODEL=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt ;;
  bp:supervised) MODEL=eval_and_enet_specs/checkpoints/model_best_bp.pth ;;
  enet:supervised) MODEL=eval_and_enet_specs/checkpoints/model_best_en.pth ;;
  *) echo "TRACK must be bp or enet; MODEL_KIND label_free or supervised" >&2; exit 2 ;;
esac
# MODEL_PATH overrides the checkpoint (e.g. a new training run); use with TAG.
MODEL=${MODEL_PATH:-$MODEL}
[ -f "$MODEL" ] || { echo "missing checkpoint: $MODEL" >&2; exit 3; }
DATA=/projects/rps/dff6142/fouheylab/solar_dem/${TRACK}_AIA_hofdeconv_full_DS
SUFFIX=""
if [ "${MODEL_KIND:-label_free}" = supervised ]; then SUFFIX=_supervised; fi
OUT=output/experiments/diagnostics/${TRACK}_bright_failure${SUFFIX}${TAG:+_$TAG}.json

args=(--model "$MODEL" --data "$DATA" --reference "$TRACK" --output "$OUT"
      --workers "$WORKERS" --threads-per-worker "$THREADS")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
echo "track=$TRACK model=$MODEL workers=$WORKERS threads/worker=$THREADS out=$OUT"
python3 -u experiments/diagnose_bright_failures.py "${args[@]}"
