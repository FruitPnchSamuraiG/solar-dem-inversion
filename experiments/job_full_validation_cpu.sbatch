#!/bin/bash
#SBATCH --job-name=dem_fullval_cpu
#SBATCH --output=logs/eval/fullval_%j.log
#SBATCH --error=logs/eval/fullval_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_cilvr

# CPU version of job_full_validation.sbatch, for when the GPU queues are long.
# experiments/full_validation.py: every pixel of a staged split, one objective for
# every model. Set TRACK=bp|enet, PHASE=val|test, MODELS="label=ckpt ...", TAG.
# LABELS=1 scores the staged solver DEM (BP only: the staged ElasticNet DEMs are the
# older alpha = 1 labels). SUP=1 adds the supervised checkpoint.
# Smoke test: MAX_BLOCKS=20 TAG=smoke.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   ROOT=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS; LOSS=(--loss barrier --mu "${MU:-100}")
        SUPPTH=eval_and_enet_specs/checkpoints/model_best_bp.pth ;;
  enet) ROOT=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS; LOSS=(--loss enet)
        SUPPTH=eval_and_enet_specs/checkpoints/model_best_en.pth ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
read -r -a M <<< "${MODELS:?}"
args=(--root "$ROOT" --phase "${PHASE:-val}" "${LOSS[@]}" --models "${M[@]}"
      --workers 4 --output "output/experiments/diagnostics/${TRACK}_full_validation_${TAG:?}.json")
if [ "${LABELS:-0}" = 1 ]; then args+=(--labels); fi
if [ "${SUP:-0}" = 1 ]; then args+=(--supervised "$SUPPTH"); fi
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max_blocks "$MAX_BLOCKS"); fi
python3 -u experiments/full_validation.py "${args[@]}"
