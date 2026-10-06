#!/bin/bash
#SBATCH --job-name=dem_objgap
#SBATCH --output=logs/eval/objgap_%j.log
#SBATCH --error=logs/eval/objgap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_41_general

# experiments/objective_gap.py on our staged test split (it needs the per-pixel
# errors): network vs per-pixel optimum vs solver under the training objective,
# plus per-pixel optima of alternative objectives. TRACK=bp|enet, PHASE=test|val.
# Choose objective settings on PHASE=val; the test split only confirms.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp) ROOT=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS; LOSS=barrier
      MODELS=(final=output/experiments/resample/scaled_mlp6_barrier_h336_resample.pt
              sqrtff=output/experiments/input_encoding/scaled_mlp6_barrier_h336_sqrt_ff12_resample.pt)
      SUP=eval_and_enet_specs/checkpoints/model_best_bp.pth
      VARIANTS=${VARIANTS:-mu3,mu10,mu30,mu100,ibr1} ;;
  enet) ROOT=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS; LOSS=enet
      MODELS=(final=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt)
      SUP=eval_and_enet_specs/checkpoints/model_best_en.pth
      VARIANTS=${VARIANTS:-mu10} ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
python3 -u experiments/objective_gap.py --loss "$LOSS" --root "$ROOT" --phase "${PHASE:-test}" \
  --models "${MODELS[@]}" --supervised "$SUP" --variants "$VARIANTS" \
  --per-group "${PER_GROUP:-1000}" --core-cap "${CORE_CAP:-6000}" --workers 16 \
  --output "output/experiments/diagnostics/${TRACK}_objective_gap${TAG:+_$TAG}.json"
