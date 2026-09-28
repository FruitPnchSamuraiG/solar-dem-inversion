#!/bin/bash
#SBATCH --job-name=dem_bright_diag
#SBATCH --output=logs/eval/bright_diag_%j.log
#SBATCH --error=logs/eval/bright_diag_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_tandon_advanced

# One track per job. This is targeted full-test analysis, not training. The
# default samples 8 pixels/block for prominence-aware shape tests (~390k curves
# over the full test); full trigger/error accounting still uses every pixel.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics

case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp) DATA=/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS
      MODEL=output/experiments/sweep/scaled_mlp6_barrier_h232.pt ;;
  enet) DATA=/projects/rps/dff6142/fouheylab/solar_dem/enet_AIA_hofdeconv_full_DS
        MODEL=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac

args=(--model "$MODEL" --data "$DATA" --reference "$TRACK"
      --output "output/experiments/diagnostics/${TRACK}_bright_failure.json")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
if [ -n "${SHAPE_SAMPLES_PER_BLOCK:-}" ]; then
  args+=(--shape-samples-per-block "$SHAPE_SAMPLES_PER_BLOCK")
fi
python3 -u experiments/diagnose_bright_failures.py "${args[@]}"
