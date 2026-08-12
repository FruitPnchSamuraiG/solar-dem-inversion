#!/bin/bash
#SBATCH --job-name=dem_fullvis
#SBATCH --output=logs/visuals/fullvis_%j.log
#SBATCH --error=logs/visuals/fullvis_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Build model and solver-reference web assets for one complete AIA disk.
# Example:
#   sbatch --export=ALL,TRACK=bp,TIMESTAMP=20150923_180356 \
#     experiments/job_export_full_disk_visuals.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/visuals

TIMESTAMP=${TIMESTAMP:-20150923_180356}
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   INPUT="$SCRATCH/dem/data/lp_AIA_hofdeconv_full/${TIMESTAMP}.npz"
        MODEL=output/experiments/sweep/scaled_mlp6_barrier_h232.pt
        LABEL=bp ;;
  enet) INPUT="$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full/${TIMESTAMP}.npz"
        MODEL=output/experiments/sweep/scaled_mlp6_enet_h232.pt
        LABEL=enet ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac

if [ ! -f "$INPUT" ]; then
  echo "missing full-disk input: $INPUT" >&2
  exit 1
fi

ROOT="$SCRATCH/dem/visuals"
EXPORT="$ROOT/exports/${LABEL}_mlp6_h232/dem_${TIMESTAMP}.npz"
SOLVER_OUT="$ROOT/assets/${LABEL}_solver/${TIMESTAMP}"
MODEL_OUT="$ROOT/assets/${LABEL}_mlp6_h232/${TIMESTAMP}"
mkdir -p "$(dirname "$EXPORT")"

if [ ! -f "$EXPORT" ] || [ "${FORCE_EXPORT:-0}" = 1 ]; then
  python3 -u experiments/export_mlp6_full_disk.py \
    --model "$MODEL" --input_npz "$INPUT" --output "$EXPORT"
else
  echo "reusing existing DEM export: $EXPORT"
fi
python3 -u experiments/render_full_disk_visuals.py \
  --input "$EXPORT" --source reference --output "$SOLVER_OUT"
python3 -u experiments/render_full_disk_visuals.py \
  --input "$EXPORT" --source model --output "$MODEL_OUT"

echo "solver assets: $SOLVER_OUT"
echo "model assets:  $MODEL_OUT"
