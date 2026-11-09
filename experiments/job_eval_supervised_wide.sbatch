#!/bin/bash
#SBATCH --job-name=dem_sup_wide
#SBATCH --output=logs/eval/sup_wide_%j.log
#SBATCH --error=logs/eval/sup_wide_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:59:00
#SBATCH --mail-type=FAIL

# The supervised paper models (paper_model/checkpoints, 10.13M parameters) through the
# same shared-test evaluation as our models, on the GPU, with the output clipped at zero
# as the paper models' own metrics do (SUP_CLAMP=0 keeps the raw output). STEP picks one
# part so each job stays under 2 h:
#   STEP=diag    bright-failure diagnostic of the supervised model (DEM metrics)
#   STEP=aia     AIA fit: our final model and the supervised one on the same pixels
#   STEP=table1  the supervised side's own Table-1 script on its raw output
#   sbatch --export=ALL,TRACK=bp,STEP=diag experiments/job_eval_supervised_wide.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics output/experiments/paper_eval
export DEM_EVAL_DEVICE=cuda SUP_DEVICE=cuda SUP_CLAMP=${SUP_CLAMP:-1}
WORKERS=8
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   SUP=${SUP_PATH:-paper_model/checkpoints/BP_5_Wide_10.1M.pth}
        LF=output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt
        REF="Basis Pursuit (BP)" ;;
  enet) SUP=${SUP_PATH:-paper_model/checkpoints/EN_2_Wide_10.1M.pth}
        LF=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
        REF="Elastic Net" ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
TAG=${TAG:-sup10m}
DATA=/projects/rps/dff6142/fouheylab/solar_dem/${TRACK}_AIA_hofdeconv_full_DS
for f in "$SUP" "$LF"; do [ -f "$f" ] || { echo "missing: $f" >&2; exit 3; }; done
echo "track=$TRACK step=${STEP:?} sup=$SUP clamp=$SUP_CLAMP"
case "$STEP" in
  diag)   python3 -u experiments/diagnose_bright_failures.py --model "$SUP" --data "$DATA" \
            --reference "$TRACK" --workers "$WORKERS" --threads-per-worker "$THREADS" \
            --output "output/experiments/diagnostics/${TRACK}_bright_failure_supervised_${TAG}.json" ;;
  aia)    python3 -u experiments/aia_fit_by_brightness.py --label-free "$LF" --supervised "$SUP" \
            --reference "$TRACK" --data "$DATA" --workers "$WORKERS" --threads-per-worker "$THREADS" \
            --samples 60 --output "output/experiments/diagnostics/${TRACK}_aia_fit_${TAG}.json" ;;
  table1) python3 -u student_package/table1_dem_metrics/compute_paper_table_metrics.py \
            --model "$SUP" --data "$DATA" --rdata RData.npz --variant "supervised_${TRACK}_wide" \
            --reference "$REF" --thresholds_json eval_and_enet_specs/aia_thresholds.json \
            --output "output/experiments/paper_eval/${TRACK}_supervised_${TAG}_table1.json" ;;
  *) echo "STEP must be diag, aia or table1" >&2; exit 2 ;;
esac
