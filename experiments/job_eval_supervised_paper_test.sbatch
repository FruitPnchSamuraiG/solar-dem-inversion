#!/bin/bash
#SBATCH --job-name=dem_sup_eval
#SBATCH --output=logs/eval/supervised_eval_%j.log
#SBATCH --error=logs/eval/supervised_eval_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Submit one job per track:
#   sbatch --export=ALL,TRACK=bp experiments/job_eval_supervised_paper_test.sbatch
#   sbatch --export=ALL,TRACK=enet experiments/job_eval_supervised_paper_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/paper_eval

case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)
    DATA=/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS
    MODEL=eval_and_enet_specs/checkpoints/model_best_bp.pth
    VARIANT=supervised_bp
    REFERENCE="Basis Pursuit (BP)"
    ;;
  enet)
    DATA=/projects/rps/dff6142/fouheylab/solar_dem/enet_AIA_hofdeconv_full_DS
    MODEL=eval_and_enet_specs/checkpoints/model_best_en.pth
    VARIANT=supervised_enet
    REFERENCE="Elastic Net"
    ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac

OUTPUT_TAG=${OUTPUT_TAG:-matched}
ARGS=(
  --model "$MODEL"
  --data "$DATA"
  --rdata RData.npz
  --variant "$VARIANT"
  --reference "$REFERENCE"
  --thresholds_json eval_and_enet_specs/aia_thresholds.json
  --output "output/experiments/paper_eval/${TRACK}_supervised_${OUTPUT_TAG}.json"
)
if [ -n "${MAX_BLOCKS:-}" ]; then
  ARGS+=(--n_samples "$MAX_BLOCKS")
fi

python3 -u student_package/table1_dem_metrics/compute_paper_table_metrics.py "${ARGS[@]}"
