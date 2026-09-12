#!/bin/bash
#SBATCH --job-name=dem_sup_cpu
#SBATCH --output=logs/eval/supervised_cpu_%j.log
#SBATCH --error=logs/eval/supervised_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_general

# CPU alternative for the short supervised Table-1 verification runs.
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUTPUT_TAG=${OUTPUT_TAG:-emfix}

python3 -u student_package/table1_dem_metrics/compute_paper_table_metrics.py \
  --model "$MODEL" \
  --data "$DATA" \
  --rdata RData.npz \
  --variant "$VARIANT" \
  --reference "$REFERENCE" \
  --thresholds_json eval_and_enet_specs/aia_thresholds.json \
  --device cpu \
  --batch_size "${BATCH_SIZE:-4}" \
  --output "output/experiments/paper_eval/${TRACK}_supervised_${OUTPUT_TAG}.json"
