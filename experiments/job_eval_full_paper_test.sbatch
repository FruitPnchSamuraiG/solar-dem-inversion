#!/bin/bash
#SBATCH --job-name=dem_paper_eval
#SBATCH --output=logs/eval/paper_eval_%j.log
#SBATCH --error=logs/eval/paper_eval_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Submit one job per solver track, for example:
# sbatch --export=TRACK=bp experiments/job_eval_full_paper_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/paper_eval

case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   DATA=/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS
        MODEL=output/experiments/sweep/scaled_mlp6_barrier_h232.pt ;;
  enet) DATA=/projects/rps/dff6142/fouheylab/solar_dem/enet_AIA_hofdeconv_full_DS
        MODEL=output/experiments/sweep/scaled_mlp6_enet_h232.pt ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac

python3 -u experiments/eval_full_paper_test.py \
  --model "$MODEL" --data "$DATA" --reference "$TRACK" \
  --output "output/experiments/paper_eval/${TRACK}_h232_full.json"
