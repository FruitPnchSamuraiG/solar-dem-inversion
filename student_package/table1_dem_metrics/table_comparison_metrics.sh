#!/bin/bash
#SBATCH --job-name=table_comparison
#SBATCH --output=logs/table_comparison_%x_%j.out
#SBATCH --error=logs/table_comparison_%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=l40s|h100|h200|a100
#SBATCH --account=torch_pr_41_general
#SBATCH --mail-user=YOUR_NETID@nyu.edu
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp,REFERENCE="Basis Pursuit (BP)" \
#       table_comparison_metrics.sh
#
#   Optionally set THRESHOLDS=<path to a aia_thresholds.json with a "test" key
#   already computed on THIS --data's test split> to skip the in-script
#   threshold pass. If unset, thresholds are computed fresh from DATA.
#
# Runs compute_paper_table_metrics.py for one (model, reference-solver
# dataset) pair and writes table_comparison_<VARIANT>.json.
#
# NOTE: adjust REPO_DIR / overlay / container paths below to your own
# environment before running (this SLURM header is copied from the lab
# cluster setup described in the top-level demdemo/CLAUDE.md).

module purge

for v in MODEL DATA VARIANT REFERENCE; do
  if [ -z "${!v}" ]; then
    echo "$v environment variable is required"
    exit 1
  fi
done

REPO_DIR=/scratch/$USER/workspace/dem/demdemo
OUTDIR=$REPO_DIR/paperplots/cache
OUTPUT=$OUTDIR/table_comparison_${VARIANT}.json

mkdir -p "$OUTDIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "VARIANT=$VARIANT  REFERENCE=$REFERENCE"
echo "OUTPUT=$OUTPUT"
echo "THRESHOLDS=${THRESHOLDS:-<computed fresh>}"

THR_ARG=""
if [ -n "$THRESHOLDS" ]; then
  THR_ARG="--thresholds_json \"$THRESHOLDS\""
fi

singularity exec --nv \
  --overlay /scratch/$USER/workspace/dem/overlay-25GB-500K.ext3:ro \
  $CONTAINER_IMAGES_FOLDER/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    conda activate dem_new
    cd $REPO_DIR
    python3 -u student_package/table1_dem_metrics/compute_paper_table_metrics.py \
      --model \"$MODEL\" \
      --data \"$DATA\" \
      --rdata RData.npz \
      --variant \"$VARIANT\" \
      --reference \"$REFERENCE\" \
      --batch_size 16 \
      $THR_ARG \
      --output \"$OUTPUT\"
  "
