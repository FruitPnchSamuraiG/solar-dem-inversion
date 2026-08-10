#!/bin/bash
#SBATCH --job-name=dem_jpdf_cache
#SBATCH --output=logs/dem_jpdf_cache_%x_%j.out
#SBATCH --error=logs/dem_jpdf_cache_%x_%j.err
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
#   sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp \
#       dem_jpdf_cache.sh
#
# Runs compute_dem_jpdf_cache.py for one (model, reference-solver dataset)
# pair and writes cache/dem_jpdf_<VARIANT>.npz, consumed by dem_jpdf_hof.py.
#
# NOTE: adjust REPO_DIR / overlay / container paths below to your own
# environment before running (this SLURM header is copied from the lab
# cluster setup described in the top-level demdemo/CLAUDE.md).

module purge

for v in MODEL DATA VARIANT; do
  if [ -z "${!v}" ]; then
    echo "$v environment variable is required"
    exit 1
  fi
done

REPO_DIR=/scratch/$USER/workspace/dem/demdemo
PKG_DIR=$REPO_DIR/student_package/fig_dem_jpdf
OUTDIR=$PKG_DIR/cache
OUTPUT=$OUTDIR/dem_jpdf_${VARIANT}.npz

mkdir -p "$OUTDIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "VARIANT=$VARIANT"
echo "OUTPUT=$OUTPUT"

singularity exec --nv \
  --overlay /scratch/$USER/workspace/dem/overlay-25GB-500K.ext3:ro \
  $CONTAINER_IMAGES_FOLDER/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    conda activate dem_new
    cd $REPO_DIR
    python3 -u $PKG_DIR/compute_dem_jpdf_cache.py \
      --model \"$MODEL\" \
      --data \"$DATA\" \
      --variant \"$VARIANT\" \
      --batch_size 16 \
      --output \"$OUTPUT\"
  "
