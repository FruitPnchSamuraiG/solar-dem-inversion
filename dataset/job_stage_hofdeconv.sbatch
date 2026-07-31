#!/bin/bash
#SBATCH --job-name=stage_hof
#SBATCH --output=logs/stage/stage_%j.log
#SBATCH --error=logs/stage/stage_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#
# Stages the generated solver outputs into zarr for training. Both configs run
# sequentially in one job: they read and write the same filesystem, so running
# them concurrently just makes them contend for it.
#
#   mkdir -p logs/stage
#   sbatch --account=<acct> --mail-user=<you>@nyu.edu dataset/job_stage_hofdeconv.sbatch
#
# Memory: each worker holds one decompressed DEM cube (26x2048^2 f32 ~ 436 MB)
# plus AIA obs and errors (6x4096^2 f32 each ~ 805 MB together), so roughly
# 1.3 GB per worker. STAGE_WORKERS must stay under mem/1.3GB or the job OOMs
# partway through and leaves a half-written zarr.

set -e

if [ -n "$SLURM_SCRIPT" ]; then
  DATASET_DIR="$(cd "$(dirname "$SLURM_SCRIPT")" && pwd)"
elif [ -d "$SLURM_SUBMIT_DIR/dataset" ]; then
  DATASET_DIR="$SLURM_SUBMIT_DIR/dataset"
else
  DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_DIR="$(dirname "$DATASET_DIR")"
echo "REPO_DIR=$REPO_DIR"

cd "$REPO_DIR"
source env.sh

export STAGE_WORKERS=${STAGE_WORKERS:-32}
echo "STAGE_WORKERS=$STAGE_WORKERS"

for CONFIG in ${CONFIGS:-lp_AIA_hofdeconv_full elasticnet_AIA_hofdeconv_full}; do
  echo "=============================================================="
  echo "staging $CONFIG  ($(date))"
  echo "=============================================================="
  python3 -u dataset/stage_hofdeconv_full.py --config "$CONFIG"
done

echo "all staging complete ($(date))"
