#!/bin/bash
#SBATCH --job-name=stage_hof
#SBATCH --account=torch_pr_41_general
#SBATCH --output=logs/stage_hof_%j.log
#SBATCH --error=logs/stage_hof_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00

set -e
DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "staging config: $CONFIG"
echo "start: $(date)"

apptainer exec \
  --overlay /scratch/$USER/workspace/dem/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate dem_new && python3 $DATASET_DIR/stage_hofdeconv_full.py --config $CONFIG"

echo "done: $(date)"
