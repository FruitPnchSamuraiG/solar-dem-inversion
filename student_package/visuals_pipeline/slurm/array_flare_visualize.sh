#!/bin/bash
#SBATCH --job-name=flare_predict_vis
#SBATCH --output=logs/flare_predict_vis_%a.out
#SBATCH --error=logs/flare_predict_vis_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --array=1-PLACEHOLDER%10
#SBATCH --mail-user=YOUR_NETID@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# flare prediction and visualization array job
# usage: sbatch --array=1-N array_flare_predict_vis.sh models.txt timestamps.txt
# where N = (number of models) * (number of timestamps)

# parse command line arguments
MODELS_FILE=$1
TIMESTAMPS_FILE=$2

if [ -z "$MODELS_FILE" ] || [ -z "$TIMESTAMPS_FILE" ]; then
    echo "usage: sbatch array_flare_predict_vis.sh <models_file> <timestamps_file>"
    echo "example: sbatch array_flare_predict_vis.sh flare_models.txt flare_timestamps.txt"
    exit 1
fi

# check if files exist
if [ ! -f "$MODELS_FILE" ]; then
    echo "error: models file '$MODELS_FILE' not found"
    exit 1
fi

if [ ! -f "$TIMESTAMPS_FILE" ]; then
    echo "error: timestamps file '$TIMESTAMPS_FILE' not found"
    exit 1
fi

# read models and timestamps into arrays
readarray -t MODELS < "$MODELS_FILE"
readarray -t TIMESTAMPS < "$TIMESTAMPS_FILE"

# calculate total number of combinations
NUM_MODELS=${#MODELS[@]}
NUM_TIMESTAMPS=${#TIMESTAMPS[@]}
TOTAL_JOBS=$((NUM_MODELS * NUM_TIMESTAMPS))

echo "found $NUM_MODELS models and $NUM_TIMESTAMPS timestamps"
echo "total combinations: $TOTAL_JOBS"

# calculate which model and timestamp to use for this array task
# SLURM_ARRAY_TASK_ID is 1-indexed
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_INDEX=$((TASK_INDEX / NUM_TIMESTAMPS))
TIMESTAMP_INDEX=$((TASK_INDEX % NUM_TIMESTAMPS))

# get the specific model and timestamp for this task
MODEL_NAME="${MODELS[$MODEL_INDEX]}"
TIMESTAMP="${TIMESTAMPS[$TIMESTAMP_INDEX]}"

echo "array task $SLURM_ARRAY_TASK_ID: processing model '$MODEL_NAME' with timestamp '$TIMESTAMP'"

# set up environment
module purge

# NOTE: adjust REPO_DIR and PKG_DIR to your own environment/repo checkout.
REPO_DIR=/scratch/$USER/workspace/dem/demdemo
PKG_DIR=$REPO_DIR/student_package/visuals_pipeline

# change to package directory
cd "$PKG_DIR"

# determine input path based on model name (outside container)
if [[ $MODEL_NAME == *"_lp_"* ]] || [[ $MODEL_NAME == *"lp"* ]]; then
    # further check for logbins/sqrtbins in name
    if [[ $MODEL_NAME == *"logbins"* ]]; then
        INPUT_PATH="xrtData_lp_logbins"
    elif [[ $MODEL_NAME == *"sqrtbins"* ]]; then
        INPUT_PATH="xrtData_lp_sqrtbins"
    else
        INPUT_PATH="xrtData_lp_full"
    fi
elif [[ $MODEL_NAME == *"_qp_"* ]] || [[ $MODEL_NAME == *"qp"* ]]; then
    # further check for logbins/sqrtbins in name
    if [[ $MODEL_NAME == *"logbins"* ]]; then
        INPUT_PATH="xrtData_qp_logbins"
    elif [[ $MODEL_NAME == *"sqrtbins"* ]]; then
        INPUT_PATH="xrtData_qp_sqrtbins"
    else
        INPUT_PATH="xrtData_qp_full"
    fi
else
    echo "warning: cannot determine input path from model name. using default lp."
    INPUT_PATH="xrtData_lp_full"
fi

echo "using input path: $INPUT_PATH"

# run the prediction and visualization
echo "starting prediction and visualization..."
echo "time: $(date)"

# use singularity container with DEM environment
  singularity exec --overlay /scratch/$USER/workspace/dem/overlay-25GB-500K.ext3:ro \
    $CONTAINER_IMAGES_FOLDER/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; conda activate dem_new; \
    cd $REPO_DIR; \

        # run single-case prediction and visualization
        python3 -u $PKG_DIR/predictVisualFlare.py \\
            --model_name \"$MODEL_NAME\" \\
            --input_path \"$INPUT_PATH\" \\
            --timestamp \"$TIMESTAMP\" \\
            --skip_existing
    "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "successfully completed prediction and visualization for model '$MODEL_NAME' with timestamp '$TIMESTAMP'"
else
    echo "error: prediction and visualization failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "finished at: $(date)"
