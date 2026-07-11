#!/bin/bash
#SBATCH --job-name=dem_loo
#SBATCH --account=torch_pr_41_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/hsr3649/dem/runs/%x_%j.out

cd ~/projects/dem
source env.sh

python experiments/train_leave_one_out.py \
    --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                ./data/20131113_0908 ./data/20140910_1731 \
    --crop 1800,1800,128,128
