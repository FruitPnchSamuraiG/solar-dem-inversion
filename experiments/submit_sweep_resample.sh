#!/bin/bash
# Fresh-pixel width sweep + CNN comparison (job_sweep_resample.sbatch), with the
# bright-failure diagnostic and AIA fit chained after each mlp6 task, and a
# CNN-vs-MLP test-split evaluation once the whole array is done.
#
#   bash experiments/submit_sweep_resample.sh
set -euo pipefail
mkdir -p logs/sweep logs/eval
arr=$(sbatch --parsable experiments/job_sweep_resample.sbatch)
echo "array $arr"
WIDTHS=(680 480 336 160 108 72 48)
for i in $(seq 0 13); do
  if (( i < 7 )); then T=bp; L=barrier; W=${WIDTHS[$i]}; else T=enet; L=enet; W=${WIDTHS[$((i - 7))]}; fi
  if [ "$W" = 680 ]; then CK=output/experiments/resample/scaled_mlp6_${L}_resample.pt
  else CK=output/experiments/resample/scaled_mlp6_${L}_h${W}_resample.pt; fi
  d=$(sbatch --parsable --dependency=afterok:${arr}_$i \
      --export=ALL,TRACK=$T,MODEL_PATH=$CK,TAG=resample_h$W experiments/job_diagnose_bright_failures.sbatch)
  a=$(sbatch --parsable --dependency=afterok:${arr}_$i \
      --export=ALL,TRACK=$T,LF_MODEL=$CK,TAG=resample_h$W experiments/job_aia_fit_by_brightness.sbatch)
  echo "task $i ($T h$W): diagnostic $d, aia fit $a"
done
e=$(sbatch --parsable --dependency=afterany:$arr --job-name=dem_eval_rs \
    --account=torch_pr_41_tandon_advanced --gres=gpu:1 --constraint='l40s|a100|h100|h200' \
    --cpus-per-task=8 --mem=64G --time=01:30:00 \
    --output=logs/eval/eval_rs_%j.log --error=logs/eval/eval_rs_%j.err \
    --wrap="cd $PWD && . ./env.sh && python3 -u experiments/eval_scaled.py \
      --bp_root \$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
      --enet_root \$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS \
      --ckpt_dir output/experiments/resample --ckpt_suffix _resample \
      --out_dir output/experiments/eval_scaled_resample")
echo "cnn-vs-mlp eval $e"
