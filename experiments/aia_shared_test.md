# Shared-test AIA reconstruction evaluation

`job_eval_aia_shared_cpu.sbatch` evaluates both models together for one track.
It does not train anything or regenerate viewer images. Predictions are projected
through `RData.npz`'s six-channel response, using the same 18-bin / 1e26 scaling
as the viewer. Metrics accumulate in float64, weighted by channel-pixel counts,
not by taking an unweighted average of image averages.

All 48,960 shared-test blocks are retained, including repeated clean observations
for five solver targets. The AIA grid is subsampled by stride two to align with
the reference DEM grid. Within each track, both models use the same mask: finite
reference DEM, observed AIA, and both reconstructions. BP and ENet reference
masks can differ. Full/Bright/Quiet use the supplied fixed thresholds. Outputs
include pooled and per-channel MAE/MSE and counts. They are not whole-disk viewer
frame metrics or 48,960 independent observations.

On Torch, first run two-block smoke checks (create the log directory before sbatch):

```bash
cd ~/projects/dem
git pull --ff-only
mkdir -p logs/eval
sbatch --export=ALL,TRACK=bp,MAX_BLOCKS=2,OUTPUT_TAG=smoke experiments/job_eval_aia_shared_cpu.sbatch
sbatch --export=ALL,TRACK=enet,MAX_BLOCKS=2,OUTPUT_TAG=smoke experiments/job_eval_aia_shared_cpu.sbatch
```

After both succeed, submit the full jobs (MAX_BLOCKS must not be set in the shell):

```bash
unset MAX_BLOCKS
sbatch --export=ALL,TRACK=bp experiments/job_eval_aia_shared_cpu.sbatch
sbatch --export=ALL,TRACK=enet experiments/job_eval_aia_shared_cpu.sbatch
```

Results: `output/experiments/paper_eval/{bp,enet}_aia_shared_test.json`.

The complete runs finished successfully on 2026-09-19: BP job `18025355` in
1:12:18 and ENet job `18025356` in 1:06:29. Both evaluated all 48,960 blocks,
reported `complete_test: true`, and excluded zero pixels for nonfinite model
predictions. Pooled Full-set results incorporated into the viewer are:

| Track | Model | AIA MAE | AIA MSE |
|---|---|---:|---:|
| BP | Label-free MLP6 | 4.3153 | 82.5570 |
| BP | Supervised | 2.9461 | 156.6699 |
| ENet | Label-free MLP6 | 0.6692 | 193.3688 |
| ENet | Supervised | 6.3214 | 597.8458 |

The JSON files additionally contain Bright/Quiet and per-channel values. MSE
and MAE rankings can disagree because squared error gives much more weight to
large bright-pixel residuals. No AIA percentile distribution was computed.
