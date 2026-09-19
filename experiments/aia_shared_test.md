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
Check completion, `complete_test: true`, matching model counts, and nonfinite
exclusion counts before publishing. CPU runtime is not yet measured for this
paired evaluator; the initial limit is four hours. No full-test values are
currently published: the viewer retains explicitly selected-frame diagnostics
until these results have been validated and incorporated.
