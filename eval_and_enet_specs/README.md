# Bright/Quiet thresholds + ElasticNet settings

## Bright/Quiet thresholds and evaluation mask

Per-channel top-5% AIA intensity threshold (DN/s):

| Channel | 94 | 131 | 171 | 193 | 211 | 335 |
|---|---|---|---|---|---|---|
| Threshold (DN/s) | 3.1212 | 16.9986 | 448.8948 | 498.3444 | 190.7388 | 10.3326 |

- **Bright** = any AIA channel >= its threshold above, AND pixel valid.
- **Quiet** = NOT bright, AND pixel valid.
- **Full** = all valid pixels.

## ElasticNet hyperparameters / objective

```
argmin_{x>=0} 1/(2N) ||Dx - y||^2 + alpha * l1_ratio * ||x||_1 + alpha * (1 - l1_ratio) * 0.5 * ||x||^2
```

`sklearn.linear_model.ElasticNet`, `fit_intercept=False`, `positive=True`, `max_iter=10000`, `alpha=0.001`, `l1_ratio=0.5`.

Objective code: `solveElasticNet()` in `fullBP.py` (repo root). Exact CLI invocation: `submit_enet_aia_hofdeconv_full.py`.

The supplied generation script creates five versions per timestamp (the clean
sample plus four photon-noise realizations). This accounts exactly for the
shared test set containing 48,960 blocks rather than the 9,792 clean-only
blocks in the earlier staged test split.

## Per-pixel error statistics (p50/p90/p99/p99.9, worst-1% squared-error share)

Not precomputed. Best checkpoints to compute them from: `checkpoints/model_best_bp.pth`, `checkpoints/model_best_en.pth`.

The fixed thresholds are also available in machine-readable form as
`aia_thresholds.json`. From the repository root on Torch, evaluate both
supervised checkpoints on the shared test sets with:

```bash
sbatch --export=ALL,TRACK=bp experiments/job_eval_supervised_paper_test.sbatch
sbatch --export=ALL,TRACK=enet experiments/job_eval_supervised_paper_test.sbatch
```

The jobs write `output/experiments/paper_eval/{bp,enet}_supervised_matched.json`.
Each Full/Bright/Quiet row includes the paper metrics plus per-pixel DEM SSE
p50/p90/p99/p99.9 and the share of total SSE contributed by the worst 1%.

For a quick two-block checkpoint/loading test, add `MAX_BLOCKS=2` to the
`--export` list. The supplied table script evaluates `test_x.zarr` and
`test_y.zarr`; use those files for comparison with the reported Table-1 test
numbers.
