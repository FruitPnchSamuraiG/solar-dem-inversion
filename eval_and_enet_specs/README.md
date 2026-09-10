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

## Per-pixel error statistics (p50/p90/p99/p99.9, worst-1% squared-error share)

Not precomputed. Best checkpoints to compute them from: `checkpoints/model_best_bp.pth`, `checkpoints/model_best_en.pth`.
