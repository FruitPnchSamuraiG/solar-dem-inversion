# 27: Exact solve at the hardest pixels, at test time (14 Nov 2026)

The network's DEM error against the solver is concentrated at flare cores: on the shared test,
pixels at 32x the Bright threshold or above are 0.010% of pixels and 91% of the squared DEM
error. Our loss's own per-pixel optimum is far better there than the network, so the gap is
amortization (one network for a billion pixels), not the objective.

Here, at test time, pixels flagged by their readings alone (T x the Bright threshold or above)
get the exact solution of their own problem, the one the network is trained on: least total
emission with every channel inside its band (reading +/- 1.4 sigma; a zeroed channel keeps
its band around zero), with the solver's band ladder (tight band if any DEM fits, else 3x,
else 5x) decided by our own feasibility check. No solver DEM is used; the check picked the same
rung as the solver at all but 9 of 3.8 million flagged pixels (val and test). Run to
convergence, test-time optimization of our loss reaches exactly this solution, so it is computed
directly as the small linear program it is (`src/refine.py`, HiGHS, about 4 ms per pixel per
CPU core). Every other pixel keeps the network's answer.

Model: the 1.5M full recipe (multipliers at mu 10, zero band kept, BP's widened band at the
pixels it cannot fit; `bp_h680_keep_mm10_relax`). Scored by `experiments/full_validation.py
--refine 32,10,3.2` on every pixel of our staged split (160 million pixels, 153 images each).
The shared test cannot be refined: it has no noise estimates. Supervised 10.1M on the same
pixels: `results/plots/25_followup_20261109/bp_full_validation_sup10m_{val,test}.json`.

## Test (`bp_full_validation_refine_test.json`)

| | DEM MSE | flare cores | DEM MAE | median pixel | AIA MSE | wild pixels |
|---|---|---|---|---|---|---|
| network alone | 4.46 | 36,020 | 0.0654 | 0.0127 | 24,880 | 40 |
| + solve flare cores (>= 32x, 0.011% of pixels) | 0.615 | 665 | 0.0641 | 0.0127 | 24,000 | 37 |
| + solve >= 10x (0.10%) | 0.607 | 665 | 0.0635 | 0.0127 | 23,990 | 37 |
| + solve >= 3.2x (1.3%) | **0.502** | 665 | **0.0607** | **0.0125** | 20,660 | 7 |
| supervised 10.1M | 2.06 | 17,380 | 0.116 | 0.1245 | **68** | 0 |

By pixel group (test, solve >= 3.2x against supervised, DEM MSE): pixels the solver fits at the
tight band (91%) 0.118 against 1.48; quiet (88%) 0.030 against 0.056; relaxed (9%) 4.59 against
8.20; bright 1-32x (12%) 3.25 against 0.96 (median there 0.73 against 2.21, so the mean is
pulled up by a few pixels). AIA MSE at tight-band pixels 52.5 against 44.4; the large overall
AIA MSE comes from relaxed bright pixels, whose band is 3x or 5x wide.

## Validation (`bp_full_validation_refine_val.json`)

| | DEM MSE | flare cores | DEM MAE | AIA MSE | wild pixels |
|---|---|---|---|---|---|
| network alone | 6.07 | 67,670 | 0.0627 | 43,630 | 191 |
| + solve >= 32x | 0.302 | 842 | 0.0599 | 5,523 | 126 |
| + solve >= 10x | 0.279 | 842 | 0.0594 | 5,438 | 123 |
| + solve >= 3.2x | **0.260** | 842 | **0.0573** | 5,108 | 109 |
| supervised 10.1M | 1.96 | 21,080 | 0.110 | **90** | 0 |

Validation picks 3.2x (lowest DEM MSE, MAE, AIA MSE and wild count); 32x already gives most of
the gain.

## Cost per full-disk image (2048 x 2048 grid; the split holds 64 of 256 blocks per image)

| threshold | pixels per image | time on 15 CPU cores |
|---|---|---|
| >= 32x | ~460 | ~0.1 s |
| >= 10x | ~4,300 | ~1.3 s |
| >= 3.2x | ~56,000 | ~16 s |

For scale: the network takes 0.3 s per image on one GPU (36 s on 16 CPU cores); the solver on
every pixel about 220 s on 16 cores.

## Reading

- At the flagged pixels the answer is the exact optimum of the solver's own problem, computed
  by us from the readings: a hybrid (network everywhere, exact solve at the hardest 0.01% to
  1%), still with no labels. The remaining flare-core error (665) is where that optimum and the
  solver's stored DEM differ (equally good DEMs, total emission ratio 0.999).
- The typical pixel is unchanged (it was already 10x better than the supervised model's).
- Pixels the solver could not fit at the tight band (relaxed or unsolved) are reported as a
  separate set, as agreed with the advisor; the headline numbers are the tight-band pixels.

## Noise for the shared test (`shared_noise_fit.json`)

The shared test stores readings and the solver's DEMs but no noise estimates. The solver's noise
is aiapy's `estimate_error` on the counts floored at 0.5, divided by the image's scale factor
(exposure time x degradation; `src/aia_noise.py`), which reproduces our staged noise to 1e-7.
The shared test was staged with a slightly different calibration (none of its blocks equals one
of ours), so its scale factor is ours times a per-channel ratio, fitted so that the solver's own
clean DEMs sit on their band edges (`experiments/fit_shared_noise.py`; the clean solve is slot 0
of the five targets per image, the other four are re-solves of noisy copies):

| channel | 94 | 131 | 171 | 193 | 211 | 335 |
|---|---|---|---|---|---|---|
| scale ratio | 1.010 | 1.046 | 0.970 | 1.206 | 1.085 | 1.266 |

With these, 100% of the clean-slot pixels (9.3 million, 2,350 of them flare cores) have the
solver's DEM inside the rebuilt band at every brightness (worst channel at most 1.004 half-widths,
the solver's 12-bit DEM rounding); with our own scale factors only 29% at quiet pixels and 0.6% at
flare cores. Shared-test scoring of the refinement: pending (jobs 23154671, 23154672).
