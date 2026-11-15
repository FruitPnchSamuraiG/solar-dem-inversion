# 27: Test-time exact solve at the brightest pixels: WITHDRAWN (14 Nov 2026)

**Not a result about the method; do not use these numbers.** At pixels flagged as bright, this
replaced the network's DEM with the optimum of the pixel's own problem (least total emission with
every channel inside its band), computed with a linear-program solver. That problem is basis
pursuit's own, so this runs the solver's procedure at those pixels and then scores the answer
against the solver's DEM: the agreement is circular, and calling a solver at test time defeats
the purpose of the work (a network that replaces the expensive solver). The tables that were here
compared such a hybrid with the supervised network; they say nothing about either network.

The network-only comparison stands (`results/plots/26_constrained_theory_20261112/`): on the
shared test the 1.5M label-free network has the lower median pixel error (0.027 against 0.089),
DEM MAE (0.070 against 0.085), W1 (0.066 against 0.113) and total emission error (10.8% against
13.1%), and the higher DEM MSE (0.976 against 0.387), which comes from flare cores: pixels at 32x
the Bright threshold or above are 0.010% of pixels and 91% of the squared DEM error. Closing that
gap is open work for the network itself.

## What remains useful here

**Noise for the shared test** (`shared_noise_fit.json`, `src/aia_noise.py`,
`experiments/fit_shared_noise.py`, `experiments/check_shared_noise.py`). The shared test stores
readings and the solver's DEMs but no noise estimates, which any band-based measure on it needs
(for example the network's band violations). The solver's noise is aiapy's `estimate_error` on the
counts floored at 0.5, divided by the image's scale factor (exposure time x degradation), which
reproduces our staged noise to 1e-7. The shared test was staged with a slightly different
calibration, so its scale factor is ours times a per-channel ratio, fitted so that the solver's
own clean DEMs (slot 0 of the five targets per image) sit on their band edges:

| channel | 94 | 131 | 171 | 193 | 211 | 335 |
|---|---|---|---|---|---|---|
| scale ratio | 1.010 | 1.046 | 0.970 | 1.206 | 1.085 | 1.266 |

With these, the solver's clean DEMs are inside the rebuilt band at 100% of 9.3 million pixels at
every brightness (within its 12-bit DEM rounding, about 2e-3 half-widths).

Files kept for the record: `bp_full_validation_refine_{val,test}.json`, `shared_runs.json`,
`shared_summary.{json,txt}` and the `*_refine*.json` diagnostics (the withdrawn comparison).
