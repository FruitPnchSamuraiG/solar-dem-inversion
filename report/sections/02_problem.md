## 2. Problem setting

### 2.1 Observations and the DEM

The data are SDO/AIA images in six EUV channels (94, 131, 171, 193, 211 and 335 Å), deconvolved for the instrument's point-spread function (with the revised functions of Hofmeister et al. 2025) and corrected for exposure time and degradation, so a pixel's six values o_c are intensities in DN/s. DEMs are computed on a 2048 × 2048 grid, DEM pixel (i, j) taking the AIA values at pixel (2i, 2j).

A DEM has 18 values, on temperature bins from logT 5.5 to 7.2 in steps of 0.1 dex (about 0.3 to 16 MK). Each is the emission measure in its bin in units of 10^26 cm^-5, which this report calls one unit of plasma. (The solver files pad the grid to logT 8.0 with zeros.) The response matrix R (6 × 18) gives each channel's DN/s per unit of plasma in each bin, so the predicted intensities are R times the DEM. With six measurements and 18 unknowns, many DEMs fit within the noise; the solver's regulariser picks one.

### 2.2 The basis

Neither solver solves for the 18 bins directly. Both solve for 54 non-negative weights w on a fixed basis B (18 × 54), with DEM = Bw and predicted intensities RBw. At each bin B has a spike (one unit in that bin and nothing elsewhere) and two Gaussians in logT centred on the bin, exp(−(ΔlogT/a)²) with a = 0.1 and 0.2 dex and a peak of 1, values below 0.04 of the peak set to zero (Table 2.1).

Table 2.1. The three basis components at a bin away from the ends of the grid.

| Component | Bins covered | Plasma per unit of weight |
| --- | --- | --- |
| Spike | 1 | 1 |
| Gaussian, a = 0.1 dex | 3 | 1.74 = 1 + 2 × 0.368 |
| Gaussian, a = 0.2 dex | 7 | 3.50 = 1 + 2 × (0.779 + 0.368 + 0.105) |

The weights are therefore not the plasma. A unit of weight on a spike is one unit of plasma in one bin, but a unit on the 0.1 dex Gaussian spreads 1.74 units over three bins, and a unit on the 0.2 dex Gaussian 3.5 units over seven. So the sum of the 54 weights, which BP minimises (our BP loss's sparsity term), is not the plasma summed over the 18 bins: a DEM shaped like one 0.2 dex Gaussian costs 1 written as that Gaussian and 3.5 written as seven spikes. Responses differ the same way: per unit of weight, the 0.2 dex Gaussian at logT 6.8 gives 94 Å 0.67 DN/s, the spike at logT 6.9 only 0.266.

Our network's basis has this truncation; the solver runs that made the reference DEMs used `--notrunc`, which keeps the tails, so their Gaussians carry about 1.77 and 3.54 units per unit of weight. Both generation commands we have pass `--notrunc` (ours for BP, the supervised side's for ElasticNet). The difference is at most 2.1% of an element's emission and moves the price bounds of section 3.3 by under 1%; where solver DEMs are rewritten as 54 weights (sections 6.3 and 8.3) the decomposition uses our truncated basis.

### 2.3 Noise

Each pixel and channel has a noise σ_c from AIA's instrument noise model: the `estimate_error` routine of aiapy (Barnes et al. 2020) with the SolarSoft error table, which adds shot (photon), dark, read, quantisation and onboard-compression noise in quadrature, with the photometric-calibration and atomic-data terms off. Shot noise grows as the square root of the counts, so the relative noise falls as a pixel brightens. In the median pixel σ/o is 1.12 in 94 Å, 0.60 in 335 Å and 0.44 in 131 Å, against 0.06 to 0.08 in 171, 193 and 211 Å. The band of channel c is o_c ± tσ_c with t = 1.4.

### 2.4 Basis Pursuit

For each pixel BP solves the linear program

minimise Σ_k w_k over w ≥ 0, subject to o_c − tσ_c ≤ (RBw)_c ≤ o_c + tσ_c for every channel c.

The band is a rule: every DEM inside it is equally acceptable, and the smallest weight sum breaks the tie. As a pixel brightens, its band narrows relative to the signal until it is tighter than the accuracy of R itself and no DEM fits. The solver then retries the pixel with the band three and then five times wider (t = 4.2, then 7.0) and our copy of the solver records the outcome per pixel as tolLevel: 1 inside the band, 3 or 5 after relaxation, 0 if no DEM was found. On one full-disk image, 89.78% of pixels were solved inside the band, 10.04% at three times its width, 0.16% at five times and 0.02% not at all; relaxation was more common on the disk (13.4% against 6.3% off the limb) and in brighter pixels. Section 7.3 returns to these pixels.

### 2.5 ElasticNet

ElasticNet fits the same weights by penalised least squares:

minimise over w ≥ 0: (1/2C) Σ_c [((RBw)_c − o_c) / (tσ_c)]² + αλ Σ_k w_k + ½α(1 − λ) Σ_k w_k²,

with each residual divided by the band's half-width tσ_c (floored at 0.5 DN/s). The supervised side's settings are α = 0.001, λ = 0.5 (the l1_ratio of scikit-learn; Pedregosa et al. 2011) and C = 6, the number of channels, since scikit-learn's ElasticNet divides the squared error by twice the number of rows; the fit is non-negative (positive=True), without intercept, with at most 10,000 iterations. With no band there is no relaxation: every pixel gets a DEM, though our solver runs at these settings logged convergence warnings. The L2 term spreads weight over more components, so ElasticNet DEMs are broader than BP's. Our first ElasticNet runs used α = 1, the code's default; every ElasticNet number in this report uses α = 0.001.

### 2.6 What the channels see

Table 2.2. AIA response to one unit of plasma (10^26 cm^-5) in a single bin, in DN/s.

| Plasma at | 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 335 Å |
| --- | --- | --- | --- | --- | --- | --- |
| 1 to 1.3 MK (warm, logT 6.0 to 6.1) | 0.19 to 0.22 | 0.5 to 1.2 | 36 to 95 | 29 to 55 | 5 to 7 | 0.4 |
| 8 MK (hot, logT 6.9) | 0.27 | 0.40 | 0.05 | 0.34 | 0.08 | 0.05 |

A unit of warm plasma lights 171 and 193 Å by tens of DN/s and 211 Å by several, so those low-noise channels pin it down. A unit of hot plasma lights no channel by more than 0.4 DN/s. At ordinary pixels most of the 94 Å signal comes from warm plasma that the other channels demand anyway. At flare cores most of it comes from hot plasma, which only 94 and 131 Å ask for, two of the three noisiest channels, so a bright 94 Å reading needs hundreds of units of it. How a loss prices those units decides whether the hot plasma appears in the DEM at all (section 6).

Sources: `CLAUDE.md` (Reference Solvers; 2026-07-31 entries: DEM pixel (i, j) from AIA pixel (2i, 2j), the 18 bins and the eight zero bins stacked on, σ/o medians over 655k pixels; 2026-09-11: α = 0.001 confirmed against the earlier α = 1; 2026-09-13: convergence warnings in our α = 0.001 ElasticNet run; 2026-10-21: 0.67 and 0.266 DN/s of 94 Å per unit of weight); `RData.npz` (R and the logT grid); `fullBP.py` (lines 91 to 116, `getBasis`: the spike and Gaussians, the 0.04 truncation and `notrunc`; `solveLP`: the linear program; `solveElasticNet`: residuals divided by max(tσ, 0.5 DN/s), scikit-learn settings; `invertDEMCube`: retries at 3 and 5 times the band; `parseArgs`: t = 1.4, the basis widths 0, 0.1 and 0.2, α = 1 by default); `dataset/fullBP.py` (our copy of the solver: exposure and degradation scaling, the `estimate_error` call with no calibration flags, `tolLevel`); `dataset/submit_bp_aia_hofdeconv_full.py` and `eval_and_enet_specs/submit_enet_aia_hofdeconv_full.py` (generation flags: the PSF deconvolution with `--errorfn full --notrunc --extendto8 --decimate 2`, and `--zerochill` with α = 0.001 for ElasticNet); `experiments/train_scaled.py` (line 186: the network's basis, built with the default truncation); `eval_and_enet_specs/README.md` (the ElasticNet objective and settings); `.venv/lib/python3.13/site-packages/aiapy/calibrate/uncertainty.py` (the noise terms and their defaults); `slides/talk_prep.md` ("What is σ?"); `results/meeting_20260728_scaling_brief.md` (one full-disk image: tolLevel shares, the share infeasible at the band on and off the disk and its rise with 171 Å brightness); `results/followup_recap_20261013.md` (section 6: Table 2.2, where 94 Å signal comes from, hundreds of units). The sums in Table 2.1 and the untruncated 1.77 and 3.54 follow from the Gaussian formula in `getBasis` on the 0.1 dex grid.
