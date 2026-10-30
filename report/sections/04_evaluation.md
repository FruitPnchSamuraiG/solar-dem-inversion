## 4. Evaluation protocol

Both models are scored on the supervised side's protocol: the same test pixels, the same solver DEMs as targets, the same Bright cutoffs and the same validity rule, so every comparison in section 5 is made on identical pixels. Each track is scored against its own solver's DEMs.

### 4.1 The shared test set

The test images are the 153 of the time split, taken on 86 distinct days from 21 September to 31 December 2015. None was used in training, and they entered model selection only in the two exceptions of section 3.5. The split has no gap at this end: the first test image follows the last validation image by 47 s. Each image has five solver targets: the solve of the clean image and four re-solves of copies with Gaussian noise added of half the estimated σ (the generation command passes no `--noisescale`, and `fullBP.py` defaults it to 0.5; earlier material said σ). The noisy files store only the DEM, so the network's input is the clean AIA for all five. Each target is scored on its own seeded random 64 of the image's 256 blocks of 128 by 128 pixels, which gives 48,960 blocks and 802,160,640 pixel-targets, all of them valid on ElasticNet. On BP, 705,016,779 carry a solver DEM and the other 12% carry none. They are the pixels where no DEM fits the 1.4σ band: the generation passes `--zerochill`, which skips the solver's relaxation, and the missing share is 7.7% at clean solves, close to the 8.5% our own clean labels relax, and 12% to 13% at the noisier re-solves (section 7.3). Nothing else is removed: the deconvolution-clamp pixels that training leaves out (section 3.4) are scored (section 7.2).


### 4.2 Bright, Quiet and brightness bands

Table 4.1. Bright cutoffs, each channel's top 5% of intensities over the test split, in DN/s.

| 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 335 Å |
| --- | --- | --- | --- | --- | --- |
| 3.1212 | 16.9986 | 448.8948 | 498.3444 | 190.7388 | 10.3326 |

A valid pixel is Bright when any channel reaches its cutoff and Quiet otherwise; Full is all valid pixels. Bright is 10.1% of the BP pixel-targets and 10.5% of the ElasticNet ones. The cutoffs come from the test split; training uses them only to log the share of Bright pixels it draws.

The analyses by brightness use I, the largest channel over its own cutoff, in nine bands with edges at 10^-1.5, 10^-1, 10^-0.5, 1, 10^0.25, 10^0.5, 10 and 10^1.5, written 0.03, 0.1, 0.3, 1, 1.8, 3.2, 10 and 32 times the cutoff. The first four bands are Quiet and the other five Bright. Flare cores are the pixels at 32 times the cutoff and above: 69,283 BP pixel-targets (0.0098%) and 75,832 on ElasticNet (0.0095%). Table 3 (section 5.2) gives each band's share of pixels.

### 4.3 DEM measures

A pixel-target is valid when the solver's DEM is finite in all 18 bins and the prediction is finite. With p the predicted DEM and y the solver's, DEM MAE and MSE average |p_b − y_b| and (p_b − y_b)² over valid pixel-targets and the 18 bins b, in 10^26 cm^-5 and its square. The total-emission error is a ratio of sums,

$$
\text{EM error} = \frac{\sum_{\text{pixels}} \left| \sum_b p_b - \sum_b y_b \right|}{\sum_{\text{pixels}} \sum_b y_b},
$$

so bright pixels weigh most in it. W1 treats each DEM, normalised to unit total, as a distribution over logT and measures the area between the two cumulative distributions, in dex:

$$
W_1 = \sum_{b=1}^{17} \left| P_b - Y_b \right| \Delta, \qquad P_b = \frac{\sum_{j \le b} p_j}{\sum_j p_j}, \quad Y_b = \frac{\sum_{j \le b} y_j}{\sum_j y_j}, \quad \Delta = 0.1\ \text{dex},
$$

averaged over the pixels where both DEMs have a positive total. It says where the emission sits in temperature, whatever its amount. DEM MSE, EM error and W1 follow the supervised side's table script; DEM MAE comes from our diagnostic on the same pixels and mask.

### 4.4 AIA reconstruction

Every DEM, the label-free model's, the supervised model's and the solver's, is pushed through the 18-bin response, ŷ = R·DEM, and compared with the six observed intensities of the same pixel. AIA MAE and MSE average |ŷ_c − o_c| and (ŷ_c − o_c)² over valid pixel-targets and channels, in DN/s and (DN/s)², on one mask per track: a finite solver DEM, six finite observations and finite reconstructions from both models. Both are in raw DN/s, so channels count by brightness: the 171 and 193 Å cutoffs are 144 and 160 times the 94 Å cutoff. The solver's own reconstruction is not exact, since BP only has to land inside the band. By channel and band the analyses report Σŷ_c/Σo_c over the band's pixels, where 1.00 is exact; "flare-core 94 Å reproduced" is this ratio for 94 Å at flare cores, and "flare-core emission" is the predicted total emission there over the solver's.

### 4.5 Percentiles and the solver's own scatter

The per-pixel error is the squared error summed over the 18 bins, one value per valid pixel-target. We report its median (the median pixel error), p90, p99 and p99.9, read from a histogram of log10 SSE in 0.02-dex bins and so good to about 2%, together with the share of the total held by the worst 1% of pixels and by flare cores. Percentiles come first because the error is heavy-tailed. On the test split in August, one pixel in five million held 94.0% of a model's mean per-pixel BP loss (2.09 × 10^8 of 2.22 × 10^8): that model's mean, 44.32, was four times the patch CNN's 11.27, yet it was lower at every percentile. For the adopted BP model, flare cores hold 94.2% of the DEM squared error, so DEM MSE mostly reports them. We give both.

The five targets also set a floor that no deterministic emulator can beat. They share one input, so a model predicts the same DEM for all of them. Where several targets cover the same block (64% of blocks are covered by two to five), the model's MSE over those targets splits exactly into the squared distance to the mean solver DEM and the variance of the solver's DEMs across targets, the solver's own scatter. For the adopted BP model the scatter is 1.3% of the model's MSE on those pixels, 0.9% on Bright pixels, 45.0% on Quiet and 0.2% at flare cores; for the ElasticNet model it is 19.1%, 8.2%, 48.6% and 0.09%. The Bright and flare-core error is the model's, not label noise, while about half of the Quiet error is the solver disagreeing with itself. With 2.3 targets per pixel on average the variance is underestimated, so these shares are lower bounds.

Sources: `eval_and_enet_specs/README.md` and `eval_and_enet_specs/aia_thresholds.json` (the cutoffs; Bright, Quiet and Full; five targets on the repeated clean input; 48,960 blocks); `student_package/table1_dem_metrics/compute_paper_table_metrics.py` (DEM MSE, EM error and W1, the validity mask, cutoffs as each channel's top 5% over the test split); `experiments/diagnose_bright_failures.py` (the nine bands and the Bright/Quiet split of them, the per-pixel SSE histogram in 0.02-dex bins, the worst-1% share, the solver-scatter split); `experiments/summarize_followup.py` (DEM MAE as the mean of the per-bin MAE); `experiments/eval_aia_shared_test.py` and `experiments/aia_fit_by_brightness.py` (AIA MAE and MSE, the common mask, reconstructed over observed per channel and band); `experiments/train_scaled.py` (the cutoffs used only for logging); `dataset/stage_hofdeconv_full.py` (each target's own seeded 64 of 256 blocks); `dataset/test_datetimes.txt` and `dataset/val_datetimes.txt` (153 images on 86 days, the 47 s boundary); `eval_and_enet_specs/submit_enet_aia_hofdeconv_full.py` and `fullBP.py` (`--noisy`, `--noisescale` default 0.5, `--zerochill`); `results/plots/21_followup_20261006/bp_bright_failure_fu_bp_mu100_ep120_cont.{json,txt}` (705,016,779 pixel-targets, Bright 10.1%, 69,283 flare cores and their 94.2% share, BP scatter shares and 2.27 targets per pixel); `results/plots/19_final_models_20261004/enet_bright_failure_final.json` (802,160,640 pixel-targets, Bright 10.5%, 75,832 flare cores, ElasticNet scatter shares); `CLAUDE.md` (2026-07-31: 0.02% unsolved without `--zerochill`; 2026-08-02: one pixel holding 94.0% of a mean; 2026-09-11: the five targets; 2026-09-28: the scatter decomposition); `STATE.md` (2026-09-28 caveat: 64% of blocks shared, scatter biased low); `slides/deck_outline.md` (slides 7 and 33).
