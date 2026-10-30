## 5. Results

Every number in this section is measured on the shared test set of section 4: 153 days never used for training or model selection, with five solver targets per block, 705 million valid pixel-targets on the BP track and 802 million on the ElasticNet track. "Label-free" means the adopted BP model (width 336, square root plus 12 Fourier frequencies, the stiff band, 120 epochs then 120 more continued from that checkpoint) and the ElasticNet model (same width and input, 40 epochs). "Supervised" is the supervised side's emulator for the same solver, scored on the same pixels against the same solver DEMs. DEM MSE is the squared error per temperature bin averaged over pixels, in units of (10^26 cm^-5)^2; the per-pixel error behind the percentiles is the squared error summed over the 18 bins. Lower is better everywhere except the two flare-core columns, where closer to 100% is better.

### 5.1 Headline comparison

Table 1a. BP track, label-free (the adopted BP model) against the supervised model; the old model (width 336, log1p input, μ = 1, 40 epochs) for reference.

| Measure | Label-free | Supervised | Better | Old model |
| --- | --- | --- | --- | --- |
| DEM MAE | 0.074 | 0.100 | label-free | 0.106 |
| DEM MSE | 1.75 | 0.91 | supervised | 4.28 |
| EM error | 11.0% | 14.3% | label-free | 19.6% |
| W1 (dex) | 0.068 | 0.133 | label-free | 0.083 |
| AIA MAE (DN/s) | 3.48 | 2.95 | supervised | 4.34 |
| AIA MSE ((DN/s)^2) | 103 | 157 | label-free | 83 |
| Median pixel error | 0.036 | 0.112 | label-free | 0.026 |
| p90 | 1.86 | 1.95 | label-free | 3.55 |
| p99 | 24.5 | 26.9 | label-free | 70.8 |
| Quiet MSE | 0.032 | 0.034 | label-free | 0.058 |
| Flare-core emission | 93% | 92% | label-free | 40% |
| Flare-core 94 Å reproduced | 77% | 85% | supervised | 23% |

Table 1b. ElasticNet track, the ElasticNet model against the supervised model.

| Measure | Label-free | Supervised | Better |
| --- | --- | --- | --- |
| DEM MAE | 0.151 | 0.188 | label-free |
| DEM MSE | 0.310 | 0.318 | label-free |
| EM error | 12.2% | 18.0% | label-free |
| W1 (dex) | 0.103 | 0.121 | label-free |
| AIA MAE (DN/s) | 0.56 | 6.32 | label-free |
| AIA MSE ((DN/s)^2) | 161 | 598 | label-free |
| Median pixel error | 0.85 | 1.12 | label-free |
| p90 | 7.76 | 8.51 | label-free |
| p99 | 29.5 | 35.5 | label-free |
| Quiet MSE | 0.122 | 0.130 | label-free |
| Flare-core emission | 99% | 87% | label-free |
| Flare-core 94 Å reproduced | 53% | 48% | label-free |

On BP, the label-free model wins four of the six headline measures (DEM MAE, EM error, W1 and AIA MSE) and the supervised model keeps DEM MSE and AIA MAE. Over all twelve columns the count is nine to three; two of the nine are narrow (p90 1.86 against 1.95, flare-core emission 92.6% against 92.3%). The supervised model's lead on DEM MSE is a factor of 1.9 and comes entirely from flare cores (section 5.2). Its lead on AIA MAE should be read with the solver in view: the BP solver's own AIA MAE on the same pixels is 4.26 DN/s (MSE 61.9), higher than either network's, because BP only has to land inside the band, so AIA MAE on this track measures something the solver itself does not optimise. Against the old model, the adopted BP model improves every measure except the median pixel error, which rises from 0.026 to 0.036 and stays three times below the supervised model's.

On ElasticNet, the label-free model is ahead on all twelve measures. The narrowest is DEM MSE (0.310 against 0.318); the widest is AIA MAE (0.56 against 6.32, eleven times lower). The ElasticNet solver's own AIA MAE is 1.41 DN/s, so here the label-free model reproduces the observations more closely than the solver it emulates.

![Example DEM curves on the BP track: solver, label-free (adopted BP model) and supervised, at eight flare cores and four ordinary Bright pixels of the test set (source: results/plots/22_adopted_bp_20261014/fig4_example_curves_bp.png)](figures/fig4_example_curves_bp.png)

![The same for the ElasticNet track (source: results/plots/22_adopted_bp_20261014/fig4_example_curves_enet.png)](figures/fig4_example_curves_enet.png)

### 5.2 The typical pixel and the mean disagree

Table 2. Per-pixel squared error (sum over the 18 bins) at the percentiles, and where the total squared error sits. Flare cores are the pixels at 32 times the Bright cutoff and above: 69,283 of the 705 million BP pixel-targets (0.010%) and 75,832 of the 802 million on ElasticNet. The last row removes them and averages the rest.

| | BP label-free | BP supervised | ENet label-free | ENet supervised |
| --- | --- | --- | --- | --- |
| Median pixel error | 0.036 | 0.112 | 0.85 | 1.12 |
| p90 | 1.86 | 1.95 | 7.76 | 8.51 |
| p99 | 24.5 | 26.9 | 29.5 | 35.5 |
| p99.9 | 170 | 162 | 102 | 178 |
| Share of SSE in the worst 1% of pixels | 97.7% | 95.2% | 54.9% | 49.4% |
| Share of SSE in flare cores | 94.2% | 90.0% | 44.5% | 33.9% |
| DEM MSE without flare cores | 0.102 | 0.091 | 0.172 | 0.210 |

On the typical BP pixel the label-free model is three times closer to the solver than the supervised model (median 0.036 against 0.112). It stays level or ahead up to the 99th percentile (p90 1.86 against 1.95, p99 24.5 against 26.9). The order flips only in the last 0.1% (p99.9 170 against 162), and because the error is so heavy-tailed those pixels set the mean: the worst 1% of pixels hold 97.7% of the label-free BP squared error and 95.2% of the supervised model's, and flare cores alone, one pixel in 10,000, hold 94.2% and 90.0%. Without flare cores the two models are 0.102 against 0.091, a ratio of 1.1 rather than 1.9. On ElasticNet the label-free model is ahead at every percentile, and flare cores hold 44.5% of its error against 33.9% of the supervised model's; without them it leads by a wider margin (0.172 against 0.210).

Table 3. DEM MSE of the label-free model divided by the supervised model's, by brightness band (the largest channel over its Bright cutoff; pixel shares from the BP track).

| Band | Share of pixels | Adopted BP / supervised | Old BP / supervised | ElasticNet / supervised |
| --- | --- | --- | --- | --- |
| below 0.03 | 3.9% | 0.55 | 0.64 | 1.06 |
| 0.03 to 0.1 | 10.5% | 0.70 | 0.80 | 1.02 |
| 0.1 to 0.3 | 33.0% | 0.81 | 0.96 | 0.98 |
| 0.3 to 1 | 42.5% | 0.98 | 1.84 | 0.93 |
| 1 to 1.8 (Bright) | 6.5% | 1.16 | 2.16 | 0.82 |
| 1.8 to 3.2 | 2.3% | 1.25 | 2.57 | 0.76 |
| 3.2 to 10 | 1.2% | 0.88 | 4.96 | 0.51 |
| 10 to 32 | 0.08% | 1.89 | 17.5 | 0.47 |
| 32 and above (flare cores) | 0.010% | 2.02 | 4.74 | 1.28 |

The adopted BP model is ahead on the faint 90% of pixels (ratio 0.55 at the faintest band, rising to 0.98 at 0.3 to 1 times the cutoff), within 25% of the supervised model across the ordinary Bright bands, and behind by a factor of 1.9 to 2.0 only from 10 times the cutoff upward. The old model's ratios ran from 0.64 to 18 across the same bands, with the worst band at 10 to 32 times the cutoff; the stiff band (section 6) removed that failure. The ElasticNet model sits at 0.47 to 0.98 of the supervised error everywhere from 0.1 times the cutoff to 32 times, at 1.02 to 1.06 on the two faintest bands, and at 1.28 at flare cores.

What remains at BP flare cores is a scale shortfall rather than a misplaced curve: the adopted model holds 92.6% of the solver's emission there, its peak height is 66% of the solver's, and its mean logT shift is 0.01 dex; the old model held 40%, 23% and shifted 0.18 dex cooler. The adopted model also flags more flare-core pixels as multi-peaked than the solver (25.4% against 18.3%; the supervised model 12.8%); section 8 takes up the multi-peaked DEMs.

![Share of pixels against share of DEM squared error in four brightness groups, BP track: flare cores are 0.0098% of pixels and 94.2% of the label-free error, 90.0% of the supervised error (source: results/plots/22_adopted_bp_20261014/fig1_error_share_by_brightness.png)](figures/fig1_error_share_by_brightness.png)

![Relative DEM error (error energy over reference energy) by brightness band, both tracks, label-free against supervised (source: results/plots/22_adopted_bp_20261014/fig2_relative_error_by_brightness.png)](figures/fig2_relative_error_by_brightness.png)

### 5.3 AIA reconstruction

Each DEM is pushed back through the response matrix and compared with the six observed channels. Pooled over all valid pixels, the BP track gives AIA MAE / MSE of 3.48 / 103 for the label-free model, 2.95 / 157 for the supervised model and 4.26 / 61.9 for the solver; the ElasticNet track gives 0.56 / 161, 6.32 / 598 and 1.41 / 134. At flare cores the per-channel picture is the following.

Table 4. Reconstructed over observed intensity per channel, summed over the flare cores of the test set (1.00 is exact).

| Channel | BP solver | BP label-free | BP supervised | ENet solver | ENet label-free | ENet supervised |
| --- | --- | --- | --- | --- | --- | --- |
| 94 Å | 0.92 | 0.77 | 0.85 | 0.55 | 0.53 | 0.48 |
| 131 Å | 1.01 | 1.01 | 0.90 | 0.78 | 0.80 | 0.68 |
| 171 Å | 0.99 | 1.00 | 0.78 | 1.00 | 0.97 | 0.70 |
| 193 Å | 1.01 | 1.00 | 0.90 | 0.98 | 0.97 | 0.76 |
| 211 Å | 0.98 | 0.99 | 0.90 | 0.98 | 0.97 | 0.76 |
| 335 Å | 1.04 | 1.09 | 0.96 | 0.94 | 0.93 | 0.76 |

On BP the adopted model reproduces five of the six channels at flare cores to within 2%, as the solver does, and over-reproduces 335 Å by 9% (the solver by 4%). The one gap is 94 Å, the channel that sees the hot plasma: 77% of the observed signal against the solver's 92% and the supervised model's 85% (the old model: 23%). The supervised model reproduces 94 Å better but every other channel worse, 171 Å by 22% too faint. Away from flare cores the label-free model tracks the solver band by band, including the solver's own shortfall on the faint channels where the band permits it (94 Å at 0.3 to 1 times the cutoff: 0.81 label-free, 0.82 solver; 131 Å below 0.03 times the cutoff: 0.30 and 0.32), whereas the supervised model does not follow the solver there (131 Å at 1.65 on the faintest band).

On ElasticNet the solver itself reproduces only 55% of 94 Å at flare cores, and the label-free model reproduces the channels the way its solver does, within 3 points on every channel (94 Å 53% against 55%). The supervised model falls 24% to 32% short on five channels and reaches 48% on 94 Å. The remaining BP 94 Å gap (77% against the solver's 92%) is a training gap and not a loss gap: section 6 shows the best answer under the stiff loss reproduces 93% of 94 Å at the validation flare cores.

![Reconstructed over observed AIA at flare cores, per channel, both tracks: solver, label-free and supervised (source: results/plots/22_adopted_bp_20261014/fig5_aia_fit_flare_cores.png)](figures/fig5_aia_fit_flare_cores.png)

### 5.4 Architecture and size

**The patch CNN against the per-pixel MLP at scale.** Both were trained on the same 917 days with fresh pixels, log1p input, 40 epochs, the original BP loss (μ = 1) or the ElasticNet objective, and matched in size: the CNN sees a 9 by 9 patch at stride 2 through three 3 by 3 convolutions of 64 channels and layers of 256, 256 and 54 (1.48M parameters); the MLP is width 680 on the centre pixel (1.43M). Table 5 scores each track's own training loss per pixel on 5,013,504 sampled test pixels.

Table 5. Per-pixel training loss on the test split, MLP against patch CNN.

| | BP MLP | BP CNN | ENet MLP | ENet CNN |
| --- | --- | --- | --- | --- |
| Median | 0.98 | 1.01 | 0.020 | 0.033 |
| p90 | 4.62 | 4.74 | 0.137 | 0.154 |
| p99 | 17.9 | 18.8 | 0.47 | 0.52 |
| p99.9 | 51 | 57 | 1.19 | 1.44 |
| Mean | 2.07 | 2.24 | 0.057 | 0.245 |
| Share of the mean in the worst 0.1% of pixels | 5.6% | 9.7% | 8.2% | 73.4% |
| Largest single-pixel loss | 7,622 | 421,309 | 1,490 | 622,832 |
| AIA MAE (DN/s) | 4.93 | 4.73 | 0.54 | 0.82 |
| Sparsity score on the 54 weights (solver 1.79) | 1.90 | 2.03 | | |
| Validation loss | 2.155 | 2.248 | 0.0528 | 0.0683 |

The MLP is lower at every percentile on both tracks, and validation agrees with test. The CNN's one win is BP AIA MAE (4.73 against 4.93). The four-fold gap in the ElasticNet mean is mostly a few CNN pixels: its worst 0.1% hold 73.4% of its loss and its largest single-pixel loss is 622,832 against the MLP's 1,490.

![Held-out test pixels of one block: solver DEM against the MLP and the patch CNN, BP track on the left and ElasticNet on the right (source: results/plots/18_cnn_vs_mlp_resample_20261003/dem_curves_block05005.png)](figures/dem_curves_block05005.png)

**The width sweep.** Seven MLP widths per track, fresh pixels, log1p input, 40 epochs; the parameter counts are for the six raw inputs (the same width with the Fourier input is 409k at 336). The validation loss is each track's training loss on the 153 validation days and needs no solver DEM.

Table 6. Width sweep on the shared test set; validation loss from training.

| Width | Parameters | BP validation loss | BP median | BP DEM MSE | BP flare-core emission | ENet validation loss | ENet median | ENet DEM MSE | ENet AIA MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 680 | 1.43M | 2.155 | 0.023 | 5.10 | 20% | 0.0528 | 0.81 | 36.2 | 908,611 |
| 480 | 722k | 2.167 | 0.025 | 5.15 | 19% | 0.0533 | 0.85 | 3.24 | 41,849 |
| 336 | 360k | 2.148 | 0.026 | 4.28 | 40% | 0.0575 | 1.07 | 0.48 | 650 |
| 232 | 176k | 2.237 | 0.030 | 5.28 | 15% | 0.0689 | 1.70 | 0.79 | 414 |
| 160 | 87k | 2.251 | 0.043 | 5.15 | 19% | 0.0614 | 1.55 | 0.64 | 199 |
| 108 | 42k | 2.261 | 0.047 | 4.58 | 35% | 0.0707 | 2.24 | 0.83 | 217 |

Bigger helps the typical pixel a little: the BP median pixel error falls from 0.068 at 20k parameters (width 72) through 0.030 at 176k to 0.023 at 1.43M, and the ElasticNet median from 2.24 at 42k to 0.81 at 1.43M. Width 48 (10k) collapsed on BP (validation loss 61.4, median 1.55). DEM MSE, by contrast, is erratic across widths because flare cores set it: BP flare-core emission jumps between 12% and 40% with no trend. On BP, width 336 is the validation choice (2.148; 1.43M ties at 2.155 and only nudges the median at four times the size). On ElasticNet the honest note applies: validation preferred the bigger models (0.0533 at 722k, 0.0528 at 1.43M, against 0.0575 at 336), but on the full test set 722k and 1.43M blow up on a handful of bright pixels (AIA MSE 41,849 and 908,611 against 650; DEM MSE 3.24 and 36.2 against 0.48) that the sampled validation never contained, so 336 was kept as the largest stable width on test evidence. The ElasticNet epochs were settled the same way: 120 epochs improves DEM MAE, the median, EM error, W1 and AIA MAE but worsens DEM MSE from 0.310 to 0.362, so 40 stays. The earlier, fixed-sample sweep had concluded that 176k was enough; that conclusion did not survive fresh pixels.

**Models above 409k parameters blow up on rare pixels.** Table 7 collects every larger model trained.

Table 7. The blow-ups, shared test set; the adopted BP model for reference.

| Model | Parameters | Loss | DEM MSE | AIA MSE | Quiet MSE | Median |
| --- | --- | --- | --- | --- | --- | --- |
| Adopted BP model (width 336, 120 + 120 epochs) | 409k | stiff band | 1.75 | 103 | 0.032 | 0.036 |
| BP width 680, 120 epochs | 1.43M | stiff band | 897 | 21.7M | 12.5 | 0.034 |
| BP width 1800, 40 epochs | 10.1M | stiff band | 35.7 | 1.7M | 0.070 | 0.041 |
| BP width 1800, 40 epochs | 10.1M | μ = 1 | 5.10 | 125 | 0.058 | 0.022 |
| ElasticNet width 1800, 40 epochs | 10.1M | ElasticNet | 2,062 | 83M | 5.22 | 0.78 |
| ElasticNet width 480 (sweep, log1p) | 722k | ElasticNet | 3.24 | 41,849 | 0.123 | 0.85 |
| ElasticNet width 680 (sweep, log1p) | 1.43M | ElasticNet | 36.2 | 908,611 | 0.123 | 0.81 |

The pattern is the same each time: the typical pixel is as good as or better than the 336-wide model's (the medians in the last column), and a few pixels fail by orders of magnitude. The 680-wide stiff-band model has the best BP median of the follow-up (0.034) and a DEM MSE of 897 with the worst 1% of its pixels carrying 99.996% of the squared error; at flare cores it over-predicts 171 Å three-fold. The stiff-band 10M model fits 98% of pixels as well as width 336 (MSE 0.037 against 0.036) and its worst 0.1% carry 14 times the error of the 336-wide model's worst 0.1%. On the original loss the 10M model does not blow up but helps nothing either: a median within 5% of the base model's (0.022 against 0.023) and 20% of the solver's emission at flare cores against 35%. The ElasticNet 10M model over-predicts flare-core emission two-fold (216%). Section 7 asks which pixels these are, how far they sit from the training data and what the input encoding does there.

**Spatial context does not help: the earlier evidence.** On the four-day, 128 by 128 crop data the patch CNN looked better than a capacity-matched centre-pixel MLP in sample (sparsity score 1.70 against 1.89, solver 1.79), but that edge came from scoring pixels of days it had trained on. Leave-one-day-out, training on three days and scoring the fourth, gave held-out sparsity scores of 1.59, 1.78, 1.83 and 1.96 for the MLP against 1.64, 1.61, 1.83 and 2.31 for the CNN: the MLP wins two folds (including the X1.6 flare day), ties one and loses one (quiet sun), with the better held-out MAE on three of four. Re-scoring both on 400 held-out pixels per day gave DEM MAE 0.042 for the MLP against 0.047 for the CNN (W1 0.043 against 0.048 dex). The scaled comparison above is the fifth look at the question, and none has found a CNN advantage on unseen days, so the final model takes one pixel and no neighbourhood.

![Leave-one-day-out on the X1.6 flare day, never seen in training: solver against the MLP and the patch CNN, ten pixels of the crop (source: results/plots/09_loo_20260711/loo_heldout_20140910_1731.png)](figures/loo_heldout_20140910_1731.png)

Sources: `STATE.md` (2026-10-14 block; 2026-10-04 blocks; 2026-10-03 fresh-pixel sweep and first-MLP rerun; 2026-10-03 resample table for the ENet width-232 AIA MSE), `results/followup_recap_20261013.md` (Parts 8 to 11), `results/plots/21_followup_20261006/summary.json` (all BP and ElasticNet run metrics: DEM MAE/MSE, EM error, W1, AIA MAE/MSE, Quiet MSE, flare-core emission and 94 Å, median), `results/plots/21_followup_20261006/bp_bright_failure_fu_bp_mu100_ep120_cont.{json,txt}` (percentiles, worst-1% share, by-band DEM MSE, flare-core emission, peak and shift, multi-peaked rates), `results/plots/21_followup_20261006/bp_aia_fit_fu_bp_mu100_ep120_cont.{json,txt}` (BP per-channel reproduction and pooled AIA numbers), `results/plots/21_followup_20261006/bp_bright_failure_fu_bp_mu100_h680_ep120.txt` and `bp_bright_failure_fu_bp_mu100_10m.txt` (the blow-up tails), `results/plots/19_final_models_20261004/bp_bright_failure_final.json` (old model percentiles and by-band DEM MSE), `results/plots/19_final_models_20261004/enet_bright_failure_final.json` and `enet_aia_fit_final.json` (ElasticNet percentiles, by-band DEM MSE, per-channel and pooled AIA), `results/plots/19_final_models_20261004/README.txt`, `results/plots/13_bright_diagnostic_20260928/bp_bright_failure_supervised.{json,txt}` and `enet_bright_failure_supervised.{json,txt}` (supervised percentiles and by-band DEM MSE), `results/plots/22_adopted_bp_20261014/README.txt` (figure provenance), `results/plots/18_cnn_vs_mlp_resample_20261003/eval_scaled_tail_summary.json`, `eval_scaled_summary.json`, `eval_tail_20005027.log` and `eval_rs_19108218.log` (CNN against MLP), `results/plots/17_sweep_resample_20261003/sweep_table.txt` and `enet_aia_fit_resample_h{680,480,336,160,108}.txt` (width sweep), `slides/deck_outline.md` (slides 2, 9, 10, 12, 13: the CNN architecture, DEM MSE without flare cores, validation numbers), `CLAUDE.md` (2026-07-02 ablation, 2026-07-11 leave-one-day-out, 2026-10-10 and 2026-10-13 blow-ups). The ratios in Table 3 and the "DEM MSE without flare cores" row of Table 2 are computed from the full-population and by-band rows of the diagnostic JSONs named above.
