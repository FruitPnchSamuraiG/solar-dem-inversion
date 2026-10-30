## 7. Failure analysis

Three questions from the last review are answered here: which pixels blow up when the network gets bigger, and why a bigger network should fail at all (7.1); what the pixels at the low end of the AIA joint distributions are (7.2); and what happens at the pixels where no DEM fits the noise band, which is where relaxing the constraint matters (7.3). The first two have the same root: a small set of inputs that training never contained.

### 7.1 Which pixels blow up, and why bigger networks fail there

Every model's DEM error is concentrated in a few pixels, but the larger models concentrate it far more (Table 7.1). On the shared test set the adopted BP model's worst 0.01% of pixels hold 95% of its squared error; the 1.5M-parameter model trained on the same loss (680 wide, μ = 100, 120 epochs) puts 99.7% of its error in its worst 100 pixels, out of 705M.

Table 7.1. Concentration of the DEM squared error on the shared test set (sum over the 18 bins, per pixel), by model.

| Track | Model | DEM MSE | Worst 0.01% of pixels | Worst 100 pixels |
| --- | --- | --- | --- | --- |
| BP | 409k, adopted | 1.75 | 95.1% | 67.3% |
| BP | 1.5M, μ = 100, 120 epochs | 897 | 99.99% | 99.7% |
| BP | 10M, μ = 100 | 35.7 | 99.7% | 98.0% |
| BP | 10M, μ = 1 | 5.10 | 91.5% | 50.1% |
| ElasticNet | 409k, final | 0.310 | 44.8% | 29.1% |
| ElasticNet | 722k, log1p input | 3.24 | 94.7% | 70.0% |
| ElasticNet | 1.4M, log1p input | 36.2 | 99.5% | 68.7% |
| ElasticNet | 10M | 2,062 | 99.99% | 96.9% |

**Two kinds of worst pixel.** The adopted model's worst 1,000 distinct inputs are 993 flare cores, and the 10M BP model's worst 1,000 overlap them almost entirely (Jaccard 0.88): these are the brightest flare cores, hard for every model. The pixels that make the large models blow up are different. The 1.5M model's worst 100 share only 7% with the adopted model's (Jaccard 0.07), and they look alike: in its worst 100 distinct inputs a median of five of the six channels read exactly zero, and in 56% only one channel has signal, always 94 Å, at a median 3.0 DN/s, 28 times the median of its eight neighbours (more than 5 times in every case). The ElasticNet 10M model's worst 100 are the same kind (64% a lone 94 Å reading of about 4.2 DN/s, 57 times its neighbours). The ElasticNet 722k and 1.4M log1p models fail on a related set: in about a quarter of their worst 100 the lone channel is 131 Å, at about 200 DN/s, 1,500 to 2,000 times its neighbours. Figure 7.2 shows one such pixel.

**Why training never saw them.** A channel that reads exactly zero is the positivity clamp of the PSF deconvolution: the deconvolved value came out negative and was set to zero. The training mask required every channel above 10⁻³ DN/s, which drops every pixel with a clamped channel. That is 40% of the valid test pixels, nearly all faint (96% of the faintest band, 2.2% at 1 to 1.8 times the Bright cutoff, none above 3.2 times), so it barely touches the pixels the metrics weigh. But it removes every input in which a channel is zero, and the lone-spike pixels are exactly such inputs. The validation loss uses the same mask, so it could not see them either (section 3.5 also notes that it covers only the first four validation images). This is why validation preferred every model larger than 409k while each of them blew up on test.

**How far they are from the training data.** We drew 2 million valid training pixels and measured, for each test pixel, the distance to the nearest one in standardised log intensity (Figure 7.1). Ordinary test pixels sit at a median 0.09 to 0.21 from a training pixel, depending on brightness. The 1.5M BP model's worst inputs sit at about 4.9, and the ElasticNet 10M model's at 3 to 5: twenty to fifty times further out than anything the network was fitted on. The adopted model's worst pixels, the brightest flare cores, sit at a median 0.40 against 0.21 for flare cores in general; 19% are outside the range of the training sample in some channel. They are at the edge of the training data, not off it.

![Figure 7.1. Distance from each model's worst 100 distinct test inputs to the nearest of 2 million training pixels, in standardised log intensity, against random test pixels of three brightness bands (controls). Shared test set. (results/plots/23_meeting_followup_20261024/fig3_blowup_distance.png)](figures/fig3_blowup_distance.png)

**The mechanism.** A per-pixel network is an interpolator. Where training pixels are dense its output is pinned by the loss; far from them nothing constrains it. A larger network has more freedom to bend between and beyond the data, and the Fourier encoding, whose purpose is to let the network represent sharp functions of its input, gives it the means. At the lone-spike inputs the larger models return DEMs up to 10⁵ times the solver's: the ElasticNet 10M model's worst 1,000 inputs have a median emission 1.6 × 10⁵ times the solver's, and the 1.5M BP model reconstructs 171 Å at more than 2 × 10⁵ times the observation at a tenth of its worst 1,000. Bigger is therefore not worse at fitting the data: the larger models match or beat 409k at the median (section 5.4). It is worse at extrapolating, and the evaluation contains inputs that training excluded.

**Confirmation runs.** If this is the cause, training on those inputs should remove the blow-ups. Four runs repeat existing ones with one change, the clamp pixels kept and only their clamped channels freed from the band (`--clamp_mode mask`).

Table 7.1b. The same recipe with and without clamp pixels in training, shared test set.

| Run | DEM MSE, excluded | DEM MSE, clamp pixels kept | Note |
| --- | --- | --- | --- |
| BP 1.5M, μ = 100, 120 epochs | 897 | **1.08** | flare-core 94 Å 88%, emission 96.7% |
| ElasticNet 10M, 40 epochs | 2,062 | 1.82 | AIA MSE still 1.5 × 10⁵ |
| BP 409k, μ = 100, 40 epochs | 2.37 | 2.79 | flare-core emission 85% against 89% |
| ElasticNet 409k, 40 epochs | 0.310 | 0.439 | AIA MSE 162 against 161 |

The score blow-ups are gone: two models that failed by orders of magnitude now score normally, and the 1.5M BP model is the best BP model we have, which is why it became the final model. Clamp pixels do not help the 409k models: they take a large share of each epoch's draws, nearly all faint, and a small network pays for that at the bright end. The ElasticNet 10M model still trails the 409k model and keeps a few wild reconstructions, so its failure is reduced rather than removed.

**What the runs did not fix.** The score blow-ups and the wildest reconstructions are not the same thing. On the eight full disks of the viewer, counting a pixel as wild when some channel is reconstructed above 10³ DN/s and more than 100 times its observation, the old 409k BP model has 126 wild pixels of 33.5M and the new model 124; the new model's largest overshoot is about 20 times higher (2.5 × 10⁸ against 1.2 × 10⁷). Nearly all have no solver DEM (123 of 124), because they are unsolved even at the 5 times band; training excludes unsolved pixels in either mode, so clamp masking cannot reach them, and they are absent from every DEM score. They are single-channel readings among zeros (335 Å at 1,320 DN/s, typical value about 10, with the other channels zero): not plasma. One does have a solver DEM: a lone 335 Å reading of 44 DN/s on 9 December, which alone holds 97.7% of the new model's error on that disk.

**A self-check closes this.** Simple input rules do not work: a clamp-plus-neighbour-spike rule catches under half of the wild pixels while flagging up to 14% of all pixels, and a dominant-channel rule catches at most 79%. The direct test does: a DEM must reproduce what was observed, so a pixel whose reconstruction overshoots its observation by more than 100 times is flagged. This needs neither a solver nor a label, flags 124 of 33.5M pixels, and removes the outliers from the viewer's images. The scores in this report do not apply it; with it the new model's BP DEM MSE would be about 1.02, since one pixel holds 6% of the total.

### 7.2 The low end of the AIA joint distributions

In the viewer's joint distributions of observed against reconstructed AIA, some pixels observed near 1 DN/s are reconstructed far above the diagonal by the label-free model but not by the supervised one. We counted them over the whole shared test set: observed below 1 DN/s and reconstructed above 10 times the observation (both floored at 0.5 DN/s, as in the plots).

Table 7.2. Pixels in the low cluster on the shared test set, BP track, by channel and by whose DEM puts them there. Out of 705M valid pixels.

| Channel | Label-free | Supervised | Solver | Label-free cluster with that channel clamped |
| --- | --- | --- | --- | --- |
| 94 Å | 0 | 1 | 0 | |
| 131 Å | 18,625 | 3,792 | 152,767 | 84% |
| 171 Å | 51 | 369 | 0 | 96% |
| 193 Å | 8,073 | 460 | 123,154 | 89% |
| 211 Å | 95 | 36 | 0 | 93% |
| 335 Å | 23,370 | 63 | 102,110 | 81% |

The cluster is small: about 50,000 pixels, 0.007% of the test set. Almost all of it is the deconvolution clamp: 81% to 96% of the pixels have that very channel at exactly zero, and 96% to 100% are isolated dips against their eight neighbours. It splits into two groups.

- **131, 193 and 335 Å: the observation is wrong, not the DEM.** Here the solver's own DEM sits above the zeroed observation more often than ours (152,767 pixels against 18,625 in 131 Å) and by as much (mean 1.25 against 1.09 dex above). No DEM that fits the other five channels can produce zero in the sixth. The supervised model sits closer to the zero, because it learned these pixels from the solver's labels.
- **171 and 211 Å: the network extrapolates.** Here only our model is far off (3.4 and 1.9 dex above the observation; the solver 0.04 and 0.0; the supervised model 0.6 and 0.0). These are the lone-spike inputs of 7.1: half of the 171 Å cluster is a lone 94 Å reading of about 3.5 DN/s, 38 times its neighbours, with the five other channels zero (Figure 7.2). The solver cannot fit that 94 Å reading without lighting the zeroed channels, so it reproduces almost nothing; our network has never seen such an input and returns a large DEM.

![Figure 7.2. One lone-spike pixel from the 171 Å low cluster. Left: the six observed channels and the median of the eight neighbouring pixels. Right: what each DEM reproduces. Values below 0.05 DN/s are drawn at 0.05. (results/plots/23_meeting_followup_20261024/fig4_lone_spike_example.png)](figures/fig4_lone_spike_example.png)

**Cosmic rays.** The advisor's suggestion fits the lone spikes: a single channel reading 30 to 2,000 times its neighbours, in a patch where everything else is clamped to zero, is the signature of a residual cosmic-ray hit or a hot pixel (one image cannot tell them apart). The broader cluster is the clamp, not cosmic rays. The opposite corner, observed above 20 DN/s and reconstructed below a tenth of it, is rare for our model (25 pixels in 131 Å, 137 in 335 Å); 60% to 83% of those are isolated spikes too.

**Where the clamp falls.** Replaying the block positions of our test split: on the disk 26% of pixels have a clamped channel, but only in the faint channels (94 Å 12%, 335 Å 12%, 131 Å 5%, the bright three at most 0.06%); within 100 pixels of the limb 16%; off the limb 63%, in every channel.

**Effect on the headline numbers.** Removing the 49,991 pixels of the label-free low cluster from both models' scores changes our BP DEM MSE from 1.751 to 1.745 but our AIA MSE from 103 to 69; the supervised model's stays at 157. A model-independent mask, every pixel with a clamped channel removed (40% of pixels, nearly all faint), gives AIA MSE 108 against 256 and DEM MSE 2.88 against 1.49, higher for both because the remaining pixels are brighter on average. On ElasticNet the cluster is larger but shared with the solver (335 Å: ours 578,931, the solver's 611,921), and removing it moves AIA MSE only from 161 to 158 (supervised 598 to 593).

**With clamp pixels in training.** The 1.5M model's low cluster is larger, not smaller: 3.6M pixels in 171 Å, 2.0M in 193 Å and 1.1M in 211 Å, all with that channel clamped, about 5.5M pixels or 0.8% of the test set. This is what freeing a channel means: nothing pulls the reconstruction down to a zero we no longer trust, so the DEM fits the other five channels and predicts roughly 15 times the floored value in the clamped one. The JPDFs therefore show a fuller column at the floor, while the scores barely move (AIA MSE 148 with the cluster, 149 without). The high corner is nearly empty (102 pixels, all in 335 Å).

### 7.3 Where no DEM fits the band: relaxing the constraint

At some pixels no non-negative DEM fits all six channels within 1.4σ. The solver then widens its band to 3 times and, failing that, 5 times (`fullBP.py:515`), and records the level. On our clean labels the band is relaxed at 8.5% of test-split pixels (3 times) and 0.08% (5 times); 0.01% stay unsolved. On the validation sample of the μ screen we solved the linear program ourselves: every tolLevel-1 pixel is feasible at 1.4σ, and every relaxed pixel is infeasible (936 of 1,055) or numerically unsolvable (119), so the flag records infeasibility. (The solver's stored DEMs sit inside their bands to within 0.01 half-widths, a median 5 × 10⁻⁴: float rounding.)

**These pixels carry the stiff band's cost.** Our loss keeps the 1.4σ band everywhere. At feasible pixels the best answer under the loss moves toward the solver at every μ, in every brightness band, all the way to μ = 1,000 (at flare cores its DEM error falls from 1.8 × 10⁵ at μ = 1 to 9.5 × 10³ at 100 and 1.9 × 10³ at 1,000). At relaxed pixels it moves away, because a stiffer band forces an inconsistent observation that the solver itself gave up on (at 0.3 to 1 times Bright, 0.055 at μ = 1, 0.92 at 100, 1.29 at 1,000). The relaxed pixels are 0.5% to 12% of each band above 0.3 times Bright, but hold 70% to 90% of the best answer's DEM error at μ = 100 between 0.3 and 10 times Bright, and 46% (10 to 32 times) and 38% (flare cores) above. The U-shape of the μ screen's Quiet error (section 6.5) is entirely theirs.

**Relaxing the band as the solver does.** We re-solved the best answer at the relaxed pixels with the solver's own band, keeping the excess measured in units of the unrelaxed 1.4σ (`experiments/relaxed_band_screen.py`). The trade-off disappears (Figure 7.3, Table 7.3): every column improves monotonically with μ.

Table 7.3. The μ screen on validation with the fixed 1.4σ band and with the solver's band (excess in noise units). Population-weighted as in section 6.5.

| μ | DEM error, fixed | DEM error, solver's band | Quiet MSE, fixed | Quiet MSE, solver's band | Flare-core 94 Å, solver's band |
| --- | --- | --- | --- | --- | --- |
| 10 | 10.9 | 12.1 | 0.037 | 0.027 | 80% |
| 30 | 4.94 | 5.47 | 0.034 | 0.015 | 89% |
| 100 | 2.30 | 2.79 | 0.046 | 0.0079 | 92% |
| 300 | 1.78 | 1.11 | 0.053 | 0.0036 | 93% |
| 1,000 | 2.17 | 0.63 | 0.055 | 0.0018 | 94% |

The solver reproduces 94% of flare-core 94 Å; the supervised model's numbers on this screen are a DEM error of 6.10 and a Quiet MSE of 0.054. How the excess is measured matters: dividing it by the widened half-width instead softens the penalty 9 or 25 times in DN/s at relaxed pixels and gives flare cores back their deficit (88% of 94 Å at μ = 100).

![Figure 7.3. The best answer under the loss on validation, with the fixed band and with the solver's band at the pixels it relaxed. (results/plots/23_meeting_followup_20261024/fig2_relaxed_band_screen.png)](figures/fig2_relaxed_band_screen.png)

**What this means for μ.** With the solver's own rule in the loss, μ has no interior optimum. The loss is the exterior penalty of the solver's program, relaxation included, and μ only sets how far outside the band the answer may sit, λ_c tσ_c/(2μ) (section 3.3). Larger is better for the best answer; the reasons to stop at a finite μ are training stiffness and the 1/μ law's diminishing returns. On the fixed band, μ = 300 trained behind μ = 100 at matched epochs (section 6.6), and the relaxed pixels, whose penalty never reaches zero, are a likely cause.

**What the headline numbers see.** The shared test set's BP targets are missing at 7.7% of clean-solve pixels and 12% to 13% of the noisy re-solves. The supervised side's generation command passes `--zerochill`, which skips the relaxation (we have the ElasticNet command; the BP rate at clean solves, 7.7%, matches our relaxed share of 8.5%), so the pixels where the band is infeasible have no BP answer and are not scored. The headline BP numbers therefore come from feasible pixels only, where the fixed and relaxed bands coincide.

**The training run.** We trained μ = 300 with the solver's band for 120 epochs (409k, no clamp pixels). At the relaxed pixels it does what the screen predicts: validation DEM MAE there is 0.15 against 0.50 for μ = 300 on the fixed band and 0.45 for μ = 100. It also trains faster than μ = 300 on the fixed band (0.090 against 0.099 at feasible pixels at epoch 76), but stays behind μ = 100 (0.079). On the shared test set it reaches DEM MSE 1.54, flare-core emission 96.8% and 94 Å 87%, better than the 409k μ = 100 model at 120 epochs (2.00, 92%, 75%), but Quiet MSE rises to 0.050 against 0.032, and one validation pass spiked to 4 × 10¹⁶ at epoch 40. So the relaxed band works as theory says at the pixels it targets, but a stiffer loss still costs training time on the faint majority. μ = 100 stays. A 1.5M model on the relaxed band is untested.

### 7.4 Summary

- The large models' score blow-ups came from inputs training never saw: a lone channel spike, consistent with a cosmic-ray hit or hot pixel, among channels the deconvolution clamped to zero. Keeping clamp pixels in training removes them (BP 1.5M: 897 to 1.08), and that model is now the BP final. The most extreme single pixels remain, nearly all without a solver answer, and a self-check on the reconstruction flags them.
- The rest of the low JPDF cluster (131, 193, 335 Å) is the clamp itself, where the observation is wrong and the solver's DEM disagrees with it as much as ours does.
- The stiff band's cost on faint pixels comes from the 9% of pixels where the band is infeasible. For the best answer, relaxing the band there as the solver does removes it and μ becomes a convergence parameter. In training, the relaxed band fixes those pixels but μ = 300 still costs the faint majority, so μ = 100 stays.

Sources: `results/plots/23_meeting_followup_20261024/` (`bp_blowup_census.txt`, `enet_blowup_census.txt`, `bp_jpdf_pixels.txt`, `enet_jpdf_pixels.txt`, `bp_jpdf_pixels_tol.txt`, `bp_mu_theory_val.txt`, `bp_relaxed_band_screen.txt`, and the figures); the per-pixel arrays behind them (`output/meeting_followup/*.npz`, pulled from `output/experiments/diagnostics/` on Torch); `lone_spike_check.txt`, `spike_flag_test.txt`, `fulldisk_outliers.txt`, `bp_bright_failure_fu_*`, `bp_aia_fit_fu_*`, `enet_*_fu_*`, `bp_h232_h680_clampmask.json`, `bp_jpdf_pixels_h680_clamp.txt`, the training histories and (the lone-spike counts for each model's worst 100 distinct inputs, and the shared test set's missing BP answers by target); `eval_and_enet_specs/submit_enet_aia_hofdeconv_full.py` (`--zerochill`); `fullBP.py:515` (relaxation); `experiments/blowup_census.py`, `experiments/jpdf_pixels.py`, `experiments/mu_theory.py`, `experiments/relaxed_band_screen.py`.
