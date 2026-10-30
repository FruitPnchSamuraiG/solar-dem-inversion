## 3. Method: amortised optimisation

Both solvers solve one problem per pixel: from the six intensities o and their noise σ, find the 54 non-negative basis weights w that minimise the solver's objective. We train one network f_θ to do this for any pixel, by minimising the same objective over the training pixels:

$$
\theta^\star = \arg\min_\theta \sum_{p} L\left(f_\theta(o_p);\ o_p, \sigma_p\right)
$$

Solving is paid for once, in training, and each new pixel then costs one forward pass. No solver DEM enters training. Solver DEMs score the result (section 4) and compare losses with different band weights on validation pixels (section 3.5).

### 3.1 Network

The network sees the six intensities of one pixel: not its neighbours (section 5.4) and not σ, which only the loss uses. Each intensity, in DN/s, is square-rooted and lifted to sines and cosines at 12 fixed frequencies 2π·2^(i/2), i = −4, …, 7 (Fourier features; Tancik et al. 2020), which with the square roots themselves gives 6 × 25 = 150 inputs. This is the supervised model's input path; the square root spreads the bright end far more than the old model's log1p (a 10^4 DN/s flare core maps to 100, against 9.2).

Four hidden layers with SiLU activations feed a linear layer to 54 outputs and a softplus, which keeps every weight non-negative. The softplus input is clamped at −20 so that its gradient never vanishes, and the output layer starts small (weights scaled by 0.01, bias −3) so that the first DEM reproduces intensities of the order of the observations; both fixes date from the collapse of the first full-disk runs (appendix A). The width differs by track. The final BP model is 680 wide, with 1,528,694 parameters, quoted as 1.5M; the ElasticNet model is 336 wide, with 408,630 parameters, quoted as 409k. For width W the count is 150W + 3W² + 54W weights and 4W + 54 biases. On six log1p inputs the same widths have 1.43M and 360k. Section 3.5 says how each width was set. The speed of either network against the solver has not been measured.

The output is the solver's own variable. Neither solver works on the 18 DEM bins (logT 5.5 to 7.2, 0.1 dex apart) directly: both solve for 54 weights on a fixed basis B, at each bin a spike and two Gaussians exp(−ΔlogT²/a²) with a = 0.1 and 0.2 dex, and report DEM = Bw, whose intensities are Dw with D = RB (R in DN/s per 10^26 cm^-5). Predicting the same weights, the network searches the solver's space of DEMs (the solver fits flare cores in it, so the basis is not the limit), BP's sparsity term means the same thing for both, and the sparsity score (Σw)²/Σw² is comparable with the solver's (1.79 on BP). The supervised model instead outputs the 18 bins. One difference remains: our basis zeroes each Gaussian below 0.04 of its peak, while the solver runs pass `--notrunc` and keep those tails, at most 2.1% of an element's emission; both generation commands we have pass it, and the difference moves the price bounds of section 3.3 by under 1%.

### 3.2 Losses

With o_c and σ_c the observation and noise of channel c in DN/s, (Dw)_c the intensity the DEM reproduces and t = 1.4 the solver's first tolerance level, the band is o_c ± tσ_c. The BP loss is

$$
L_{\mathrm{BP}}(w) = \sum_{k=1}^{54} w_k + \mu \sum_{c=1}^{6} \left[ \frac{\max\left(0,\ |(Dw)_c - o_c| - t\sigma_c\right)}{t\sigma_c} \right]^2, \qquad w \ge 0.
$$

The first term is BP's objective, the total weight. The hinge is zero inside the band, so there is no fit term there: any DEM within the noise is equally good and sparsity breaks the tie, as in BP. Dividing by the half-width tσ_c makes a half-width outside cost the same in every channel, so 94 Å at a few DN/s counts as much as 171 Å at hundreds. The square is our choice (BP is a linear program with a hard band): it makes the penalty smooth at the edge, at the cost of an optimum that sits slightly outside the band at any finite μ (section 3.3). The band weight μ is the exchange rate between squared half-widths and units of weight. BP treats the band as a rule and the loss treats it as a price; section 6 turns on that difference. The old model used μ = 1, the final BP model the stiff band, μ = 100. Per pixel the loss is convex in w, so each pixel has one best answer under the loss.

![Figure 3.1. The BP loss's band penalty for one channel: zero inside the band of ±tσ around the observation, growing as the square of the distance outside it, measured in half-widths. (slides/img/band_penalty.png)](figures/band_penalty.png)

The ElasticNet loss is

$$
L_{\mathrm{EN}}(w) = \frac{1}{2C} \sum_{c=1}^{6} \left[ \frac{(Dw)_c - o_c}{t\sigma_c} \right]^2 + \alpha\lambda \sum_{k=1}^{54} w_k + \frac{\alpha(1-\lambda)}{2} \sum_{k=1}^{54} w_k^2, \qquad w \ge 0,
$$

with the supervised side's α = 0.001, λ = 0.5 and C = 6. This is the solver's objective term for term (scikit-learn's non-negative ElasticNet fitted to D and o divided by tσ_c, whose 1/(2N) over N = 6 rows is the 1/(2C) above), except that the solver floors tσ_c at 0.5 DN/s; on the staged validation errors 1.4σ falls below 0.5 DN/s at under 0.1% of pixels, so the floor hardly ever binds. The L2 term makes it strictly convex, so each pixel's best answer under the loss is, up to that floor, the solver's DEM; and with no band there is nothing to price, which is why ElasticNet never had BP's flare-core failure. Both losses are averaged over the 8,192 pixels of a batch.

### 3.3 The band weight μ

The BP loss is the quadratic exterior penalty of BP's linear program,

$$
\min_{w \ge 0} \ \sum_{k} w_k \quad \text{subject to} \quad o_c - t\sigma_c \le (Dw)_c \le o_c + t\sigma_c, \quad c = 1, \dots, 6,
$$

and penalty-method theory says what it does at finite μ (Nocedal and Wright 2006, chapter 17). At the penalised optimum, a channel whose band edge binds sits outside the band by v_c DN/s, where the marginal penalty, 2μv_c/(tσ_c)² per DN/s, equals the marginal saving in total weight: the linear program's dual price λ_c of that edge, in weight units per DN/s. Hence

$$
\frac{2\mu\, v_c}{(t\sigma_c)^2} = \lambda_c \qquad \Longrightarrow \qquad \frac{v_c}{t\sigma_c} = \frac{\lambda_c\, t\sigma_c}{2\mu}.
$$

In half band-widths the violation falls as 1/μ and grows with the noise and the price. The relation is exact once μ is large enough that the penalised optimum keeps the basis elements and binding edges of the linear program's solution.

When a single edge binds, the linear program can always buy a DN/s of that channel from its most efficient basis element, so R and B alone bound the price:

$$
\lambda_c \le \frac{1}{\max_k \, (RB)_{c,k}}.
$$

Table 3.1. Single-channel price bound, in weight units (10^26 cm^-5) per DN/s.

| 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 335 Å |
| --- | --- | --- | --- | --- | --- |
| 1.48 | 0.10 | 0.003 | 0.007 | 0.026 | 0.60 |

The bound differs more than 400-fold between 94 Å and 171 Å: a unit of weight buys at most two thirds of a DN/s of 94 Å but a few hundred of 171 Å. 94 Å is the dear channel.

At the flare core worked through in section 6, 94 Å reads 164 DN/s with σ = 11.5, so tσ = 16.1 and the band runs from 148 to 180. At the bound the 94 Å violation is 1.48 × 16.1/(2μ) = 11.9/μ half-widths. At μ = 1 that exceeds the 9.2 half-widths (148/16.1) by which a DEM without hot plasma falls short, so the best answer drops the hot plasma outright. At μ = 10 the violation is 1.19 half-widths, at 30 it is 0.40, at 100 0.12 and at 300 0.04; μ = 120 holds the pixel within 0.1 half-width.

Turned round, a tolerance of ε half-widths needs μ ≥ λ_c tσ_c/(2ε), a number set by the response, the basis and the noise, with no solver DEM. Photon noise makes σ grow as the square root of the counts, so the μ needed grows with the square root of the brightness. Faint pixels have small σ, and their 94 Å comes free with the warm plasma that 171, 193 and 211 Å demand (section 6), so they hardly respond to μ; flare cores combine the largest σ with the dearest channel and need μ near 100. The same law, a small violation below the lower band edges that shrinks as 1/μ, explains the old model's emission deficit at ordinary pixels: on validation pixels at 0.3 to 1 times the Bright cutoff, the best answer under the loss emits 0.861 of the solver's emission at μ = 1 and 1.009 at μ = 100.

Two refinements follow, neither yet trained. A per-pixel, per-channel weight proportional to λ_c tσ_c, with the bound standing in for λ_c, needs only R, B and σ and would hold every pixel to the same tolerance. An L1 exterior penalty, μ Σ_c max(0, |(Dw)_c − o_c| − tσ_c)/(tσ_c), is exact: once μ exceeds λ_c tσ_c in every binding channel (about 24 at the flare core above), its minimiser solves BP's linear program itself, at the cost of a slope that jumps at the band edge.

The bound assumes one binding edge. When several bind, the price can exceed it, because the element that sets the bound may push another channel out of its band. The 94 Å bound comes from warm plasma, the 0.2 dex Gaussian at logT 6.1 (0.676 DN/s per unit), which also lights 171 Å; at a flare core the hot element at logT 6.8 (0.669 DN/s per unit) sets a price 1% higher, and other combinations can raise it further. The theory is therefore checked on validation pixels.

**Validation check.** On the 11,954 validation pixels of the μ screen (every flare core of the split and about 1,000 per brightness band), we solved BP's linear program at t = 1.4 for its dual prices and the best answer under the loss at μ = 1 to 1,000, and compared the measured violation with λ_c tσ_c/(2μ) (`experiments/mu_theory.py`, Figure 3.2).

- **The law holds.** At μ = 100 the measured violation divided by the predicted one has a median of 1.00 over 45,925 violated pixel-channels, and the median is 1.00 in every brightness band for μ ≥ 10. μ times the violation stays constant from μ = 10 to 1,000 (a coefficient of variation of 0.00 to 0.06 at the median pixel below flare cores, 0.19 at flare cores). Below μ ≈ 10 the best answer drops basis elements the linear program keeps, the hot plasma at bright pixels, and the first-order law no longer applies (at flare cores the median ratio is 0.66 at μ = 1).
- **The prices.** At flare cores the median 94 Å price is 1.46, at the single-edge bound of 1.48; it exceeds the bound at 49% of them (99th percentile 8.8) where other edges bind. 94 Å is the dearest channel at 70% of flare cores, 335 Å at 15% and 131 Å at 13%; over all pixels 94 Å leads at 40%, 335 Å at 26% and 131 Å at 17%.
- **μ for a tolerance.** From the measured prices, holding the median flare core within 0.5, 0.2 and 0.1 half-widths needs μ = 34, 84 and 168; the 90th-percentile flare core needs 153, 382 and 763. At μ = 100 the median flare core sits about 0.17 half-widths outside its band and the 90th percentile about 0.8, roughly one noise σ beyond the edge. Over all pixels the median needs only μ = 13 for 0.5 and 64 for 0.1. A single global μ cannot hold every pixel to one tolerance: the most demanding feasible pixel in the sample, a flare core priced on 335 Å, would need 63,801 for 0.1 half-width. This is the case for the per-pixel weight above.
- **Infeasible pixels.** At 8.8% of the sample (12% of flare cores) no DEM fits the 1.4σ band, and these are exactly the pixels the solver relaxed (section 7.3). There the violation stops falling at a floor set by the observation, whatever μ, and the law does not apply.

![Figure 3.2. Left: the largest band violation at feasible validation pixels against μ, median and 90th percentile, for three brightness ranges, with a 1/μ guide. Right: measured against predicted violation at μ = 100, per violated pixel-channel. (results/plots/23_meeting_followup_20261024/fig1_violation_vs_mu.png)](figures/fig1_violation_vs_mu.png)

So μ is not a free hyperparameter. Given the response, the basis and the noise, the theory states how far outside its band the best answer sits at any μ, and validation confirms it pixel by pixel. μ = 100 is the value that holds the typical flare core to about a sixth of a half-width; section 7.3 shows that once the band is relaxed where the solver relaxes it, larger μ only moves the best answer closer to the solver.

### 3.4 Training

The 1,223 full-disk images (2014 and 2015, about two a day, deconvolved, with σ per pixel and channel; section 2) are split in time: 917 for training (1 January 2014 to 16 June 2015), 153 for validation (to 21 September 2015) and 153 for test (to 31 December 2015). Each image's 2048 by 2048 DEM grid is cut into 256 blocks of 128 by 128 pixels, of which a seeded random 64 are kept once at staging: 58,688 training blocks. AIA is subsampled by 2, so DEM pixel (i, j) sees AIA pixel (2i, 2j), the one the solver used. Each epoch draws 512 fresh pixels uniformly from the valid pixels of every block, 30,048,256 in all, off-disk pixels included.

A pixel is valid if the solver solved it (this flag, not a DEM, is the one solver output training reads) and its observations and σ are finite. The positivity clamp after deconvolution sets negative values to zero, so a channel can read zero where the others carry signal. We call a channel clamped when it reads at or below MIN_OBS = 10^-3 DN/s, three decades below the faintest channel's median (94 Å, 0.76 DN/s). A clamped channel is an artefact, and the solver fits a DEM to it anyway.

There are two treatments. The first, used for the ElasticNet model and for every BP run before the confirmation runs of section 7.1, drops every pixel with a clamped channel from training and validation. That mask was estimated at about 5% of pixels from the two brightest channels; counted over all six it drops about 40% of valid pixels (39.7% on the shared test set), nearly all faint (section 7.1). The second (`--clamp_mode mask`), used for the final BP model, keeps those pixels and frees only their clamped channels: such a channel's band is made so wide that it adds nothing to the loss, the other channels keep their bands, and the network still reads the zero as an input. The ElasticNet loss drops a freed channel the same way. Either way the pixels stay in the test set (section 7.2). Keeping them is what removed the large models' score blow-ups (section 7.1).

Until 2 October 2026 each block's draw was seeded by its index alone, so every epoch reused one fixed sample of 30M pixels, about 0.8% of the training images' pixels (appendix C); every full-disk model in sections 5 to 8 was trained with fresh pixels.

Training uses Adam (Kingma and Ba 2015; peak learning rate 3 × 10^-4, no weight decay) with a linear warm-up of 500 steps on BP and 3,000 on ElasticNet, then cosine decay to zero; batches of 16 blocks of 512 pixels, 3,668 steps per epoch; and a global gradient-norm clip at 1, which on the stiff band acts on every step (norms 20 to 500; without it the AIA fit improved but every DEM measure got worse). The ElasticNet model trains 40 epochs. The final BP model trains 120 epochs at μ = 100 from scratch. The earlier 409k stiff-band model (width 336, clamp pixels excluded) trained 120 epochs at μ = 100, then 120 more from that checkpoint with `--init_from` (weights only; new optimiser, warm-up and schedule). The full training state is saved every epoch, so runs proceed in resumable segments of under two hours, past which the cluster cancels GPU jobs at low utilisation; the model used is the last epoch's, with no early stopping. An epoch of the ElasticNet model takes 152 to 174 s on one GPU, so its 40 epochs take under two hours; the final BP model took a median 169 s per epoch and 6.6 hours in all.

### 3.5 Model selection on validation

For a fixed loss we select on the training loss itself, computed after every epoch on a fixed validation sample, with no solver DEM: 512 pixels from each of the first 256 validation blocks, 131,072 pixels. Blocks are stored in time order, 64 to an image, so these are the first four validation images, 16 to 18 June 2015.

This signal chose the per-pixel MLP over the patch CNN (section 5.4), width 336 on the original BP loss (2.148 against 2.155 at 1.43M, a tie that favours the smaller model) and the square root plus Fourier input on ElasticNet (0.0533 against 0.0575 for log1p); on the original BP loss the input tied (2.1472 against 2.1478), so the old model kept log1p. On the stiff band it chose the new input for BP (5.731 against 6.006), kept the clip (against 8.052 without it) and ranks the 409k stiff-band model first among the 409k stiff-band runs (5.444, against 5.488 for 240 epochs from scratch and 5.648 for 120). The screen's DEM error, below, preferred 240 epochs from scratch (14.0 against 15.8), but that run put 130 times the 120-epoch model's band penalty on the validation flare cores, against 13 times for the continuation.

Different band weights give different losses, so μ cannot be chosen on the training loss. The band weight and the stiff-band candidates were compared on about 12,000 pixels drawn from all 153 validation images, 6,000 of them flare cores, by DEM error against the solver's DEMs with each brightness band weighted by its share of pixels (section 6): the one place solver DEMs enter a choice. The screen's 94 Å column needs no solver DEM and levels off at μ = 100 (26%, 81%, 89%, 93% and 94% at μ = 1, 10, 30, 100 and 300). The band weight was chosen this way before the account of section 3.3 existed; that account now says why the screen levelled off there.

Three choices were made with test evidence, because 131k pixels from four images are too few for the rarest pixels and, with clamp pixels excluded, contain none of the inputs that fail. The first is width. At matched training the validation loss tied with or preferred every model above 409k parameters that we trained (on ElasticNet 0.0533 at 722k, 0.0528 at 1.43M and 0.0522 at 10M, against 0.0575 with the log1p input and 0.0533 with the final input at width 336; on the stiff BP band 5.562 at 1.5M and 5.706 at 10M, against 5.648 and 5.731), yet each blew up on a few rare test pixels (DEM MSE 3.24, 36.2 and 2,062 against 0.48 and 0.310 on ElasticNet; 897 and 35.7 against 2.00 and 2.37 on BP). With clamp pixels excluded, width 336 is the largest that does not blow up on either track, which is why the ElasticNet model stays at 409k; on BP the screen's stratified sample also flagged the 1.5M model (DEM error 4,170 against 17.8). The second is the ElasticNet training length: 120 epochs reaches a lower validation loss (0.0523 against 0.0533) and improves DEM MAE, the median pixel error, EM error, W1 and AIA MAE on test, but worsens DEM MSE from 0.310 to 0.362 and AIA MSE from 161 to 188, so 40 epochs stay.

The third is the final BP model itself. The validation losses of clamp-mask runs use a different mask from the earlier runs (the 1.5M run ends at 4.910 against the 409k stiff-band model's 5.444, on a sample that now holds faint clamp pixels), so they cannot rank the two, and no 409k clamp run at 120 epochs exists to compare at matched training. The 1.5M clamp model was therefore chosen on test scores: DEM MSE 1.08, against 1.75 for the 409k stiff-band model and 2.79 for the 409k clamp run at 40 epochs. μ = 100 was kept from the screen; the relaxed-band run at μ = 300 was not adopted (section 7.3). Validating on every pixel of the 153 validation images, clamp pixels included, is the fix (section 9).

Sources: `experiments/train_ablations.py` (FourierFeatures at line 47, CenterMLP at line 72); `experiments/train_scaled.py` (NormalizedInput at line 109 and its square-root mapping, ClampedSoftplus, init_output_head, make_scheduler, Adam at 3 × 10^-4, clip 1, checkpoints and resumable segments, `--init_from`); `src/losses.py` (barrier_loss_batch at line 169, enet_loss_batch); `src/zarr_data.py` (ZarrPatchBlockDataset and its validity mask, `clamp_mode` and UNCONSTRAINED, EpochResampler, MIN_OBS with the 0.76 DN/s and 5% figures); `fullBP.py` (getBasis at line 91, the 0.04 truncation and `--notrunc`, `--tolfac` 1.4, solveElasticNet, `_minimumDNs` = 0.5); `dataset/stage_hofdeconv_full.py` (256 blocks per image, a seeded 64 kept, time order); `dataset/{train,val,test}_datetimes.txt` (split dates, the first four validation images); `experiments/job_train_followup.sbatch`, `experiments/job_train_input_encoding.sbatch`, `experiments/submit_followup.sh` (16 blocks per batch, warm-up 500 and 3,000 steps, segments, gradient norms 20 to 500, the continuation, the clamp-mask runs); `results/plots/20_sqrtff_final_20261004/sqrt_ff_19141672.log` (408,630 parameters, 58,688 blocks, 30,048,256 and 131,072 pixels, 3,668 steps per epoch, C = 6, 152 to 174 s per epoch, validation 0.0533); `eval_and_enet_specs/README.md`, `eval_and_enet_specs/submit_enet_aia_hofdeconv_full.py` (ElasticNet settings, `--notrunc`); `results/plots/21_followup_20261006/scaled_mlp6_*_history.json` (stiff-band, 10M and ElasticNet 120-epoch validation losses); `results/plots/23_meeting_followup_20261024/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120_history.json` (the 1.5M clamp run: validation 4.910, epoch times, 6.6 hours), `bp_jpdf_pixels_h680_clamp.txt` (39.71% of valid pixels clamped), `bp_h232_h680_clampmask.json`, `bp_bright_failure_fu_bp_clampmask.txt` (DEM MSE 1.08 and 2.79); `results/plots/21_followup_20261006/bp_objective_gap_val.txt` (emission against μ at 0.3 to 1 times the cutoff); `results/plots/17_sweep_resample_20261003/sweep_table.txt` (width sweep); `results/followup_recap_20261013.md` (Parts 3, 6, 7, 8 and 10); `STATE.md` (2026-10-04 blocks); `CLAUDE.md` (2026-07-31: the 5% estimate; 2026-10-02; 2026-10-21: the price bounds and the worked violations); `slides/deck_outline.md` (slides 4 to 6, 9, 10); `slides/talk_prep.md` (σ not an input, training time, speed unmeasured); `RData.npz` with `experiments/train_scaled.py:load_operators` (the 94 Å yields 0.676 and 0.669 per unit of weight and the 2.1% truncated share, computed for this section and not yet in a results file).
