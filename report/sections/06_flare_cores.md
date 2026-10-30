## 6. Flare cores: the failure and the fix

Flare cores, the pixels at 32 times the Bright cutoff and above, are about one pixel in 10,000 of the shared test set (69,283 of 705 million BP pixel-targets) and hold most of the BP DEM squared error for every model (section 5.2). On the BP track the old model failed there, and the cause was the loss, not training: under the original band weight (μ = 1) the best answer under the loss leaves the hot flare plasma out of the DEM, and the old model had already reached it. A stiffer band (μ = 100), chosen on validation, makes the loss ask for the hot plasma, and longer training takes the network most of the way there; the final BP model, wider and with clamp pixels in training, goes further (section 6.7). This section follows the work in order, so it is mostly about the 409k stiff-band model that came first. The ElasticNet track never had this failure (Appendix B).

### 6.1 The failure

At flare cores the old model (width 336, log1p input, μ = 1, 40 epochs) held 40% of the solver's emission. Its peaks were 23% as high as the solver's, its mean logT 0.18 dex cooler, its curves almost twice as wide (an emission-weighted width of 0.41 dex against 0.23), and 83% of them had extra peaks, against 18% of the solver's. Flare cores held 90.5% of its DEM squared error, but the supervised model puts 90.0% of its error there too: that concentration is what squared error does to DEMs a thousand times larger than typical. The label-free defect was the amount, with 97.6% of the flare-core error lying along the solver's curve rather than across it.

The AIA reconstruction showed which plasma was missing. At flare cores the old model's DEM reproduced 171, 193 and 211 Å to within 2%, as the solver's did, but only 23% of the observed 94 Å, 76% of 131 Å and 67% of 335 Å, where the solver reproduced 92%, 101% and 104%. 94 Å is the channel that sees hot flare plasma (logT about 6.9, 8 MK): where the solver draws a tall peak at logT 6.8 to 6.9, the old model draws a low bump near 6.3 to 6.4 and a tail near 7.0.

![Figure 6.1. The old model at eight randomly drawn flare cores and four ordinary Bright pixels of the test set: solver in black, label-free in blue, supervised in orange. The same twelve pixels as Figure 5.1. (results/plots/19_final_models_20261004/fig4_example_curves_bp.png)](figures/fig4_example_curves_bp.png)

Two other explanations were ruled out: the solver's own scatter across re-solves of a block is 0.38% of the old model's Bright error (0.09% at flare cores), and pixels where the solver's DEM has two or more peaks and the model's one hold 1.25% of its squared error, so neither label noise nor missed second peaks is the cause. The diagnosis was first made in September on an earlier width-232 model trained on the fixed pixel sample of Appendix C (41% of the emission, 24% of 94 Å).

Because the network fit the cool channels and missed 94 Å, it seemed to be failing its own objective, which asks for every channel inside its band: a training problem. Hence the three suggestions: show it more flare cores, train it longer, make it bigger.

### 6.2 Training-side suggestions that did not help

Each suggestion was one change to a common recipe: width 336, fresh pixels, 40 epochs, μ = 1, and the square root plus 12 Fourier frequencies input the follow-up adopted for both tracks. This base is slightly worse than the old model (DEM MSE 4.56 against 4.28, flare-core emission 35% against 40%). Oversampling draws pixels in proportion to brightness (clipped to 1 to 1,000 times the Bright cutoff), so flare cores rise from about 1 in 8,900 training pixels to 1 in 1,060, about 8.4 times as often. The 10M model is 1,800 wide (10.1M parameters).

Table 6.1. The suggestions on the original loss (μ = 1), shared test set.

| BP run, μ = 1 | DEM MSE | DEM MAE | Median pixel error | AIA MSE | Quiet MSE | Flare-core emission | Flare-core 94 Å |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base (square root + Fourier, 40 epochs) | 4.56 | 0.106 | 0.023 | 83 | 0.057 | 35% | 15% |
| Oversample flare cores | 5.05 | 0.106 | 0.025 | 87 | 0.058 | 24% | 9% |
| 120 epochs | 4.21 | 0.105 | 0.022 | 108 | 0.057 | 37% | 16% |
| 10M parameters | 5.10 | 0.106 | 0.022 | 125 | 0.058 | 20% | 8% |

Away from flare cores the four runs score alike (DEM MAE 0.105 to 0.106, Quiet MSE 0.057 to 0.058). At flare cores none helps: oversampling empties the DEMs further (11% of the solver's emission on the validation flare cores, against the base's 34%), 120 epochs trims DEM MSE by 8% but moves flare-core emission only from 35% to 37% while AIA MSE rises from 83 to 108, and the 10M model holds 20%. Training harder did not move the failure, so the loss itself needed checking.

### 6.3 The objective gap

The BP loss is convex in the 54 weights, so each pixel has one lowest value, which an optimiser can find with no network. That minimiser is the best answer under the loss: no network can do better on that pixel. `experiments/objective_gap.py` finds it with L-BFGS-B (started from the network's output and from the solver's DEM, keeping the lower) on 11,954 validation pixels across the brightness bands, 6,000 of them flare cores. It then scores four answers under the same loss: the network's output, the best answer, the solver's DEM and the supervised model's DEM, the last two rewritten as the sparsest set of 54 weights that reproduces them.

Table 6.2. Flare cores of the validation split, every answer scored under the original loss (μ = 1): the sparsity term plus the band penalty.

| Answer | Loss | Sparsity + band penalty | 94 Å reproduced | Emission vs solver | DEM MSE |
| --- | --- | --- | --- | --- | --- |
| Old network | 565 | 294 + 270 | 28% | 38% | 233,000 |
| Best answer under the loss | 463 | 220 + 243 | 26% | 30% | 233,000 |
| Solver's DEM | 1,125 | 1,119 + 6 | 94% | 100% | 0 |
| Supervised model's DEM | 1,878 | 1,107 + 771 | 91% | 95% | 38,000 |

**The network had converged on its loss.** Its median loss sits 0.8% to 3.6% above the best answer in every band from 0.03 to 32 times the Bright cutoff, 7.4% in the faintest band and 8.0% at flare cores, where its DEM error equals the best answer's (233,000 for both). Over all validation pixels, weighted by band as in section 6.5, its DEM MSE is 36.9, the same as the best answer's. A perfect network would land on the best answer, which reproduces 26% of 94 Å and holds 30% of the solver's emission: on this loss no training change can fix flare cores.

**The loss prefers the DEM without the hot plasma.** The solver's DEM keeps every channel in its band (penalty 6) and pays 1,119 in sparsity for the hot plasma; the best answer drops most of it and pays 220 plus 243 in band penalty. The loss rates the solver's DEM 2.4 times worse than a DEM without the hot plasma (1.3 to 1.5 times in the other Bright bands), so every training step at a flare core moves away from the solver. The supervised model shows it from the other side: under our loss its DEM scores worst (1,878), yet it is the closest network to the solver (DEM MSE 38,000 against 233,000). A low loss does not mean close to the solver.

Two caveats. L-BFGS-B stopped at its iteration limit on 271 of the 11,954 pixels at μ = 1; the loss is convex, so the true minimum there is at or below the value found, which can only widen the solver's deficit. And flare cores are a larger share of the validation days than of the test days (0.015% of pixels against 0.0098%), so validation values run higher than test values: compare rows, not sections.

### 6.4 Why: the price of hot plasma

The solver and the loss share one goal, the DEM with the smallest total weight that reproduces the six observations, and differ in how they treat the band. For the solver the band is a rule: every channel must land inside it. For the loss it is a price: a channel may land outside whenever that saves more in sparsity than it costs in band penalty.

![Figure 6.2. The BP training loss: the sum of the 54 weights (the sparsity term) plus μ times the band penalty over the six channels; x_k are the weights, (Dx)_c the channel the DEM reproduces, l_c and u_c the band edges, σ_c the noise; the penalty is zero inside the band and grows as the square of the distance outside it, in half band widths (source: results/plots/21_followup_20261006/loss_formula.png)](figures/loss_formula.png)

Table 2.2 (section 2.6) gives what each channel sees per unit of plasma, a unit being 10^26 cm^-5 of emission measure (the unit of the weights, the DEM and the sparsity term). Warm plasma (1 to 1.3 MK, logT 6.0 to 6.1) lights 171, 193 and 211 Å by several to tens of DN/s per unit; it has to be in the DEM for those channels, and its 94 Å signal comes free. At flare cores most of the 94 Å signal comes from hot plasma, which lights no channel by more than 0.4 DN/s per unit (a spike at logT 6.9, 8 MK, gives 0.266 DN/s of 94 Å, 0.40 of 131 Å and 0.34 of 193 Å), so a bright 94 Å reading needs hundreds of units, all paid in the sparsity term, and only 94 and 131 Å ask for them.

Take a typical flare core, counting 94 Å alone: it reads 164 DN/s with σ = 11.5, so its band runs from 148 to 180 (half width 16.1).

Table 6.3. The worked example at μ = 1.

| Option | Sparsity | Band penalty | Total |
| --- | --- | --- | --- |
| Hot plasma in, as a spike at logT 6.9: 164 ÷ 0.266 ≈ 616 units | 616 | 0 | 616 |
| Hot plasma in, as the 0.2 dex Gaussian centred at logT 6.8: 164 ÷ 0.67 ≈ 245 units | 245 | 0 | 245 |
| Hot plasma out: 94 Å near 0, 148 below the band | 0 | (148 ÷ 16.1)² = 85 | 85 |

One unit of weight on a spike at logT 6.9 gives 0.266 DN/s of 94 Å, so including the hot plasma that way costs 616. The cheapest hot shape is the 0.2 dex Gaussian centred at logT 6.8, at 0.67 DN/s per unit, and including the plasma that way costs about 245. That is within 1% of the most 94 Å any basis element gives (0.676 DN/s per unit, the bound of 1.48 weight units per DN/s in section 3.3), which comes from the warm Gaussian at logT 6.1; that element also lights 171 Å by about 170 DN/s per unit, so it cannot stand in for hot plasma. Both exceed the 85 that leaving the plasma out costs at μ = 1, so the conclusion holds either way; at μ = 100 the same miss costs 8,500 and the plasma goes back in. The real price, the linear program's dual price of the 94 Å lower edge solved pixel by pixel, agrees: at the validation flare cores its median is 1.46 weight units per DN/s, at the single-edge bound, and it exceeds the bound at 49% of them (99th percentile 8.8), where another channel's upper edge also binds and makes the plasma dearer (section 3.3).

The failure grows with brightness because brighter pixels draw more of their 94 Å from hot plasma. On validation the best answer reproduces 91% of the observed 94 Å at 1 to 1.8 times the Bright cutoff, 92% at 1.8 to 3.2, 78% at 3.2 to 10, 44% at 10 to 32 and 26% at flare cores, where the solver's DEM reproduces 100%, 101%, 95%, 88% and 94%. The band itself is sound: σ follows photon counting, growing as the square root of the signal.

The same price explains why the label-free BP model emitted about 15% less than the solver even at ordinary pixels, a September finding left unexplained. Inside the band nothing pulls a channel toward its observation, so the sparsity term drags the emission down to the band's lower edge, where the solver's answer also sits; a squared penalty has zero slope at that edge, so with a finite μ it always pays to go a little past it. At 0.3 to 1 times the Bright cutoff the best answer emits 0.855 of the solver's emission at μ = 1, then 0.886, 0.941, 0.981 and 1.002 at μ = 3, 10, 30 and 100 (the first screen, on a sample of test pixels; the validation repeat gives 0.861 at μ = 1 and 1.009 at μ = 100). The old network sat at 0.866 in that sample. Over the full test set the old model emits 0.848 of the solver's emission, the 409k stiff-band model 0.984 and the final BP model 0.988.

### 6.5 The μ screen on validation

The screen tests the loss, not a network. For each μ it solves every validation pixel's best answer and scores it against the solver's DEMs, asking how close to the solver a network that learned this loss perfectly would get. The DEM error is the DEM MSE with each brightness band weighted by its share of the validation split; the Quiet MSE is the same over the bands below the Bright cutoff.

Table 6.4. The μ screen: the best answer under each loss on the validation split, with the supervised model (a trained network) for reference.

| Loss | DEM error | Quiet MSE | Flare-core 94 Å | Flare-core emission |
| --- | --- | --- | --- | --- |
| μ = 1 (original) | 36.9 | 0.076 | 26% | 30% |
| μ = 10 | 10.8 | 0.037 | 81% | 90% |
| μ = 30 | 5.10 | 0.032 | 89% | 97% |
| μ = 100 | 2.32 | 0.041 | 93% | 100% |
| μ = 300 | 1.80 | 0.048 | 94% | 101% |
| Supervised model | 6.10 | 0.054 | 91% | 95% |

**Stiffer helps bright pixels and costs faint ones.** The DEM error falls to a minimum near μ = 300 (1.89 at 200, 1.80 at 300, 2.16 at 1,000). The Quiet MSE is lowest at μ = 20 to 30 (0.031 to 0.032) and rises beyond: at faint pixels the band is mostly noise, and a stiff band makes the best answer chase it. At a flare core, by contrast, a price 100 times higher works like the solver's rule.

**μ = 100 takes most of the gain** (DEM error 2.32) with the Quiet MSE still below the supervised model's (0.041 against 0.054); μ = 300 lowers the DEM error only to 1.80 and raises the Quiet MSE to 0.048. The flare-core 94 Å column, which compares the reconstruction with the observation and so needs no solver DEM, plateaus at μ = 100 (93%, then 94% at 300 and 95% at 1,000). The stiff band is still not the solver's rule: at the validation flare cores it rates the solver's DEM about 1.3 times worse than its best answer (1,688 against 1,264), but that best answer now holds the hot plasma.

A second candidate (μ = 30 with the sparsity term divided by the square root of brightness on Bright pixels) screened nearly as well, DEM error 1.95, but trained worse (section 6.6). μ is one scalar chosen on validation with the solver's DEMs, like any loss weight; training never sees them. An earlier screen on test pixels was repeated on validation, which kept μ = 100, and the choice rests on validation alone. Section 3.3 gives the theory: the band violation at the best answer falls as 1/μ, so μ follows from the response matrix, the noise and a chosen tolerance, with no solver DEM.

### 6.6 The stiff-band runs

The recipe stays the same (width 336, square root plus 12 Fourier frequencies, fresh pixels, clamp pixels excluded); the runs vary the training length, the sampling and the size. The row above the supervised model in Table 6.5 repeats the 680-wide run with clamp pixels kept (section 7.1).

Table 6.5. BP runs on the stiff band (μ = 100), shared test set. The validation error is the DEM error of Table 6.4 for the trained network.

| BP run | DEM MSE | Median pixel error | AIA MSE | Quiet MSE | Flare-core emission | Flare-core 94 Å | Validation error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original loss, base (reference) | 4.56 | 0.023 | 83 | 0.057 | 35% | 15% | 39.7 |
| 40 epochs | 2.37 | 0.062 | 44 | 0.035 | 89% | 68% | 20.1 |
| 120 epochs | 2.00 | 0.041 | 44 | 0.032 | 92% | 75% | 17.8 |
| 240 epochs from scratch | 1.74 | 0.036 | 591 | 0.035 | 91% | 73% | 14.0 |
| 120 + 120 epochs, continued (the 409k stiff-band model) | 1.75 | 0.036 | 103 | 0.032 | 93% | 77% | 15.8 |
| Oversample flare cores, 40 epochs | 2.50 | 0.071 | 39 | 0.037 | 88% | 63% | 21.1 |
| 10M parameters, 40 epochs | 35.7 | 0.041 | 1.7M | 0.070 | 92% | 74% | not scored |
| 680 wide (1.5M), 120 epochs | 897 | 0.034 | 21.7M | 12.5 | 115% | 78% | 4,170 |
| **680 wide (1.5M), 120 epochs, clamp pixels in training: the final BP model** | **1.08** | **0.031** | **148** | **0.033** | **97%** | **88%** | not scored |
| Supervised model | 0.91 | 0.112 | 157 | 0.033 | 92% | 85% | 6.10 |

**The band alone halves the DEM MSE** (4.56 to 2.37) and lifts flare-core emission from 35% to 89%. Training then becomes the limit: at the validation flare cores the 40-epoch network's loss sits a median 118% above its best answer (the old network on the old loss: 8%). Longer training therefore helps; after 120 epochs the gap is 47%, the DEM MSE 2.00 and the flare-core 94 Å 75%.

**240 epochs from scratch misfits rare pixels.** It keeps improving the DEM (MSE 1.74, validation error 14.0), but its AIA MSE rises from 44 to 591 and its Quiet MSE from 0.032 to 0.035: a few pixels break the band badly, and on the validation flare cores its mean band penalty is 129 times the 120-epoch model's.

**120 + 120 epochs, continued, is the 409k stiff-band model,** the best BP model until the clamp runs. Warm-started from the 120-epoch checkpoint with a fresh cosine schedule, it gets the same DEM gains (MSE 1.75, median pixel error 0.036) while holding the AIA MSE at 103, below the supervised model's 157, with a flare-core band penalty 13 times the 120-epoch model's. It improves on the 120-epoch model in every measure except AIA MAE and AIA MSE. Not chosen: 120 epochs alone (AIA MSE 44 but DEM MSE 2.00) and 240 from scratch.

**Nothing else helps.** Oversampling still hurts (DEM MSE 2.50, median pixel error 0.071). The 10M and 680-wide models blow up on rare pixels while their median pixel error stays low (section 5.4). Section 7.1 shows which pixels, and that keeping them in training takes the 680-wide model from DEM MSE 897 to 1.08: that run is the final BP model. log1p input is worse than square root plus Fourier (DEM MSE 3.23 against 2.37). Gradient clipping at norm 1 fires on every step on the stiff band (90% of steps on the original loss); without it AIA fits better (AIA MSE 36) but every DEM measure worsens (DEM MSE 3.20). The second screen candidate trained worse than μ = 100 (validation error 22.4 against 20.1). μ = 300, on the same 120-epoch recipe, trailed μ = 100 at every matched epoch (validation DEM MAE 0.114 against 0.088 at epoch 48, 0.099 against 0.079 at epoch 76) and was stopped after about 90 epochs, never scored on the test set.

### 6.7 Flare cores now

Table 6.6. BP flare cores on the shared test set (69,283 pixel-targets).

| | Old model | 409k stiff-band model | Supervised | Solver |
| --- | --- | --- | --- | --- |
| Emission vs solver | 40% | 93% | 92% | 100% |
| 94 Å reproduced | 23% | 77% | 85% | 92% |
| 131 Å reproduced | 76% | 101% | 90% | 101% |
| Peak height vs solver | 23% | 66% | 91% | 100% |
| Mean logT shift (dex) | -0.18 | -0.01 | 0.00 | 0 |
| W1 (dex) | 0.255 | 0.033 | 0.015 | 0 |
| Curves with extra peaks | 83% | 25% | 13% | 18% |
| Share of the model's DEM squared error | 90.5% | 94.2% | 90.0% | |

With the stiff band alone the plasma is back at the right temperature: 93% of the solver's emission (supervised 92%), a mean shift of -0.01 dex, W1 down from 0.255 to 0.033 dex, and 131 Å reproduced as the solver does. Spurious extra peaks no longer carry the error: pixels where the solver's DEM has one peak and the model's several held 42.0% of the old model's squared error and hold 3.5% of this model's (supervised 1.4%), though it still draws extra peaks on 25% of flare-core curves against the solver's 18% (section 8). Figure 5.1 shows this model at the same twelve pixels as Figure 6.1.

What was left is height at the very top. The 409k stiff-band model tracks the solver's peak up to a height of about 100 (0.97 of it at 32 to 100), then falls behind: 0.81 at 100 to 320, 0.61 at 320 to 1,000, 0.51 at 1,000 to 3,200 and 0.40 above 3,200. The old model fell from 0.77 to 0.095 over the same range; the supervised model holds 0.89 to 0.93, then 0.71 above 3,200. The 269 pixel-targets above 3,200 hold 73.1% of its DEM squared error.

![Figure 6.3. Predicted over solver peak height, binned by the solver's peak height, both tracks, label-free (the 409k stiff-band model, the ElasticNet model) against supervised: on BP that model follows the solver to a peak of about 100 and reaches 0.40 of it above 3,200; at the faint end both models draw peaks above the solver's near-zero ones (source: results/plots/22_adopted_bp_20261014/fig3_peak_height_calibration.png)](figures/fig3_peak_height_calibration.png)

By brightness band (Table 5.4 in section 5.2), its DEM MSE is 0.55 of the supervised model's on the faintest pixels and 2.02 at flare cores; the old model's ratios ran from 0.64 to 17.5.

**What remained on the 409k stiff-band model.**

- Flare cores held 94.2% of its DEM squared error and set the BP DEM MSE gap: 1.75 against 0.91 (a factor of 1.9), and 0.102 against 0.091 without them (section 5.2).
- The shortfall was in 94 Å: 77% of the observed signal against the solver's 92%; 131, 171, 193 and 211 Å are within 2%, and 335 Å is at 109% (section 5.3).
- It was a training gap, not a loss gap. Under the stiff band the best answer reproduces 93% of 94 Å at the validation flare cores and this model 77.5%, with its loss there a median 35% above the best answer. The solver's own re-solve scatter is 0.2% of its flare-core error, so this is model error, not label noise.
- Longer training kept closing the gap, but 240 epochs from scratch began to break the band on rare pixels.
- Open: why the median pixel error rose from 0.026 (old model) to 0.036 while the Quiet MSE fell from 0.058 to 0.032. The final model is at 0.031 and 0.033.

**The final BP model at flare cores.** The final model (1.5M parameters, the stiff band, 120 epochs, clamp pixels in training) closes most of what was left. At flare cores it holds 96.7% of the solver's emission (supervised 92%), reproduces 88% of 94 Å (supervised 85%, solver 92%) and 100% of 131 Å, and peaks at 0.89 of the solver's height (supervised 0.91), with a mean shift of -0.01 dex, W1 of 0.018 dex (supervised 0.015) and extra peaks on 14% of curves (supervised 13%, solver 18%). It follows the solver's peak much further: 0.98 at 32 to 100, 0.97 at 100 to 320, 0.92 at 320 to 1,000, 0.78 at 1,000 to 3,200 and 0.53 above 3,200. Flare cores still hold 92.0% of its DEM squared error, 83.0% of the total in those same 269 pixel-targets above 3,200, and its DEM MSE there is 1.22 times the supervised model's. That is the whole of the remaining BP gap: 1.08 against 0.91 overall, and 0.087 against 0.091 without flare cores (section 5.2). Tables 5.1 and 5.5 give the rest.

Sources: `results/followup_recap_20261013.md` (Parts 2 to 9: run recipes, the worked example, the screen and run tables), `results/plots/21_followup_20261006/summary.json` (the shared-test run metrics in Tables 6.1 and 6.5), `results/plots/23_meeting_followup_20261024/bp_bright_failure_fu_bp_h680_clampmask.{txt,json}`, `bp_aia_fit_fu_bp_h680_clampmask.txt` and `bp_h232_h680_clampmask.json` (the final BP model: the last run of Table 6.5, emission over all pixels, its flare-core numbers and peak calibration), `results/plots/21_followup_20261006/bp_objective_gap_val.txt` (Table 6.2, the old network's gap by band and band-weighted DEM MSE, 94 Å by brightness for the best answer and the solver, validation populations, the non-converged count), `results/plots/21_followup_20261006/bp_objective_gap_val_sweep.txt` (Table 6.4 and μ = 20, 200 and 1,000, band-weighted from the per-band rows), `results/plots/21_followup_20261006/bp_objective_gap.txt` (the first screen on test pixels: emission 0.855 to 1.002, the old network's 0.866), `results/plots/21_followup_20261006/bp_objective_gap_val_final_mu100.txt`, `bp_objective_gap_val_overnight_cont.txt`, `bp_objective_gap_val_overnight_ep240.txt`, `bp_objective_gap_val_overnight_h680.txt` and `bp_objective_gap_val_nets_early.txt` (gaps and band penalties under the stiff band, validation errors, the solver's DEM under μ = 100, the oversampled emission on validation), `results/plots/21_followup_20261006/bp_bright_failure_fu_bp_mu100_ep120_cont.{txt,json}` (Table 6.6, the 409k stiff-band column, peak calibration, the shape cross-tab, solver scatter, emission over all pixels), `results/plots/21_followup_20261006/bp_aia_fit_fu_bp_mu100_ep120_cont.txt` (per-channel reconstruction at flare cores), `results/plots/21_followup_20261006/scaled_mlp6_barrier_*_history.json` (flare-core sampling rates, clipping rates, μ = 300 against μ = 100 at matched epochs), `results/plots/19_final_models_20261004/bp_bright_failure_final.json` and `bp_aia_fit_final.json` (the old model: emission, peak, shift, width, extra peaks, along-curve share, solver scatter, missed second peaks, per-channel reconstruction), `results/plots/13_bright_diagnostic_20260928/bp_bright_failure_supervised.json` (the supervised model at flare cores), `results/plots/23_meeting_followup_20261024/bp_mu_theory_val.txt` (the single-channel price bound of 1.480 weight units per DN/s for 94 Å), `RData.npz` with `getBasis` in `fullBP.py` (the 94 Å response of each basis element, computed: 0.266 DN/s per unit for the spike at logT 6.9 and its other channels, 0.669 for the 0.2 dex Gaussian at 6.8, 0.676 for the one at 6.1, which gives 171 DN/s of 171 Å per unit), `results/bright_failure_story_20260929.md` (the September diagnosis on the width-232 model), `slides/deck_outline.md` (slides 15 to 22: the 0.266 response and the talk notes), `slides/talk_prep.md` (the emission-deficit mechanism; μ = 300), `CLAUDE.md` (2026-09-28; 2026-10-10; 2026-10-13, including the μ = 300 value 0.099 at epoch 76; 2026-10-15; 2026-10-21, the cheapest shape at 0.67 DN/s per unit and the price of 245), `experiments/objective_gap.py` (the method) and `experiments/submit_followup.sh` (the warm start). The old network's band-weighted DEM MSE (36.9) and the band-penalty ratios (13 and 129) are computed from the per-band rows of the files named above.
