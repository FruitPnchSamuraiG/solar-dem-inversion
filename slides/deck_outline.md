# Advisor deck: slide outline

Master outline: one block per slide. This file and the deck
(`slides/DEM_deck_v2.pptx`, built by `slides/build_deck_v2.py`) are kept in
sync: change this file first, then the deck is rebuilt from it.

Each slide: **Title** (a takeaway sentence), **On slide** (the actual text, kept
short), **Visual** (what the image or diagram shows, if any), **Notes** (what to
say; goes into speaker notes, not onto the slide). Each part opens with a
divider slide ("Part N" over the section name, template's BIG_NUMBER layout).

Edits made in Google Slides get copied back here and into the build script
before the next rebuild, or the rebuild would discard them.

---

## Section 1: Problem and approach (settled)

### Slide 1 · Title

- **Title:** Predicting DEM label-free

### Slide 2 · Divider

- **Part 1:** Problem and approach

### Slide 3 · Why label-free

- **Title:** Same physics, different training signal
- **On slide:**
  - Top line: Solvers (BP, ElasticNet) run one optimisation per pixel: accurate, but slow at full resolution.
  - Row labels: "Supervised: learns the solver's DEMs"; "Label-free (unsupervised): learns the solver's objective implicitly".
  - Bottom line (bold): We never train on the solver's DEMs; we use them only to evaluate.
- **Visual:** the two training loops as two rows of boxes, styled like the model diagram. Supervised: observed AIA, model, DEM, then a "loss" link to the solver's DEM. Label-free: observed AIA, model, DEM, predicted AIA, then a "loss" link to the observed AIA.
- **Status:** settled.
- **Notes:** The audience knows the inverse problem. This slide only says what is different about our training signal.

### Slide 4 · The model

- **Title:** The model: a small MLP feeding a fixed forward model
- **On slide:**
  - 6 log-intensities in; 4 hidden layers of 232, SiLU; 54 non-negative basis weights out.
  - BP and ElasticNet solve for the same 54 weights (at each of 18 bins, a spike and Gaussians 0.1 and 0.2 dex wide), and their L1 term acts on those weights.
  - DEM = B·w over 18 bins, logT 5.5 to 7.2; predicted AIA = R·DEM. B and R are fixed; only the MLP is learned.
  - 176k parameters; one forward pass per pixel.
- **Visual:** pipeline diagram: observed AIA, MLP, basis weights ("54, same as solver"), DEM, predicted AIA, with the loss closing the loop.
- **Notes:** Not everyone knows the solvers' internals: they never solve for the 18 bins directly. Both solve for weights on a fixed basis B (`fullBP.py` `getBasis`, widths 0, 0.1, 0.2), with D = R·B, and report DEM = B·x. Predicting the same weights means the network searches the same space of DEMs the solver does, and BP's sparsity criterion means the same thing for both.
- **Status:** settled.

### Slide 5 · The loss

- **Title:** The loss is the solver's own objective
- **On slide:**
  - BP track: L = Σ_c [ max(0, |ŷ_c − o_c| − tσ_c) / tσ_c ]² + Σ_k w_k
    - Zero anywhere inside the noise band (t = 1.4), quadratic outside.
    - Then the L1 term picks the sparsest DEM that fits: BP's own criterion.
  - ElasticNet track: L = (1/2C) Σ_c [ (ŷ_c − o_c) / tσ_c ]² + αλ Σ_k w_k + ½α(1−λ) Σ_k w_k², with the solver's α = 0.001, λ = 0.5, C = 6.
  - In both, ŷ = R·B·w and w ≥ 0.
- **Visual:** the BP band penalty against prediction error: flat at zero inside the noise band, quadratic outside.
- **Notes:** There is no fit term inside the band. Any DEM within noise is equally good, and sparsity breaks the tie, exactly as in BP.
- **Status:** settled.

### Slide 6 · Data

- **Title:** Data: two years of full-disk AIA, split in time
- **On slide:**
  - SDO/AIA, 6 EUV channels, about two images a day in 2014–2015: 1,223 full-disk images.
  - PSF-deconvolved (Hofmeister); a noise σ per pixel and channel from the AIA error model; DEMs on a 2048² grid.
  - Our model trains on AIA and σ only: each training image is cut into 64 blocks of 256×256 per image. Each epoch draws 512 fresh random pixels from each of the 58,688 blocks(917*64), for 40 epochs.
  - Earlier runs reused one fixed ~30M-pixel sample (a sampler bug) and scored better on validation (BP 4%, ElasticNet 18%).
  - Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.
  - Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves(only supervised).
- **Visual:** a timeline of the split, to scale: train Jan 2014 – Jun 2015 (917 images), validation Jun – Sep 2015 (153), test Sep – Dec 2015 (153).
- **Notes:** Every run before October seeded each block's pixel draw by the block alone, so it reused the same 30M pixels every epoch. All results from here on use fresh pixels every epoch, as described. The fixed-sample runs scored better (validation loss BP 2.156 vs 2.237, ElasticNet 0.0582 vs 0.0689, far beyond the 0.2–0.4% run-to-run spread); only ElasticNet's flare cores improved with fresh pixels. The split is chronological, so every test image is later than anything seen in training. Each epoch draws 512 pixels from each of 58,688 training blocks (917 images × 64 blocks of 256²). The test set uses 64 random 128² blocks of each test image per solver target, five targets per image: one clean solve and four re-solves under simulated photon noise of the same observation. Pixels where deconvolution clamped a bright channel to zero (about 5%) are excluded.
- **Status:** draft.

---

## Section 2: Finding a trainable objective (draft)

### Slide 7 · Divider

- **Part 2:** Finding a trainable objective

### Slide 8 · Which loss

- **Title:** Six channels underdetermine the DEM: the regulariser picks it
- **On slide:**
  - Before any network: five losses optimised directly, pixel by pixel, on 4 timestamps, each compared with BP.
  - Fit-only losses (χ² + smoothness, max entropy, Tikhonov) reconstruct AIA ~12x better than BP, yet land ~5x further from its DEM.
  - BP's own terms (noise band + L1) land on BP's DEM: MAE 0.03 vs 0.18–0.20.
  - Takeaway (bold): the network has to train on the solver's objective, not on fit.
- **Visual:** two example pixels side by side (`slides/img/loss_pixels.png`, cropped from `results/plots/01_multiloss_20260609/loss_comparison.png`): BP in black, the barrier losses on top of it, the fit-only losses adding a hot component BP does not need.
- **Notes:** Averages over 4 timestamps. AIA MAE against the observation: BP 5.2, barrier 5.0, barrier + fit 2.6, fit-only 0.40–0.44. DEM MAE against BP: barrier 0.034, barrier + fit 0.043, fit-only 0.18–0.20. W1 on the brightest 5% of pixels: 0.030 and 0.033 vs 0.15–0.18. Six equations, eighteen unknowns: many DEMs fit within noise, so whatever breaks the tie decides the shape. L-BFGS and Adam reach the same curves; SGD often fails to converge in the same budget.

## Section 3: Architecture on small data (draft)

All on 4 images, each a crop near disk centre (128² for the CNNs, 256² for the first MLP):
taken during the X2.1 flare of 2011-09-06 (AR 11283), quiet Sun on 2012-06-03,
moderate activity on 2013-11-13, and during the X1.6 flare of 2014-09-10 (AR 12158).
The crops start at AIA pixel (1800, 1800), so they need not contain the flare cores.

### Slide 9 · Divider

- **Part 3:** Architecture on small data

### Slide 10 · First network, then the patch CNN

- **Title:** A per-pixel MLP missed BP's shape; a patch CNN tracked it
- **On slide:**
  - Top line (muted): 4 images: X2.1 flare (2011), quiet Sun (2012), moderate activity (2013), X1.6 flare (2014); small crops near disk centre.
  - First try: the 6 intensities of one pixel → MLP (213k params) → 54 weights, trained on the BP loss over ~221k pixels. Smooth, single-peaked curves, but further from BP: DEM error 0.078 vs the CNN's 0.047.
  - Patch CNN: a 9×9 neighbourhood → CNN (~1.5M params) → 54 weights. Tracked BP's shape, with sparsity close to BP's on held-out pixels.
  - Trained on one image, then on all 4 jointly; zeroing the neighbourhood for 10% of batches kept it usable on a single pixel.
- **Visual:** `slides/img/first_mlp_vs_cnn.png` (from `results/plots/15_first_mlp_repro_20261002/`): BP, first MLP and patch CNN on three held-out pixels of the X2.1 flare image.
- **Notes:** The June write-up called the first MLP's curves noisy and oscillating, but no figure of them survives and its checkpoint was not kept. Retrained with the unchanged June code (job 19077429), its curves are smooth: 0% have 3+ peaks (BP 0.25–0.5%). The real difference is accuracy, scored on 400 held-out pixels per image (job 19095906): mean DEM error vs BP 0.078 against the CNN's 0.047, W1 0.055 against 0.048 dex. The first MLP trained on a 256² crop with no held-out split, so it had seen these pixels and was still further from BP. It is the same architecture as today's production model, at width 256 with raw intensities.
- **Status:** draft.

### Slide 11 · Ablation

- **Title:** Ablation: the gap was capacity, not missing context
- **On slide:**
  - Four variants at ~1.45M params, same BP loss, same 4 images.
  - Size, not neighbours, closed the gap: DEM error vs BP 0.042 for the 1.43M centre-pixel MLP, 0.047 for the patch CNN, 0.078 for the first MLP.
  - The patch CNN was sparsest, but measured on held-out pixels of the training images.
- **Visual:** dot plot of mean sparsity (effective number of active basis weights, lower = sparser) with BP's 1.79 as a dashed line: patch CNN 1.70, shuffled-patch CNN 1.88, centre-pixel MLP 1.89, flat-patch MLP 1.95.
- **Notes:** The shuffled-patch CNN (a fixed random permutation of the 81 patch pixels) was less sparse than the CNN on all 4 images (1.88 vs 1.70 on average, level with the centre-pixel MLP), so the spatial arrangement of the neighbours mattered, not just their values. The flat-patch MLP was erratic across images (1.61 on the X2.1 flare, 2.27 on quiet Sun). The centre-pixel MLP was closest to BP on 3 of 4 images (DEM error 0.033/0.020/0.041/0.075 vs CNN 0.045/0.030/0.055/0.058); the CNN was closest on the X1.6 flare. So the CNN's lower sparsity did not mean DEMs closer to BP. Capacity and input scaling were not separated: the 213k MLP took raw intensities, and at full scale a 176k MLP on log1p inputs does well.
- **Status:** draft.

### Slide 12 · Leave one day out

- **Title:** On unseen days, the patch CNN's edge disappeared
- **On slide:**
  - Leave one image out: train on 3, test on the 4th; patch CNN and centre-pixel MLP, 4 folds.
  - Neither is consistently closer to BP's sparsity on the unseen image; the MLP fits AIA better on 3 of 4.
  - Little overfitting (held-out within ±0.2 of in-sample on 3 folds), except the X1.6 flare: peaks too cool (logT ~6.1 vs BP ~6.4), with one other flare image in training.
  - Takeaway (bold): carry the simpler centre-pixel MLP forward, and get more flare data → scaling.
- **Visual:** `slides/img/loo_flare.png`: the held-out X1.6 flare image, both models (dashed) against BP.
- **Notes:** Closer to BP's sparsity: CNN on 2 images, MLP on 1, tie on 1; the MLP is closer on average (mean distance 0.19 vs 0.25) because of the X1.6 flare image. Held-out sparsity by fold, MLP vs CNN (BP): X2.1 flare 1.59 vs 1.64 (1.97); quiet Sun 1.78 vs 1.61 (1.67); moderate 1.83 vs 1.83 (1.82); X1.6 flare 1.96 vs 2.31 (1.71). The ablation's in-sample edge (1.70 vs 1.89) came from scoring pixels of images the model trained on. Four folds is a small sample; scaling is the real test (Part 4).
- **Status:** draft.

## Section 4: Scaling to the full dataset (draft)

### Slide 13 · Divider

- **Part 4:** Scaling to the full dataset

### Slide 14 · The collapse

- **Title:** The first full-disk runs collapsed to an all-zero DEM
- **On slide:**
  - Full-disk intensities span ~7 decades (10⁻³ off-limb to 10⁴ in flare cores); the small crops spanned about one.
  - Fed raw, the untrained network predicted AIA ~7× outside the noise band: loss 5×10¹³ at step 0. The output softplus underflowed, its gradient became exactly zero, and every pixel predicted zero.
  - Fix: log1p on the input only (loss and metrics stay in physical units), learning-rate warmup, a clamped softplus whose gradient cannot vanish, and a guard that stops a run whose loss freezes.
  - Bold: a 64-block smoke test could not catch it: it never drew a flare pixel.
- **Visual:** a small strip: raw input range 10⁻³ … 10⁴ → log1p → 0 … 9.2.
- **Notes:** Three submissions died the same way before the fix; the cause was visible at step 0, before any weight update. Gradient clipping does not help (Adam normalises the step).
- **Status:** draft.

### Slide 15 · How we evaluate

- **Title:** How we score at scale
- **On slide:**
  - 153 test days, never seen in training; each scored on random quarters of the image against 5 solver targets (the clean solve and 4 noisy re-solves).
  - Bright = any channel above its top-5% cutoff (~10% of pixels); the rest is Quiet.
  - DEM against the solver: MSE, total-emission error, W1 (temperature-shape distance). AIA reconstruction against the observation: MAE, MSE.
  - We also report medians and percentiles: the error is extremely heavy-tailed (one pixel in 5 million once made up 94% of a model's mean loss).
- **Visual:** none.
- **Notes:** The Bright cutoffs and the test set come from the supervised side's protocol, so both models are scored on identical pixels. W1 treats each DEM as a distribution over logT.
- **Status:** draft.

## Section 5: Experiments at scale (draft; numbers pending the fresh-pixel rerun)

### Slide 16 · Divider

- **Part 5:** Experiments at scale

### Slide 17 · CNN vs MLP at scale

- **Title:** At scale, the centre-pixel MLP matched the patch CNN
- **On slide:**
  - Same 917 training days, same losses, ~1.5M parameters each; scored on the 153 test days.
  - Table: sparsity, AIA MAE and training objective for MLP vs CNN on both tracks.
  - Bold: the simplest model scales: no neighbourhood needed.
  - Muted: numbers shown are from the earlier fixed-sample runs; the fresh-pixel rerun (job 19108218) replaces them.
- **Visual:** small table. Fixed-sample (BP): sparsity 1.90 vs 1.96 (BP 1.79), objective 2.145 vs 2.241, AIA MAE 4.88 vs 4.84; (ENet, alpha=1): 3.68 vs 3.61, 1.850 vs 1.858, 4.61 vs 4.75.
- **Notes:** Fourth independent look at CNN vs MLP (ablation, leave-one-out, 30-epoch run, converged at scale); none found a CNN advantage on unseen days.
- **Status:** draft, pending.

### Slide 18 · Width sweep

- **Title:** 176k parameters are enough
- **On slide:**
  - MLP widths from 10k to 1.43M parameters (and 2.8M–11M once, earlier), both tracks.
  - Below 176k, detection of multi-peaked pixels collapses; above it, nothing we report improves.
  - Bold: production model: 4 hidden layers of 232, 176k parameters.
  - Muted: earlier fixed-sample sweep; the fresh-pixel sweep (array 19108189) replaces it.
- **Visual:** line chart: multi-peak recall vs parameter count (fixed-sample: 10k 21%, 20k 8%, 42k 9%, 87k 8%, 176k 26%, 360k 27%, 722k 28%, 1.43M 29%).
- **Notes:** Average metrics alone would have picked a much smaller model: AIA MAE kept improving as recall collapsed. 10k fires at the base rate (guessing).
- **Status:** draft, pending.

## Section 6: Results (draft)

### Slide 19 · Divider

- **Part 6:** Results

### Slide 20 · Against the supervised model

- **Title:** Against the supervised model: worse DEM MSE, better reconstruction
- **On slide:** table, fresh-pixel label-free vs supervised, both tracks (lower is better, winner in bold):
  - BP: DEM MSE 5.28 vs **0.91**; EM error 25.5% vs **14.3%**; W1 **0.091** vs 0.133; AIA MAE 4.35 vs **2.95**; AIA MSE **93** vs 157.
  - ENet: DEM MSE 0.79 vs **0.32**; EM error **14.7%** vs 18.0%; W1 0.126 vs **0.121**; AIA MAE **0.72** vs 6.32; AIA MSE **414** vs 598.
  - Muted: the earlier fixed-sample models scored better on DEM MSE (BP 4.19, ENet 0.59).
- **Visual:** the table.
- **Notes:** Label-free trains on the observation through R; supervised trains on solver DEMs. So supervised wins the DEM-vs-solver metric, label-free wins reconstruction of the observation on most counts.
- **Status:** draft.

### Slide 21 · Mean vs median

- **Title:** The typical pixel and the average tell different stories
- **On slide:**
  - BP, median pixel error: label-free 0.030, supervised 0.112: ours ~4× better on the typical pixel.
  - From the 90th percentile up, supervised is better (p99 27 vs 117).
  - The worst 1% of pixels hold 97% of our BP error (supervised 95%).
  - ENet: supervised better at the median too (1.12 vs 1.70).
- **Visual:** two stat callouts (median BP: 0.030 vs 0.112; worst 1%: 97% of error).
- **Notes:** Per-pixel error = squared error summed over 18 bins.
- **Status:** draft.

### Slide 22 · Multi-peaked DEMs

- **Title:** Multi-peaked DEMs: we flag them precisely but find a quarter
- **On slide:**
  - BP's DEM has 2+ peaks on 13.5% of test pixels.
  - Label-free flags 5.0% of pixels: 69% are right (precision), finding 25% of the real ones (recall).
  - Supervised flags 41%: 24% precision, 72% recall.
  - Where BP is multi-peaked, our gap is about the size of BP's own re-solve scatter (pending: job 19108236).
- **Visual:** small 3-row table (solver / label-free / supervised: flagged %, precision, recall).
- **Notes:** Peaks counted on the zero-padded curve with prominence 0.15 × max, over the full test set. The fixed-sample model had the same recall (25%) at lower precision (46%).
- **Status:** draft, last point pending.

## Section 7: Failures: bright pixels (draft)

### Slide 23 · Divider

- **Part 7:** Failures: bright pixels

### Slide 24 · Where the error is

- **Title:** Flare cores, 1 pixel in 10,000, hold ~90% of the error
- **On slide:**
  - Pixels at 32× the Bright cutoff or more: 0.01% of pixels, 88% of our BP DEM error, 90% of the supervised model's.
  - So the squared-error metric itself is dominated by flare cores, for both models.
  - Ordinary Bright pixels (1–10×): 10% of pixels, 5.5% of the error, for both.
- **Visual:** `fig1_error_share_by_brightness.png` (fresh-pixel version).
- **Status:** draft.

### Slide 25 · Gap grows with brightness

- **Title:** Relative to DEM size, our gap grows with brightness
- **On slide:**
  - Our BP error / supervised error: 0.65 on the faintest pixels, 2.1 at 0.3–1×, 9 at 3–10×, 25 at 10–32×.
  - Supervised gets relatively better as pixels brighten; we get worse above ~3×.
  - ENet: ratio 1.3–2.5 below 10×, ~6 at 10–32×.
- **Visual:** `fig2_relative_error_by_brightness.png`.
- **Status:** draft.

### Slide 26 · At flare cores

- **Title:** At flare cores, the model under-predicts and runs too cool
- **On slide:**
  - BP flare cores: 15% of the solver's emission, peak 12% as high, 0.6 dex too cool.
  - 89% of our flare-core DEMs have extra peaks the solver doesn't (solver 18%); those spurious peaks are 81% of our BP error.
  - It tracks the solver's peak height up to a peak of ~50, then falls away.
- **Visual:** `fig4_example_curves_bp.png` (random flare-core and ordinary Bright pixels).
- **Notes:** ENet flare cores are milder: 73% of emission, 40% multi-peaked. The fixed-sample BP model was less extreme (41% of emission).
- **Status:** draft.

### Slide 27 · Its own objective

- **Title:** At flare cores, the model fails its own objective
- **On slide:**
  - The solver's DEM reproduces 92% of the observed 94 Å; ours reproduces 7%, and 35% of 131 Å.
  - The cool channels (171, 193, 211 Å) are fit to within a few percent.
  - So better DEMs exist inside our basis: this is a training problem, not a representation limit.
- **Visual:** `fig5_aia_fit_flare_cores.png`.
- **Notes:** The sqrt + Fourier-feature input (the supervised model's) made flare cores worse, consistent with this.
- **Status:** draft.

### Slide 28 · Ruled out

- **Title:** Not solver noise, not missed double peaks
- **On slide:**
  - Solver noise: the spread across the 5 solver targets is 0.3% of our Bright error (ENet 4.8%).
  - Missing the solver's second peak: 1.3% of our BP error.
  - What remains is the flare-core shortfall and spurious peaks.
- **Visual:** none.
- **Status:** draft.

## Section 8: Lessons and next steps (draft)

### Slide 29 · Divider

- **Part 8:** Lessons and next steps

### Slide 30 · Lessons

- **Title:** What we learned
- **On slide:**
  - Training on the solver's objective reproduces its DEMs without labels.
  - The simplest model scales: a 176k-parameter centre-pixel MLP.
  - The error is extremely heavy-tailed: report medians and percentiles, not just means.
  - The remaining failure is the rare extreme, flare cores, and it is a training problem.
  - Tried, didn't help: the supervised model's sqrt + Fourier input (flare cores worse); fresh pixels every epoch scored worse than one fixed sample.
- **Visual:** none.
- **Status:** draft.

### Slide 31 · Next

- **Title:** Next
- **On slide:**
  - Flare cores: oversample or up-weight the brightest pixels; handle the pathological pixels that dominate the loss.
  - Predict the 18 DEM bins directly instead of basis weights (deferred).
  - An uncertainty head: a distribution over DEMs, trained on the solver's noisy re-solves.
  - AIA + XRT, for the hot plasma AIA barely constrains.
- **Visual:** none.
- **Status:** draft.
