# Advisor deck: slide outline

Master outline: one block per slide. This file and the deck
(`slides/DEM_deck_v2.pptx`, built by `slides/build_deck_v2.py`) are kept in
sync: change this file first, then the deck is rebuilt from it.

Each slide: **Title** (a takeaway sentence), **On slide** (the actual text, kept
short), **Visual** (what the image or diagram shows, if any), **Notes** (what to
say; goes into speaker notes, not onto the slide). Each part opens with a
divider slide ("Part N" over the section name, template's BIG_NUMBER layout),
with a one-line subtitle: the question that part answers.

Edits made in Google Slides get copied back here and into the build script
before the next rebuild, or the rebuild would discard them.

---

## Section 1: Problem and approach (settled)

### Slide 1 · Title

- **Title:** Predicting DEM label-free

### Slide 1b · Summary

- **Title:** In one slide
- **On slide (four cards):**
  - No labels needed: a network trained on the solver's own objective reproduces its DEMs; solver DEMs are only used to evaluate.
  - Small is enough: a 360k-parameter MLP on one pixel's 6 intensities, one forward pass per pixel, no neighbourhood.
  - Beats the supervised model on ElasticNet: better on every metric, DEM error included; BP typical-pixel error 4× lower (median 0.026 vs 0.112).
  - Flare cores are the open problem: 0.01% of pixels, ~90% of the BP error; there our BP model misses the hot channels, a training problem.
- **Visual:** 2×2 cards.
- **Notes:** Final models: one 360k MLP (h336) per track, fresh pixels, chosen 2026-10-04 (Part 5); BP with log1p input, ElasticNet with the supervised model's square-root + Fourier-feature input.
- **Status:** draft.

### Slide 2 · Divider

- **Part 1:** Problem and approach
- **Subtitle:** What we train, and on what

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
  - 6 intensities in (log1p for BP; √ + Fourier features for ElasticNet); 4 hidden layers of 336, SiLU; 54 non-negative basis weights out.
  - BP and ElasticNet solve for the same 54 weights (at each of 18 bins, a spike and Gaussians 0.1 and 0.2 dex wide), and their L1 term acts on those weights.
  - DEM = B·w over 18 bins, logT 5.5 to 7.2; predicted AIA = R·DEM. B and R are fixed; only the MLP is learned.
  - 360k parameters (size chosen in Part 5); one forward pass per pixel.
- **Visual:** pipeline diagram: observed AIA, MLP ("360k params"), basis weights ("54, same as solver"), DEM, predicted AIA, with the loss closing the loop.
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
  - Earlier runs reused one fixed ~30M-pixel sample (a sampler bug); at the old 176k size it scored better on validation (BP 4%, ElasticNet 18%).
  - Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.
  - Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves.
- **Visual:** a timeline of the split, to scale: train Jan 2014 – Jun 2015 (917 images), validation Jun – Sep 2015 (153), test Sep – Dec 2015 (153).
- **Notes:** Every run before October seeded each block's pixel draw by the block alone, so it reused the same 30M pixels every epoch. All results from here on use fresh pixels every epoch, as described. The fixed-sample runs scored better (validation loss BP 2.156 vs 2.237, ElasticNet 0.0582 vs 0.0689, far beyond the 0.2–0.4% run-to-run spread); only ElasticNet's flare cores improved with fresh pixels. The split is chronological, so every test image is later than anything seen in training. Each epoch draws 512 pixels from each of 58,688 training blocks (917 images × 64 blocks of 256²). The test set uses 64 random 128² blocks of each test image per solver target, five targets per image: one clean solve and four re-solves under simulated photon noise of the same observation. Pixels where deconvolution clamped a bright channel to zero (about 5%) are excluded.
- **Status:** draft.

---

## Section 2: Finding a trainable objective (draft)

### Slide 7 · Divider

- **Part 2:** Finding a trainable objective
- **Subtitle:** Which loss gives BP's DEMs?

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
- **Subtitle:** Which network, on four images

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
- **Subtitle:** What broke at full scale, and how we score

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
  - We also report medians and percentiles: the error is extremely heavy-tailed. For our final BP model, the worst 1% of pixels hold 98% of the DEM error, and the flare cores (0.01% of pixels, 1 in 10,000) alone hold 90%.
- **Visual:** diagram of one test day: the 2048² DEM image as a 16×16 grid of 128² blocks, 64 of them highlighted (one target's random quarter), with "× 5 targets: the clean solve + 4 noisy re-solves, each with its own 64 blocks".
- **Notes:** The Bright cutoffs and the test set come from the supervised side's protocol, so both models are scored on identical pixels. W1 treats each DEM as a distribution over logT.
- **Status:** draft.

## Section 5: Experiments at scale (draft)

### Slide 16 · Divider

- **Part 5:** Experiments at scale
- **Subtitle:** CNN or MLP, and how big

### Slide 17 · CNN vs MLP at scale

- **Title:** At scale, the centre-pixel MLP beat the patch CNN
- **On slide:**
  - Same 917 training days and losses, matched at ~1.5M parameters each, fresh pixels; scored on the 153 test days. (The final model is smaller: next slide.)
  - Table (test split): training objective BP 2.07 vs 2.24, ENet 0.057 vs 0.245; BP objective median / p99 0.98 / 17.9 vs 1.01 / 18.8; sparsity (BP 1.79) 1.90 vs 2.03, ENet 5.53 vs 5.17; AIA MAE BP 4.93 vs 4.73, ENet 0.54 vs 0.82.
  - Bold: the simplest model scales: no neighbourhood needed.
- **Notes:** The MLP is better on the objective at every percentile and every brightness decile on BP; the CNN only reconstructs BP's AIA slightly better. Fifth comparison, none found a CNN advantage on unseen days. Validation agrees: BP 2.155 vs 2.248, ENet 0.0528 vs 0.0683. Source: `results/plots/18_cnn_vs_mlp_resample_20261003/`.
- **Status:** draft.

### Slide 18 · Choosing the model

- **Title:** Choosing the model: 360k parameters, for both tracks
- **On slide:**
  - We choose by validation loss.
  - BP: validation is best at 360k; 1.43M ties, but only nudges the typical pixel (0.026 → 0.023) at 4× the size and with worse tails.
  - ElasticNet: from 722k up, a few bright pixels blow up by orders of magnitude; 360k is the largest stable size.
  - Input, also by validation: log1p for BP (√ + Fourier ties); √ + Fourier features for ElasticNet (7% better).
  - Bold: Final: one 360k MLP per track; ElasticNet with the √ + Fourier input.
- **Visual:** line chart: BP median pixel error by size (20k 0.068 … 360k 0.026 … 1.43M 0.023), 360k highlighted; 10k collapsed, left off.
- **Notes:** BP validation 2.148 at 360k, 2.155 at 1.43M (tie), 2.167 at 722k, 2.237 at 176k. ElasticNet validation best at 1.43M (0.0528) but 722k and 1.43M blow up on a few bright pixels (AIA MSE 41,849 and 908,611 vs 650 at 360k); validation scores a fixed 512 pixels per block and missed them. Multi-peak recall 22–26% from 42k up. With the old fixed sample nothing above 176k helped. Input: BP 2.1472 (√ + Fourier) vs 2.1478 (log1p), a tie, so log1p stays; ElasticNet 0.0533 vs 0.0575. Sources: `results/plots/17_sweep_resample_20261003/sweep_table.txt`, `20_sqrtff_final_20261004/`.
- **Status:** draft.

## Section 6: Results (draft)

### Slide 19 · Divider

- **Part 6:** Results
- **Subtitle:** Against the solver and the supervised model

### Slide 20 · Against the supervised model

- **Title:** Against supervised: better on ElasticNet, behind on BP's DEM error
- **On slide:** table, final label-free vs supervised (winner in bold):
  - BP: DEM MAE 0.106 vs **0.100**; DEM MSE 4.28 vs **0.91**; EM error 19.6% vs **14.3%**; W1 **0.083** vs 0.133; AIA MAE 4.34 vs **2.95**; AIA MSE **83** vs 157.
  - ENet (√ + Fourier input): DEM MAE **0.151** vs 0.188; DEM MSE **0.31** vs 0.32; EM error **12.2%** vs 18.0%; W1 **0.103** vs 0.121; AIA MAE **0.56** vs 6.32; AIA MSE **161** vs 598.
  - Point: ElasticNet: we win all six, DEM MAE and MSE included. BP: we win W1 and AIA MSE; the DEM error is set by flare cores (Part 7).
- **Status:** draft.

### Slide 21 · Mean vs median

- **Title:** The typical pixel and the average tell different stories
- **On slide:**
  - On the typical BP pixel we are ~4× closer to the solver than the supervised model (median 0.026 vs 0.112).
  - From the 90th percentile up, supervised is better on BP (p99: 27 vs 71).
  - ElasticNet: ours is better at the median (0.85 vs 1.12), p99 (30 vs 35) and p99.9 (102 vs 178).
  - The BP average is set by the worst 1% of pixels (98% of our error, 95% of supervised's).
- **Visual:** two stat callouts (0.026 vs 0.112; 98%).
- **Status:** draft.

## Section 7: Failures: bright pixels (draft)

### Slide 22 · Divider

- **Part 7:** Failures: bright pixels
- **Subtitle:** Where the error is, and why

### Slide 23 · Where the error is

- **Title:** Flare cores, 1 pixel in 10,000, hold ~90% of the BP error
- **On slide:** ≥32×: 0.01% of pixels, 90% of our BP error, 90% of supervised's; the metric itself is set by flare cores; ordinary Bright (1–10×): 10% of pixels, 4% of our error (supervised 5.5%).
- **Visual:** `results/plots/19_final_models_20261004/fig1_error_share_by_brightness.png`.
- **Status:** draft.

### Slide 24 · Gap by brightness

- **Title:** On BP, our gap grows with brightness; on ElasticNet we are ahead
- **On slide:** BP ratio 0.64 faintest, 1.8 at 0.3–1×, 5 at 3–10×, 18 at 10–32×; ElasticNet: ours ahead from 0.1× to 32× (ratio 0.5–1.0), level on the faintest, 1.3 at flare cores.
- **Visual:** `fig2_relative_error_by_brightness.png` (final models).
- **Status:** draft.

### Slide 25 · At flare cores

- **Title:** At BP flare cores, the model under-predicts and runs too cool
- **On slide:** BP flare cores: 40% of the solver's emission, peak 23% as high, 0.2 dex too cool; 83% have extra peaks (solver 18%), 42% of our BP error; tracks the solver's peak up to ~100, then falls away.
- **Visual:** `fig4_example_curves_bp.png` (final models).
- **Notes:** ElasticNet flare cores match the solver: 99% of emission, peak 101%, 18% multi-peaked vs 21%.
- **Status:** draft.

### Slide 26 · Its own objective

- **Title:** At BP flare cores, the model fails its own objective
- **On slide:** the solver reproduces 92% of observed 94 Å, ours 23% (131 Å: 76%); cool channels within a few percent; bold: better DEMs exist in our basis, a training problem.
- **Visual:** `fig5_aia_fit_flare_cores.png` (final models).
- **Notes:** On ElasticNet our flare-core reconstruction matches the solver's (94 Å 0.53 vs 0.55) and beats supervised's on the cool channels. On BP the √ + Fourier input does not fix flare cores either (94 Å 0.15 vs 0.23).
- **Status:** draft.

### Slide 27 · Ruled out

- **Title:** Not solver noise, not missed double peaks
- **On slide:** solver spread is 0.4% of our BP Bright error (ENet 8%); missing BP's second peak is 1.2% of our BP error; what remains is the BP flare-core shortfall and spurious peaks.
- **Status:** draft.

## Section 8: Next steps (draft)

### Slide 28 · Divider

- **Part 8:** Next steps
- **Subtitle:** What we learned, and what to decide

### Slide 29 · Lessons

- **Title:** What we learned
- **Visual:** six cards: Train on the objective / Simple scales / Size needs data (fixed sample: nothing above 176k helped; fresh pixels: size pays off up to 360k) / Heavy tails (a handful of pixels can own the mean, or blow up) / The open failure (BP flare cores, training problem) / Input encoding (√ + Fourier input lifts ElasticNet past supervised; neutral on BP).
- **Status:** draft.

### Slide 30 · Next

- **Title:** Next
- **Visual:** four cards: Flare cores (BP up-weight brightest; ENet keep big models stable) / Validate on every pixel (sampled validation missed the blow-ups) / Direct DEM output (deferred) / Uncertainty, AIA + XRT.
- **Status:** draft.

### Slide 31 · For discussion

- **Title:** For discussion
- **On slide:**
  - Which metric should lead the paper? DEM MSE is set by 0.01% of pixels; the median, W1 and AIA reconstruction tell a different story.
  - Is fixing flare cores worth changing the training objective, or do we report it as the known limitation?
  - Next model change: up-weight flare cores, or predict the 18 bins directly?
  - Is AIA + XRT in scope for this paper?
- **Visual:** none (numbered questions).
- **Status:** draft.
