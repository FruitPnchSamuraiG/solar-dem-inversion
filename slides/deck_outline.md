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
  - Our model trains on AIA and σ only: each training image is cut into 64 blocks of 256×256 per image. Each epoch draws 512 random pixels from each of the 58,688 blocks(917*64), for 40 epochs.
  - Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.
  - Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves(only supervised).
- **Visual:** a timeline of the split, to scale: train Jan 2014 – Jun 2015 (917 images), validation Jun – Sep 2015 (153), test Sep – Dec 2015 (153).
- **Notes:** The split is chronological, so every test image is later than anything seen in training. Each epoch draws 512 pixels from each of 58,688 training blocks (917 images × 64 blocks of 256²). The test set uses 64 random 128² blocks of each test image per solver target, five targets per image: one clean solve and four re-solves under simulated photon noise of the same observation. Pixels where deconvolution clamped a bright channel to zero (about 5%) are excluded.
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

All on 4 images: two X-class flares (X2.1, X1.6), quiet Sun, moderate activity.

### Slide 9 · Divider

- **Part 3:** Architecture on small data

### Slide 10 · First network, then the patch CNN

- **Title:** A per-pixel MLP drew noisy DEMs; a patch CNN fixed it
- **On slide:**
  - First try: the 6 intensities of one pixel → MLP (213k params) → 54 weights, trained on the BP loss over ~221k pixels. The loss converged, but per-pixel curves oscillated.
  - Patch CNN: a 9×9 neighbourhood → CNN (~1.5M params) → 54 weights. Smooth, single-peaked curves at BP's temperatures, sparsity close to BP's on held-out pixels.
  - Trained on one image, then on all 4 jointly; zeroing the neighbourhood for 10% of batches kept it usable on a single pixel.
- **Visual:** `slides/img/patch_cnn.png`: patch CNN (dotted) against BP, three pixels of the X2.1 flare image. No figure of the noisy MLP exists: its checkpoint was not kept (the v1 deck's "noisy_nn.png" actually shows the direct-optimisation baselines, not the network).
- **Notes:** The first MLP is the same architecture as today's production model (4 hidden layers, SiLU, softplus), at width 256 with raw intensities as input. Its noise was attributed at the time to the overlapping basis amplifying small weight errors and to the problem's non-uniqueness.
- **Status:** draft.

### Slide 11 · Ablation

- **Title:** Ablation: the noise was capacity, not missing context
- **On slide:**
  - Four variants at ~1.45M params, same BP loss, same 4 images.
  - A centre-pixel MLP this size was smooth too: the first MLP's noise was not about the neighbourhood.
  - The patch CNN was sparsest, but measured on held-out pixels of the training images.
- **Visual:** dot plot of mean sparsity (effective number of active basis weights, lower = sparser) with BP's 1.79 as a dashed line: patch CNN 1.70, shuffled-patch CNN 1.88, centre-pixel MLP 1.89, flat-patch MLP 1.95.
- **Notes:** The shuffled-patch CNN (a fixed random permutation of the 81 patch pixels) nearly matches the CNN, so the gain came from neighbouring values more than geometry. The flat-patch MLP was erratic across images (1.61 on the X2.1 flare, 2.27 on quiet Sun). Capacity and setup were never fully separated: the 213k MLP took raw intensities; at full scale a 176k MLP on log1p inputs is smooth.
- **Status:** draft.

### Slide 12 · Leave one day out

- **Title:** On unseen days, the patch CNN's edge disappeared
- **On slide:**
  - Leave one image out: train on 3, test on the 4th; patch CNN and centre-pixel MLP, 4 folds.
  - Held-out sparsity: CNN lower on 1 day, MLP on 2, tied on 1; the MLP fit AIA better on 3 of 4.
  - Little overfitting (held-out within ±0.2 of in-sample on 3 folds), except the X1.6 flare: peaks too cool (logT ~6.1 vs BP ~6.4), with one other flare image in training.
  - Takeaway (bold): carry the simpler centre-pixel MLP forward, and get more flare data → scaling.
- **Visual:** `slides/img/loo_flare.png`: the held-out X1.6 flare image, both models (dashed) against BP.
- **Notes:** Held-out sparsity by fold, MLP vs CNN (BP): X2.1 flare 1.59 vs 1.64 (1.97); quiet Sun 1.78 vs 1.61 (1.67); moderate 1.83 vs 1.83 (1.82); X1.6 flare 1.96 vs 2.31 (1.71). The ablation's in-sample edge (1.70 vs 1.89) came from scoring pixels of images the model trained on. Four folds is a small sample; scaling is the real test (Part 4).
- **Status:** draft.

## Section 4: Scaling to the full dataset

*(next)*

## Section 5: DEM shape, multi-peaked pixels

*(next)*

## Section 6: Against the supervised model

*(next)*

## Section 7: Why bright pixels fail

*(next)*

## Section 8: Lessons and next steps

*(next)*
