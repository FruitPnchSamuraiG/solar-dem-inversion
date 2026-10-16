# Advisor deck: slide outline

Master outline: one block per slide. This file and the deck
(`slides/DEM_deck_v2.pptx`, built by `slides/build_deck_v2.py`) are kept in
sync: change this file first, then the deck is rebuilt from it.

The deck is built for a 30-minute talk (2026-10-21). Parts 1 to 5 are the main
line: 20 content slides, about 27 minutes of talking, questions taking the rest.
The appendix (from slide 27) holds backup slides, pulled up only when asked: the
history of the model choice, the collapse, the scoring diagram, ElasticNet detail
and multi-peaked DEMs. The prep sheet for the talk is `slides/talk_prep.md`.

Each slide: **Title** (a takeaway sentence), **On slide** (the actual text, kept
short), **Visual** (what the image or diagram shows, if any), **Notes** (what to
say, opening with the slide's time budget; goes into speaker notes, not onto the
slide; backup slides keep their documentation notes). Each part opens with a
divider slide ("Part N" over the section name, template's BIG_NUMBER layout),
with a one-line subtitle: the question that part answers.

Edits made in Google Slides get copied back here and into the build script
before the next rebuild, or the rebuild would discard them.

---

## Part 1: Problem and approach

### Slide 1 · Title
- **Title:** Predicting DEM label-free

### Slide 2 · Summary
- **Title:** Brief Summary
- **On slide (four cards):**
  - No labels needed: a network trained on the solver's own objective reproduces its DEMs; solver DEMs are only used to evaluate.
  - Small is enough: a 400k-parameter MLP on one pixel's 6 intensities, one forward pass per pixel, no neighbourhood. Bigger models fail on rare pixels.
  - Ahead of supervised on most measures: ElasticNet better on every metric; BP 4 of 6, median pixel error 3× lower (0.036 vs 0.112), DEM MSE still 1.9× higher.
  - Flare cores: found and fixed in the loss. 0.01% of pixels, ~94% of the BP error; the BP loss priced hot plasma out; a stiffer band puts it back (emission 40% → 93%). Without flare cores, our BP DEM MSE is only 1.1× the supervised model's.
- **Visual:** 2×2 cards.
- **Notes:** [2 min] Four things to remember; the rest of the talk is the evidence. One: we never train on a solver DEM. The network minimises the solver's own objective, and solver DEMs are only used to score. Two: a 400k-parameter MLP on one pixel's six intensities is enough; bigger models fail on rare pixels. Three: against the supervised model, which trained on the same solver's DEMs and is scored on the same pixels, we are ahead on every ElasticNet measure and on four of six BP measures. Four: the one place BP lost, flare cores, was the loss and not training, and a stiffer noise band fixed it. If asked: the final models are one MLP of width 336 per track, square root plus Fourier input, fresh pixels every epoch; BP trains on the stiff band for 120 plus 120 epochs, ElasticNet for 40. Without flare cores our BP DEM MSE is 0.102 against supervised's 0.091.
- **Status:** draft.

### Slide 3 · Divider
- **Part 1:** Problem and approach
- **Subtitle:** What we train, and on what

### Slide 4 · Why label-free
- **Title:** Same physics, different training signal
- **On slide:**
  - Top line: Solvers (BP, ElasticNet) run one optimisation per pixel: accurate, but slow at full resolution.
  - Row labels: "Supervised: learns the solver's DEMs"; "Label-free (unsupervised): optimises the solver's objective directly".
- **Visual:** the two training loops as two rows of boxes, styled like the model diagram. Supervised: observed AIA, model, DEM, then a "loss" link to the solver's DEM. Label-free: observed AIA, model ("54 basis weights"), DEM ("18 bins, via basis B"), predicted AIA, then a "loss" link to the observed AIA. (The bold bottom line was dropped in Google Slides, 15 Oct.)
- **Status:** settled.
- **Notes:** [1 min] Both solvers run one optimisation per pixel. A supervised network copies their DEMs, so it needs them as labels. A label-free network minimises the same objective directly, so it never sees a solver DEM; we use those only to evaluate. That is the point of this slide. If asked: our network outputs 54 basis weights, as the solvers do, and the DEM is B times w; the supervised model outputs the 18 DEM values directly.

### Slide 5 · The model
- **Title:** The model: a small MLP feeding a fixed forward model
- **On slide:**
  - 6 intensities in, encoded as √ then 12 Fourier frequencies (150 inputs; both tracks); 4 hidden layers of 336, SiLU; 54 non-negative basis weights out.
  - BP and ElasticNet solve for the same 54 weights (at each of 18 bins, a spike and Gaussians 0.1 and 0.2 dex wide), and their penalty terms act on those weights: L1 for BP, L1 and L2 for ElasticNet.
  - DEM = B·w over 18 bins, logT 5.5 to 7.2; predicted AIA = R·DEM. B and R are fixed; only the MLP is learned.
  - 409k parameters (width chosen in Part 2); one forward pass per pixel.
- **Visual:** pipeline diagram: observed AIA, MLP ("409k params"), basis weights ("54, same as solver"), DEM, predicted AIA, with the loss closing the loop.
- **Notes:** [1.5 min] Six intensities in, encoded as the supervised model does it: square root, then 12 Fourier frequencies, 150 inputs. Four hidden layers of 336 with SiLU, 54 non-negative basis weights out. The 54 weights matter: neither solver solves for the 18 bins directly. Both solve for weights on a fixed basis, a spike and two Gaussians at each bin, and report DEM = B times w. Predicting the same weights means the network searches the same space of DEMs the solver does, and BP's sparsity means the same thing for both. B and R are fixed; only the MLP is learned. If asked: 409k parameters, width chosen in Part 2; with the earlier log1p input the same width was 360k.
- **Status:** settled.

### Slide 6 · The loss
- **Title:** The loss is the solver's own objective
- **On slide:**
  - BP track: L = Σ_k w_k + μ Σ_c [ max(0, |ŷ_c − o_c| − tσ_c) / tσ_c ]²
    - Zero anywhere inside the noise band (t = 1.4), quadratic outside; μ prices leaving it.
    - L1 picks the sparsest DEM that fits: BP's own criterion. Final model: μ = 100 (Part 4).
  - ElasticNet track: L = (1/2C) Σ_c [ (ŷ_c − o_c) / tσ_c ]² + αλ Σ_k w_k + ½α(1−λ) Σ_k w_k², with the solver's α = 0.001, λ = 0.5, C = 6.
  - In both, ŷ = R·B·w and w ≥ 0.
- **Visual:** the BP band penalty against prediction error: flat at zero inside the noise band, quadratic outside.
- **Notes:** [1.5 min] The BP loss is the solver's objective: the sum of the 54 weights plus a band penalty that is zero inside the noise band, o plus or minus 1.4 sigma, and quadratic outside. There is no fit term inside the band, so any DEM within noise is equally good and sparsity breaks the tie, exactly as BP does. One difference to hold on to for Part 4: BP treats the band as a rule, our loss treats it as a price, set by the band weight mu. It was 1; the final model uses 100. ElasticNet is the solver's own objective with its settings: alpha 0.001, l1 ratio 0.5, C equal to the six channels. If asked: t = 1.4 is the solver's first tolerance level; sigma comes from the AIA error model.
- **Status:** settled.

### Slide 7 · Data and scoring

- **Title:** Data and scoring: two years of full-disk AIA, split in time
- **On slide:**
  - SDO/AIA, 6 EUV channels, about two images a day in 2014–2015: 1,223 full-disk images, PSF-deconvolved, with a noise σ per pixel and channel. Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.
  - Training sees AIA and σ only, no noise added: 917 images, each cut into 256 blocks of 128²; a random 64 kept per image (a quarter, fixed). Every epoch draws 512 fresh pixels per block. 40 epochs (final BP model: 240, Part 4).
  - Test: 153 later days on the supervised side's protocol, the same for both models. Input: the clean AIA. Targets: 5 solver runs per image (the clean solve + 4 on noise-added copies), each scored on its own random quarter of the image: ~800M pixel DEMs (705M with a BP answer). Bright: any channel above its top-5% cutoff, ~10% of pixels.
  - Scores: DEM against the solver (MAE, MSE, total-emission error, W1 temperature-shape distance); AIA reconstruction against the observation (MAE, MSE).
  - And medians and percentiles, because the error is heavy-tailed: the worst 1% of pixels hold 98% of our BP error; flare cores (1 pixel in 10,000) alone hold 94%.
- **Visual:** a timeline of the split, to scale: train Jan 2014 – Jun 2015 (917 images), validation Jun – Sep 2015 (153), test Sep – Dec 2015 (153). (Merged from the old data and scoring slides; the block diagram of one test day is in the appendix, slide 33.)
- **Notes:** [1.5 min] Two years of full-disk AIA, 1,223 images, deconvolved, with a noise sigma per pixel and channel, split in time: train to June 2015, validation to September, test to December, so every test image is later than anything trained on. Training sees AIA and sigma only, with fresh pixels every epoch. Solver DEMs exist for every image: labels for the supervised model, evaluation only for us. We score on the supervised side's protocol: 153 test days, the same pixels and the same Bright cutoffs, five solver targets per image. Scores: DEM against the solver, AIA reconstruction against the observation, and, because the error is heavy-tailed, medians and percentiles. If asked: each image keeps a random 64 of its 256 blocks, chosen once; every epoch takes 512 fresh pixels from each, 30M in all. The noisy targets come from adding Gaussian noise of size sigma to the AIA before the solve; the input stays the clean AIA, so they measure the solver's own scatter, the same floor for both models. About 800M pixel-targets per solver, 705M where BP returned an answer. Pixels where deconvolution clamped a bright channel to zero, about 5%, are left out of training only. Off-disk corona pixels are real data and are used. Runs before October reused one fixed 30M-pixel sample through a seeding bug, fixed since.
- **Status:** draft.

## Part 2: Getting to the model

The history behind these two slides (which loss, the first MLP and the patch CNN, the
ablation, leave-one-day-out, the collapse) is in the appendix.

### Slide 8 · Divider
- **Part 2:** Getting to the model
- **Subtitle:** Which network, and how big

### Slide 9 · CNN vs MLP at scale
- **Title:** At scale, the centre-pixel MLP beat the patch CNN
- **On slide:**
  - Same 917 training days and losses, matched at ~1.5M parameters each, fresh pixels; scored on the 153 test days. (The final model is smaller: next slide.)
  - Two tables, one per track, MLP vs CNN, better cell shaded and bold (test split; ElasticNet percentiles from job 20005027):
    - BP: loss median 0.98 vs 1.01; p99 17.9 vs 18.8; mean 2.07 vs 2.24; AIA MAE 4.93 vs 4.73 (the CNN's one win); sparsity (solver 1.79) 1.90 vs 2.03.
    - ElasticNet: loss median 0.020 vs 0.033; p99 0.47 vs 0.52; mean 0.057 vs 0.245; AIA MAE 0.54 vs 0.82.
  - Caption: loss = each track's own training loss on the 153 test days; lower is better, except sparsity (closer to the solver's 1.79); shaded = better.
  - Bold: the simplest model scales: no neighbourhood needed.
- **Notes:** [1.5 min] On four images the patch CNN looked better than a per-pixel MLP, but that edge came from scoring pixels of images it had trained on; on held-out days it disappeared. At full scale, matched at 1.5M parameters, the MLP wins the training objective on both tracks and at every percentile. The CNN only reconstructs BP's AIA slightly better. This is the fifth comparison and none found a CNN advantage on unseen days, so the final model takes one pixel and no neighbourhood. On ElasticNet the MLP is lower at every percentile; the 4x gap in the mean is mostly a few CNN pixels, whose worst 0.1% hold 73% of its loss. If asked: the CNN sees a 9 by 9 patch, three 3 by 3 convolutions of 64 channels, then 256, 256 and 54, 1.48M parameters, against the width-680 MLP at 1.43M; the small-data story, the ablation and leave-one-day-out are in the appendix; validation agrees with test, BP 2.155 against 2.248, ElasticNet 0.0528 against 0.0683.
- **Status:** draft.

### Slide 10 · Choosing the model
- **Title:** Choosing the model: width 336, for both tracks
- **On slide:**
  - We choose on the 153 validation days, never on test.
  - BP: validation is best at width 336 (360k); 1.43M ties, but only nudges the typical pixel (0.026 → 0.023) at 4× the size, and on the stiff band (Part 4) it blows up on a few flare-core pixels.
  - ElasticNet: validation favours bigger models, but from 722k up a few bright pixels blow up by orders of magnitude on the full test set (too rare for the validation sample); 336 is the largest stable width.
  - Input, also by validation: √ + Fourier for ElasticNet (7% better) and, on the stiff band, for BP (validation DEM error 20.1 vs 29.3).
  - Bold: Final: one MLP of width 336 (409k parameters) per track, √ + Fourier input; BP on the stiff band (Part 4).
- **Visual:** line chart titled "Median pixel error on test (BP, log1p, μ = 1)": 20k 0.068 … 360k 0.026 … 1.43M 0.023, 360k highlighted; 10k collapsed, left off. Test numbers; the choice itself is made on validation.
- **Notes:** [1.5 min] We choose on the validation days: the training loss computed there, no solver DEM needed. For BP, width 336 is the validation best; 1.43M ties but only nudges the typical pixel at four times the size, and on the stiff band it blows up on a few flare-core pixels. ElasticNet is the one exception, and worth saying plainly: validation preferred the bigger models, but from 722k up they blow up on a handful of bright pixels, by orders of magnitude, that the validation sample never contained. We saw it on the test set, so 336 is the largest width that does not blow up; validating on every pixel is the clean fix. The input encoding is chosen on validation too: square root plus Fourier wins for ElasticNet by 7% and, on the stiff band, for BP. If asked: the chart is test numbers from the log1p sweep; BP validation 2.148 at 336 against 2.155 at 1.43M; ElasticNet validation 0.0575, 0.0533 and 0.0528 at 336, 722k and 1.43M, test DEM MSE 0.48, 3.24 and 36.2, AIA MSE 650, 41,849 and 908,611; the validation sample is about 131k pixels; the 10M models blew up on both tracks; 409k parameters with the 150 encoded inputs.
- **Status:** draft.

## Part 3: Results

### Slide 11 · Divider
- **Part 3:** Results
- **Subtitle:** Against the solver and the supervised model

### Slide 12 · Against the supervised model
- **Title:** Against supervised: all six on ElasticNet, four of six on BP
- **On slide:** table, final label-free vs supervised; every header carries ↓ (lower is better), the better cell in each track is shaded and bold, and a heavier rule separates BP from ElasticNet; BP label-free is the stiff-band model of Part 4:
  - BP: DEM MAE **0.074** vs 0.100; DEM MSE 1.75 vs **0.91**; EM error **11.0%** vs 14.3%; W1 **0.068** vs 0.133; AIA MAE 3.48 vs **2.95**; AIA MSE **103** vs 157.
  - ENet (√ + Fourier input): DEM MAE **0.151** vs 0.188; DEM MSE **0.31** vs 0.32; EM error **12.2%** vs 18.0%; W1 **0.103** vs 0.121; AIA MAE **0.56** vs 6.32; AIA MSE **161** vs 598.
  - Point: lower is better in every column; shaded = better. ElasticNet: we win all six, DEM MAE and MSE included. BP: we win DEM MAE, EM error, W1 and AIA MSE; supervised keeps DEM MSE (set by flare cores, Part 4) and AIA MAE.
  - Muted line: the old BP final, for reference: DEM MSE 4.28, EM error 19.6%, W1 0.083, AIA MSE 83.
- **Notes:** [2 min] Same solver DEMs, same test pixels; the supervised model trained on those DEMs, ours never saw one. ElasticNet: we win all six, DEM MAE and MSE included. BP: we win DEM MAE, total- emission error, W1 and AIA MSE; supervised keeps DEM MSE and AIA MAE. DEM MSE is the one to explain, and Part 4 does: it is set by flare cores. If asked: the BP row is the stiff-band model; the old BP final had DEM MSE 4.28, EM error 19.6%, W1 0.083, AIA MSE 83. EM error is the total-emission error; W1 is in dex.
- **Status:** draft.

### Slide 13 · Mean vs median
- **Title:** The typical pixel and the average disagree
- **On slide:**
  - On the typical BP pixel we are ~3× closer to the solver than the supervised model (median 0.036 vs 0.112).
  - Up to the 99th percentile we are level or ahead on BP (p90 1.9 vs 1.9, p99 25 vs 27); the gap opens in the last 0.1%: flare cores.
  - ElasticNet: ours is better at the median (0.85 vs 1.12), p99 (30 vs 35) and p99.9 (102 vs 178).
  - The BP average is set by the worst 1% of pixels (98% of our error, 95% of supervised's).
- **Visual:** two stat callouts (0.036 vs 0.112; 98%).
- **Notes:** [1.5 min] Per-pixel error is the squared error summed over the 18 bins. On the typical BP pixel we are three times closer to the solver than the supervised model, 0.036 against 0.112, and up to the 99th percentile we are level or ahead. The gap opens in the last 0.1%, the flare cores, and because the error is so heavy-tailed those pixels set the mean: the worst 1% hold 98% of our BP error and 95% of supervised's. ElasticNet: better at the median, p99 and p99.9. If asked: BP p90 1.9 against 1.9, p99 25 against 27, p99.9 170 against 162; ElasticNet median 0.85 against 1.12, p99 30 against 35, p99.9 102 against 178; the old BP final had median 0.026 and p99 71.
- **Status:** draft.

## Part 4: Flare cores, the failure and the fix

Rewritten 2026-10-14 around the follow-up (`results/followup_recap_20261013.md`): one slide on the failure, one on the suggestions that did not help, two on the cause, two on the fix, two on the result. Old-model figures from `results/plots/19_final_models_20261004/`, new-model figures from `results/plots/22_adopted_bp_20261014/`.

### Slide 14 · Divider
- **Part 4:** Flare cores: the failure and the fix
- **Subtitle:** Why BP missed them, and what fixed it

### Slide 15 · The failure
- **Title:** Flare cores, 1 pixel in 10,000, held 90% of the BP error
- **On slide:** old BP model (width 336, log1p input, μ = 1, 40 epochs): 0.01% of pixels, 90% of the DEM error (supervised 90% too); there it held 40% of the solver's emission, peaked 23% as high, 0.2 dex too cool, 83% of its curves had extra peaks (solver 18%); it fit 171, 193 and 211 Å to a few percent but reproduced only 23% of 94 Å: the hot plasma was missing. Bold: read as a training problem: oversample, train longer, go bigger.
- **Visual:** `19_final_models_20261004/fig4_example_curves_bp.png` (old model at random flare cores).
- **Notes:** [1 min, a recap: this is where the last meeting left off] The old BP model, width 336 with log1p input, mu = 1 and 40 epochs: flare cores, pixels at 32 times the Bright cutoff, are one in ten thousand and held 90% of its DEM error, supervised's too. There it held 40% of the solver's emission, peaked a quarter as high, 0.2 dex too cool, with extra peaks on 83% of curves. The clue: it fit the cool channels to a few percent but reproduced only 23% of 94 A, so the hot plasma was missing. We read it as a training problem, and that is where the suggestions came from: oversample, train longer, go bigger. If asked: solver noise is 0.4% of the Bright error and missed second peaks 1.2%, both ruled out; the figure is the old model at random flare cores, black solver, blue ours, orange supervised.
- **Status:** draft.

### Slide 16 · The suggestions on the original loss
- **Title:** On the original loss, none of the suggestions helped
- **On slide:** table (base √ + Fourier 40 epochs / oversample flare cores 8.5× / 120 epochs / 10M): DEM MSE 4.56 / 5.05 / 4.21 / 5.10; median 0.023 / 0.025 / 0.022 / 0.022; AIA MSE 83 / 87 / 108 / 125; flare-core emission 35% / 24% / 37% / 20%; flare-core 94 Å 15% / 9% / 16% / 8%. Bullets: away from flare cores all four score the same; at flare cores oversampling and 10M make it worse; bold: training harder does not help, the loss itself needs checking.
- **Notes:** [1 min] We ran all three on the original loss, on the square root plus Fourier input. Away from flare cores all four score the same. At flare cores oversampling and the 10M model make things worse, and 120 epochs moves emission from 35% to 37%. Training harder does not help, so the loss itself needed checking. If asked: oversampling draws pixels in proportion to brightness, 8.5 times more flare cores; the 10M model is 1,800 wide; the base DEM MSE of 4.56 is the square root plus Fourier model, the log1p final was 4.28; the solver reproduces 92% of 94 A at flare cores.
- **Status:** draft.

### Slide 17 · The cause
- **Title:** The cause: the network already sits at its loss's best answer
- **On slide:** per pixel, solve the loss directly (convex): the best answer no network can beat; 12,000 validation pixels, 6,000 flare cores; the network sits 1 to 4% above it (8% at flare cores). Table (flare cores, validation, original loss): old BP network loss 565 = 294 + 270, 94 Å 28%, emission 38%; best answer 463 = 220 + 243, 26%, 30%; solver's DEM 1,125 = 1,119 + 6, 94%, 100%; supervised model's DEM 1,878 = 1,107 + 771, 91%, 95%. Bullets: a perfect network lands on the best answer, which still drops the hot plasma; our loss rates the solver's DEM 2.4× worse than one without it; bold: the loss, not training, was the limit.
- **Notes:** [2 min] Forget the network. For one pixel, which 54 weights give the lowest possible loss? The loss is convex in the weights, so a solver finds that bottom, and no network can beat it on that pixel. We did this on 12,000 validation pixels, 6,000 of them flare cores, and scored four answers under the same loss. The network sits 1 to 4% above its best answer, 8% at flare cores: training had converged. And the best answer itself reproduces only 26% of 94 A. Look at the two terms: the solver's DEM keeps every channel in band, penalty 6, but pays 1,119 in sparsity for the hot plasma; the best answer drops it. Our loss rates the solver's DEM 2.4 times worse than a DEM without the hot plasma, so every training step moves away from the solver there. The loss, not training, was the limit. If asked: this also explains the 15% emission shortfall at ordinary pixels, where the best answer holds 82 to 86% of the solver's emission; the supervised DEM scores worst under our loss yet is closest to the solver.
- **Status:** draft.

### Slide 18 · Why the loss drops hot plasma
- **Title:** Why: hot plasma is expensive to include and cheap to leave out
- **On slide:** rule vs price; loss = sparsity + μ × band penalty, μ = 1 originally. Response table, two rows ("One unit of plasma at / What AIA sees"): 1 to 1.3 MK (warm): 171 Å 36 to 95 DN/s, 193 Å 29 to 55; 8 MK (hot): 94 Å 0.266 DN/s, no channel above 0.4 (the full six-channel response is in the notes). Worked example (94 Å = 164 ± 11.5): hot plasma in, 164 ÷ 0.266 ≈ 616 units, sparsity 616, penalty 0, total 616; out, 94 Å ≈ 0, 148 below the band, penalty (148 ÷ 16.1)² = 85, total 85. Right bullets: warm plasma has to be in the DEM and its 94 Å comes free; hot plasma needs hundreds of units paid by the sparsity term; the best answer leaves it out; bold: at μ = 100 the penalty is 8,500: the plasma goes back in. Muted: the best answer reproduces 91% of 94 Å at 1 to 1.8× the Bright level, 44% at 10 to 32×, 26% at flare cores (solver 100, 88, 94%).
- **Notes:** [2 min] Why does the loss drop it? The solver's band is a rule; ours is a price: sparsity plus mu times the band penalty, with mu = 1. The response table: a unit of warm plasma lights 171, 193 and 211 A by tens of DN/s, so it has to be in the DEM anyway and its 94 A comes free. A unit of hot plasma lights no channel by more than 0.4 DN/s, so a bright 94 A reading needs hundreds of units, all paid by the sparsity term. The worked example at a typical flare core: putting the hot plasma in costs 616, leaving it out costs 85, so the best answer leaves it out. At mu = 100 leaving it out costs 8,500 and the plasma goes back in. If asked: one unit is 10^26 cm^-5 of emission measure, the DEM's own unit; the plasma is never measured, AIA measures DN/s and R is a fixed table; the full response per unit, warm (logT 6.0 to 6.1): 94 A 0.2, 131 A 0.5 to 1.2, 171 A 36 to 95, 193 A 29 to 55, 211 A 5 to 7, 335 A 0.4; hot (logT 6.9): 0.266, 0.40, 0.05, 0.34, 0.08, 0.05; the band itself is sound, the noise follows photon counting; brighter pixels draw more of their 94 A from hot plasma, which is why the share dropped grows with brightness: the best answer reproduces 91% at 1 to 1.8x the Bright level, 44% at 10 to 32x, 26% at flare cores.
- **Status:** draft.

### Slide 19 · The fix
- **Title:** The fix: a stiffer band, chosen on validation
- **On slide:** screen table (best answer under the loss, validation): μ = 1: DEM error 36.9, Quiet MSE 0.076, flare-core 94 Å 26%, emission 30%; μ = 10: 10.8, 0.037, 81%, 90%; μ = 30: 5.10, 0.032, 89%, 97%; μ = 100: 2.32, 0.041, 93%, 100%; μ = 300: 1.80, 0.048, 94%, 101%; supervised model (a trained network): 6.10, 0.054, 91%, 95%. Bullets: if a network learned the loss perfectly, how close would it get; stiffer helps bright pixels and costs faint ones; bold: μ = 100, most of the gain, Quiet error still below the supervised model's. Muted line: μ is one scalar chosen on validation, like any loss weight; training never sees a DEM. The 94 Å column, which needs no solver DEM, plateaus at μ = 100.
- **Notes:** [1.5 min] The screen tests the loss, not the network: for each candidate mu we solved every validation pixel's best answer and scored it against the solver. It answers: if a network learned this loss perfectly, how close to the solver would it get? Stiffer helps bright pixels and costs faint ones, where the band is mostly noise, so Quiet error is lowest near 30 and rises beyond. mu = 100 takes most of the gain with Quiet error still below the supervised model's. On the choice: mu is one scalar chosen on validation, as any loss weight would be; training never sees a DEM; and the 94 A column, which needs no solver DEM, plateaus at 100. If asked: mu = 300 lowers DEM error from 2.32 to 1.80 but raises Quiet error to 0.048, and trained behind 100 at matched epochs; a second candidate, mu = 30 with the sparsity term divided by root brightness, screened well but trained worse (validation 22.4 against 20.1); validation holds more flare cores than the test set, so these values run higher than the test numbers.
- **Status:** draft.

### Slide 20 · Trained on the stiff band
- **Title:** On the stiff band, longer training is what helps
- **On slide:** table (BP runs, band ×100; DEM MSE / median / AIA MSE / Quiet MSE / flare-core EM / flare-core 94 Å / validation): original loss base 4.56 / 0.023 / 83 / 0.057 / 35% / 15% / 39.7; 40 epochs 2.37 / 0.062 / 44 / 0.035 / 89% / 68% / 20.1; 120 epochs 2.00 / 0.041 / 44 / 0.032 / 92% / 75% / 17.8; 240 from scratch 1.74 / 0.036 / 591 / 0.035 / 91% / 73% / 14.0; **120 + 120 continued 1.75 / 0.036 / 103 / 0.032 / 93% / 77% / 15.8**; + oversample 2.50 / 0.071 / 39 / 0.038 / 88% / 63% / 21.1; 10M 35.7 / 0.041 / 1.7M / 0.070 / 92% / 74%; 680 wide 897 / 0.034 / 21.7M / 12.5 / 115% / 78% / 4,170; supervised 0.91 / 0.112 / 157 / 0.034 / 92% / 85% / 6.10. Bullets: the band alone halves DEM MSE; the network then sits far above its best answer (118% at flare cores), so training is the limit; 240 from scratch misfits rare pixels; bigger models blow up. Bold: final BP model: band ×100, 120 epochs, then 120 more continued from that checkpoint.
- **Notes:** [1.5 min] Trained on the stiff band: the band alone halves DEM MSE and lifts flare-core emission from 35% to 89%. Now the network sits far above its best answer, 118% at flare cores after 40 epochs, so training is the limit and longer training helps. 120 epochs, then 120 more continued from that checkpoint, is the final model: DEM MSE 1.75, AIA MSE 103, below supervised's 157. 240 from scratch gets the same DEM gains but breaks the band on a few rare pixels, AIA MSE 591. Oversampling still hurts and bigger models blow up. If asked: the gap after 120 epochs is 47%; 240 from scratch has a band penalty at the validation flare cores 130 times the 120-epoch model's, the continuation 13 times; log1p is worse here, 3.23; no gradient clip fits AIA better but worsens every DEM measure; an epoch is about 145 s on one GPU, so 240 epochs is about 10 hours.
- **Status:** draft.

### Slide 21 · Flare cores now
- **Title:** Flare cores now: 93% of the solver's emission, 77% of its 94 Å
- **On slide:** table (old BP / new BP / supervised / solver): emission 40% / 93% / 92% / 100%; 94 Å reproduced 23% / 77% / 85% / 92%; 131 Å reproduced 76% / 101% / 90% / 101%; peak height 23% / 66% / 91% / 100%; temperature shift −0.18 / −0.01 / 0.00 / 0 dex; curves with extra peaks 83% / 25% / 13% / 18%. Bullets: spurious extra peaks were 42% of the old error, 3.5% now; it tracks the solver's peak height to ~100, above 3,000 it reaches 40%.
- **Visual:** `22_adopted_bp_20261014/fig4_example_curves_bp.png` (same draw as slide 23).
- **Notes:** [1 min] Flare cores now: 93% of the solver's emission, supervised 92%; 94 A at 77% against the solver's 92% and supervised's 85%; peak height two thirds of the solver's; no temperature shift; extra peaks on 25% of curves, the solver has them on 18%. Spurious peaks were 42% of the old error and are 3.5% now. The figure is the same random draw as the failure slide. If asked: W1 at flare cores went from 0.25 to 0.03 dex; it tracks the solver's peak height to about 100 and reaches 40% above 3,000; 131 A is at 101%.
- **Status:** draft.

### Slide 22 · What remains
- **Title:** What remains: flare cores still set the BP DEM MSE
- **On slide:** error vs supervised by brightness 0.55 faintest, 1.0 at 0.3–1×, 0.9 to 1.3 at 1–10×, 1.9 at 10–32×, 2.0 at flare cores (old model 0.64, 1.8, 2 to 5, 18, 4.7); flare cores 94% of our BP error and the DEM MSE gap (1.75 vs 0.91); 94 Å at 77% vs the solver's 92%, every other channel at 100%; longer training keeps closing it, 240 from scratch starts to misfit a few pixels' AIA, a third 120-epoch continuation is untested. Bold: the BP loss was the limit; training is now the limit, and it still has room.
- **Visual:** `22_adopted_bp_20261014/fig5_aia_fit_flare_cores_bp.png`.
- **Notes:** [1 min] Against supervised by brightness we are now at 0.55 on the faintest pixels, about 1 through the ordinary bright range, and 2 at flare cores; the old model was at 18 there. Flare cores still hold 94% of our BP error and set the DEM MSE gap. The remaining shortfall is 94 A at 77% against 92%; everything else is at 100%. Longer training keeps closing it: 240 from scratch cut the flare-core error another 13% but started to misfit a few pixels' AIA; a third continuation is untested. The loss was the limit; training is now the limit, and it still has room. If asked: why the median pixel error rose from 0.026 to 0.036 while Quiet MSE fell from 0.058 to 0.032 is open.
- **Status:** draft.

## Part 5: Next steps

### Slide 23 · Divider
- **Part 5:** Next steps
- **Subtitle:** What we learned, and what to decide

### Slide 24 · Lessons
- **Title:** What we learned
- **Visual:** six cards: Train on the objective / Check the loss first (the BP network sat at its loss's optimum, which dropped the hot plasma; no training change could fix that, a stiffer band did) / Simple scales / Size has a ceiling (fresh pixels pay off up to width 336; 1.43M and 10M blow up on rare pixels, both tracks) / Heavy tails / Input encoding (√ + Fourier for both tracks; lifts ElasticNet past supervised and beats log1p on the stiff BP band).
- **Notes:** [1 min] Six lessons. The two new ones are 2 and 4: check the loss before changing the training, and size has a ceiling. The others carry over: train on the objective, the simplest model scales, report medians and percentiles, and the input encoding matters.
- **Status:** draft.

### Slide 25 · Next
- **Title:** Next
- **Visual:** four cards: Keep training BP (a third 120-epoch continuation untested; watch the AIA fit on rare pixels) / Validate on every pixel / Direct DEM output (deferred) / Uncertainty, AIA + XRT.
- **Notes:** [1 min] Next: keep training BP, watching the AIA fit on rare pixels; validate on every pixel, because the sampled validation set missed the blow-ups; direct 18-bin output, deferred so far; uncertainty from the noisy re-solves, and XRT for the hot plasma.
- **Status:** draft.

### Slide 26 · For discussion
- **Title:** For discussion
- **On slide:**
  - Which metric should lead the paper? DEM MSE is set by 0.01% of pixels; up to the 99th percentile we are level with or ahead of the supervised model on BP.
  - Train the BP model longer (a third continuation), or freeze here at 1.9× the supervised DEM MSE?
  - Is AIA + XRT in scope for this paper?
- **Visual:** none (numbered questions).
- **Notes:** [the rest] Three questions. One: which metric leads the paper; DEM MSE is set by 0.01% of pixels, and up to the 99th percentile we are level or ahead on BP. Two: train longer or freeze at 1.9 times the supervised DEM MSE. Three: is AIA plus XRT in scope. If the band weight comes up: one scalar, chosen on validation like any loss weight; training never sees a DEM; the 94 A column of the screen, which needs no solver DEM, picks the same value.
- **Status:** draft.

## Appendix: backup slides

Pulled up only when asked. Slides 28 to 31 are on 4 images, each a crop near disk
centre (128² for the CNNs, 256² for the first MLP): taken during the X2.1 flare of
2011-09-06 (AR 11283), quiet Sun on 2012-06-03, moderate activity on 2013-11-13, and
during the X1.6 flare of 2014-09-10 (AR 12158). The crops start at AIA pixel
(1800, 1800), so they need not contain the flare cores.

### Slide 27 · Divider
- **Appendix:** Backup slides
- **Subtitle:** The history, and answers to likely questions

### Slide 28 · Which loss
- **Title:** Six channels underdetermine the DEM: the regulariser picks it
- **On slide:**
  - Before any network: five losses optimised directly, pixel by pixel, on 4 timestamps, each compared with BP.
  - Fit-only losses (χ² + smoothness, max entropy, Tikhonov) reconstruct AIA ~12x better than BP, yet land ~5x further from its DEM.
  - BP's own terms (noise band + L1) land on BP's DEM: MAE 0.03 vs 0.18–0.20.
  - Takeaway (bold): the network has to train on the solver's objective, not on fit.
- **Visual:** two example pixels side by side (`slides/img/loss_pixels.png`, cropped from `results/plots/01_multiloss_20260609/loss_comparison.png`): BP in black, the barrier losses on top of it, the fit-only losses adding a hot component BP does not need.
- **Notes:** Before any network, five losses optimised directly per pixel on 4 timestamps. Averages: AIA MAE against the observation, BP 5.2, band penalty 5.0, band penalty plus fit 2.6, fit-only 0.40 to 0.44. DEM MAE against BP: band penalty 0.034, band penalty plus fit 0.043, fit-only 0.18 to 0.20. W1 on the brightest 5% of pixels: 0.030 and 0.033 against 0.15 to 0.18. Six equations, eighteen unknowns: many DEMs fit within noise, so whatever breaks the tie decides the shape. L-BFGS and Adam reach the same curves; SGD often fails to converge.

### Slide 29 · First network, then the patch CNN
- **Title:** A per-pixel MLP missed BP's shape; a patch CNN tracked it
- **On slide:**
  - Top line (muted): 4 images: X2.1 flare (2011), quiet Sun (2012), moderate activity (2013), X1.6 flare (2014); small crops near disk centre.
  - First try: the 6 intensities of one pixel → MLP (213k params) → 54 weights, trained on the BP loss over ~221k pixels. Smooth, single-peaked curves, but further from BP: DEM error 0.078 vs the CNN's 0.047.
  - Patch CNN: a 9×9 neighbourhood → CNN (~1.5M params) → 54 weights. Tracked BP's shape, with sparsity close to BP's on held-out pixels.
  - Trained on one image, then on all 4 jointly; zeroing the neighbourhood for 10% of batches kept it usable on a single pixel.
- **Visual:** `slides/img/first_mlp_vs_cnn.png` (from `results/plots/15_first_mlp_repro_20261002/`): BP, first MLP and patch CNN on three held-out pixels of the X2.1 flare image.
- **Notes:** The June write-up called the first MLP's curves noisy and oscillating, but no figure of them survives and its checkpoint was not kept. Retrained with the unchanged June code, its curves are smooth: none has three or more peaks. The real difference is accuracy, on 400 held-out pixels per image: mean DEM error against BP 0.078 against the CNN's 0.047, W1 0.055 against 0.048 dex. It had even seen these pixels in training. It is the same architecture as today's production model, at width 256 with raw intensities.
- **Status:** draft.

### Slide 30 · Ablation
- **Title:** Ablation: the gap was capacity, not missing context
- **On slide:**
  - Four variants at ~1.45M params, same BP loss, same 4 images.
  - Size, not neighbours, closed the gap: DEM error vs BP 0.042 for the 1.43M centre-pixel MLP, 0.047 for the patch CNN, 0.078 for the first MLP.
  - The patch CNN was sparsest, but measured on held-out pixels of the training images.
- **Visual:** dot plot of mean sparsity (effective number of active basis weights, lower = sparser) with BP's 1.79 as a dashed line: patch CNN 1.70, shuffled-patch CNN 1.88, centre-pixel MLP 1.89, flat-patch MLP 1.95.
- **Notes:** The shuffled-patch CNN applies a fixed random permutation to the 81 patch pixels. It was less sparse than the CNN on all 4 images, 1.88 against 1.70 on average, level with the centre-pixel MLP, so the spatial arrangement of the neighbours mattered. The flat-patch MLP was erratic across images: 1.61 on the X2.1 flare, 2.27 on quiet Sun. The centre-pixel MLP was closest to BP on 3 of 4 images; the CNN on the X1.6 flare. So the CNN's lower sparsity did not mean DEMs closer to BP. Capacity and input scaling were not separated: the 213k MLP took raw intensities, and at full scale MLPs on log1p inputs does well.
- **Status:** draft.

### Slide 31 · Leave one day out
- **Title:** On unseen days, the patch CNN's edge disappeared
- **On slide:**
  - Leave one image out: train on 3, test on the 4th; patch CNN and centre-pixel MLP, 4 folds.
  - Neither is consistently closer to BP's sparsity on the unseen image; the MLP fits AIA better on 3 of 4.
  - Little overfitting (held-out within ±0.2 of in-sample on 3 folds), except the X1.6 flare: peaks too cool (logT ~6.1 vs BP ~6.4), with one other flare image in training.
  - Takeaway (bold): carry the simpler centre-pixel MLP forward, and get more flare data → scaling.
- **Visual:** `slides/img/loo_flare.png`: the held-out X1.6 flare image, both models (dashed) against BP.
- **Notes:** Closer to BP's sparsity: the CNN on 2 images, the MLP on 1, one tie; the MLP is closer on average, 0.19 against 0.25, because of the X1.6 flare image. Held-out sparsity by fold, MLP against CNN, with BP in brackets: X2.1 flare 1.59 vs 1.64 (1.97); quiet Sun 1.78 vs 1.61 (1.67); moderate 1.83 vs 1.83 (1.82); X1.6 flare 1.96 vs 2.31 (1.71). The ablation's in-sample edge, 1.70 vs 1.89, came from scoring pixels of images the model trained on. Four folds is a small sample; scaling is the real test.
- **Status:** draft.

### Slide 32 · The collapse
- **Title:** The first full-disk runs collapsed to an all-zero DEM
- **On slide:**
  - Full-disk intensities span ~7 decades (10⁻³ off-limb to 10⁴ in flare cores); the small crops spanned about one.
  - Fed raw, the untrained network predicted AIA ~7× outside the noise band: loss 5×10¹³ at step 0. The output softplus underflowed, its gradient became exactly zero, and every pixel predicted zero.
  - Fix: a bounded input (log1p then, √ + Fourier now; loss and metrics stay in physical units), learning-rate warmup, a clamped softplus whose gradient cannot vanish, and a guard that stops a run whose loss freezes.
  - Bold: a 64-block smoke test could not catch it: it never drew a flare pixel.
- **Visual:** a small strip: raw input range 10⁻³ … 10⁴ → log1p → 0 … 9.2.
- **Notes:** Three submissions died the same way before the fix; the cause was visible at step 0, before any weight update. Gradient clipping does not help, because Adam normalises the step size.
- **Status:** draft.

### Slide 33 · How we evaluate
- **Title:** How we score at scale
- **On slide:**
  - 153 test days, never seen in training; each scored on random quarters of the image against 5 solver targets (the clean solve and 4 noisy re-solves).
  - Bright = any channel above its top-5% cutoff (~10% of pixels); the rest is Quiet.
  - DEM against the solver: MSE, total-emission error, W1 (temperature-shape distance). AIA reconstruction against the observation: MAE, MSE.
  - We also report medians and percentiles: the error is extremely heavy-tailed. For our final BP model, the worst 1% of pixels hold 98% of the DEM error, and the flare cores (0.01% of pixels, 1 in 10,000) alone hold 94%.
- **Visual:** diagram of one test day: the 2048² DEM image as a 16×16 grid of 128² blocks, 64 of them highlighted (one target's random quarter), with "× 5 targets: the clean solve + 4 noisy re-solves, each with its own 64 blocks".
- **Notes:** The Bright cutoffs and the test set come from the supervised side's protocol, so both models are scored on identical pixels. W1 treats each DEM as a distribution over logT.
- **Status:** draft.

### Slide 34 · ElasticNet

- **Title:** ElasticNet: the final model, and its flare cores
- **On slide:** table (ENet flare cores, shared test; label-free / supervised / solver): share of ENet DEM error 44% / 34% / none; emission vs solver 99% / 87% / 100%; peak height 101% / 92% / 100%; 94 Å reproduced 53% / 48% / 55%; 171 Å reproduced 97% / 70% / 100%; curves with extra peaks 18% / 18% / 21%. Bullets: final is width 336, √ + Fourier input (chosen on validation), 40 epochs, the solver's α = 0.001, λ = 0.5, C = 6, ahead of supervised on all six measures (Part 3); no band to price, the fit term is quadratic, and at flare cores it reproduces the channels as the solver itself does (94 Å 53% vs 55%); from 722k up a few bright pixels blow up by orders of magnitude, 336 is the largest stable width.
- **Visual:** `22_adopted_bp_20261014/fig4_example_curves_enet.png` (ElasticNet model at random flare cores).
- **Notes:** Backup. ElasticNet final: width 336 (409k parameters with the square root plus Fourier input), fresh pixels, 40 epochs, the solver's alpha 0.001, l1 ratio 0.5, C = 6. The input encoding was chosen on validation (0.0533 against log1p's 0.0575) and lifted it past the supervised model on every measure: DEM MAE 0.151 vs 0.188, DEM MSE 0.310 vs 0.318, EM error 12.2% vs 18.0%, W1 0.103 vs 0.121, AIA MAE 0.56 vs 6.32, AIA MSE 161 vs 598; median pixel error 0.85 vs 1.12, p99 30 vs 35, p99.9 102 vs 178. Flare cores (32x the Bright cutoff and above) are 0.01% of pixels and hold 44% of the ElasticNet error (supervised 34%), but they are not a failure of the kind BP had: the ElasticNet fit term is quadratic, there is no band to price, and the model reproduces the channels as the ElasticNet solver itself does (94 A 53% against the solver's 55%, 131 A 80% against 78%, 171 A 97% against 100%). From 722k parameters up, a few bright pixels blow up by orders of magnitude, so 336 is the largest stable width. Sources: results/plots/19_final_models_20261004 (tag final), 20_sqrtff_final_20261004.
- **Status:** draft.

### Slide 35 · Multi-peaked DEMs

- **Title:** Multi-peaked DEMs: the new BP model finds 60% of them
- **On slide:** table (BP track, shared test; flagged multi-peaked / precision / recall): solver (BP) 13.5% of pixels; old BP model (original loss) 9.9% / 36% / 26%; new BP model (stiff band) 15.5% / 53% / 60% (highlighted); supervised 41.1% / 24% / 72%. Bullets: multi-peaked = two or more interior peaks with prominence ≥ 0.15 of the maximum, on the full shared test set; we flag fewer pixels than the supervised model and are right more often, and the stiff band lifted recall from 26% to 60%; where BP is multi-peaked our gap from BP is the size of BP's own re-solve scatter (0.97×; 30 re-solves under photon noise at 80 pixels; 0.70× at single-peaked pixels); ElasticNet: the solver is multi-peaked on 62% of pixels, we find 82% of them at 87% precision (supervised 84% at 80%); at BP flare cores, spurious extra peaks were 42% of the old error and are 3.5% now.
- **Notes:** Backup, for the bimodality question. Multi-peaked means two or more interior peaks with prominence at least 0.15 of the curve maximum, counted on the full shared test set. BP is multi-peaked on 13.5% of pixels. The old BP final flagged 9.9% at 36% precision and 26% recall; the stiff-band model flags 15.5% at 53% precision and 60% recall; the supervised model flags 41.1% at 24% precision and 72% recall. Self-consistency on the width-336 model: 80 multi-peaked bright pixels, 30 BP re-solves each under photon noise; our deviation from BP is 0.97 times BP's own re-solve scatter (0.70 at single-peaked pixels), so where BP is multi-peaked our gap is the size of BP's own noise. ElasticNet is multi-peaked on 62% of pixels; we find 82% of them at 87% precision, supervised 84% at 80%. At BP flare cores, spurious extra peaks were 42% of the old error and are 3.5% now. Sources: results/plots/21_followup_20261006/bp_bright_failure_fu_bp_mu100_ep120_cont.json, 19_final_models_20261004/bp_self_consistency_h336.json, 13_bright_diagnostic_20260928.
- **Status:** draft.
