# Advisor deck: slide outline

Master outline: one block per slide. This file and the deck
(`slides/DEM_deck_v2.pptx`, built by `slides/build_deck_v2.py`) are kept in
sync: change this file first, then the deck is rebuilt from it.

Each slide: **Title** (a takeaway sentence), **On slide** (the actual text, kept
short), **Visual** (what the image or diagram shows, if any), **Notes** (what to
say; goes into speaker notes, not onto the slide).

---

## Section 1: Problem and approach (settled)

### Slide 1 · Title

- **Title:** Predicting DEM label-free

### Slide 2 · Why label-free

- **Title:** Same physics, different training signal
- **On slide:**
  - Solvers (BP, ElasticNet) run one optimisation per pixel: accurate, but slow at full resolution.
  - Supervised: a network learns to reproduce the solver's DEMs, so it needs them as labels.
  - Label-free (ours): a network learns to minimise the solver's objective; no labels.
  - We never train on the solver's DEMs; we use them only to evaluate.
- **Visual:** the two training loops as two rows of boxes, styled like the model diagram. Supervised: observed AIA, network, DEM, then a "loss" link to the solver's DEM. Label-free: observed AIA, network, DEM, predicted AIA, then a "loss" link to the observed AIA.
- **Status:** settled.
- **Notes:** The audience knows the inverse problem. This slide only says what is different about our training signal.

### Slide 3 · The model

- **Title:** The model: a small MLP feeding a fixed forward model
- **On slide:**
  - 6 log-intensities in; 4 hidden layers of 232, SiLU; 54 non-negative basis weights out.
  - DEM = B·w over 18 bins, logT 5.5 to 7.2; predicted AIA = R·DEM. B and R are fixed.
  - 176k parameters; one forward pass per pixel.
- **Visual:** pipeline diagram: observed AIA, MLP, basis weights, DEM, predicted AIA, with the loss closing the loop.
- **Notes:** Only the MLP is learned. The basis and the response are the same ones the solver uses, so the network searches the same space of DEMs the solver does.
- **Status:** settled.

### Slide 4 · The loss

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

### Slide 5 · Three routes, one test

- **Title:** Three routes to a DEM, compared on the same test
- **On slide (points, no table):**
  - Solver (BP, ElasticNet): optimises its objective for each pixel. This is the reference.
  - Supervised network: learns to predict the solver's DEMs, so it needs them as labels.
  - Label-free network (ours): learns to minimise the solver's objective, with no labels.
  - Same test for all: 153 days never seen in training, identical pixels. DEM metrics against the solver; AIA reconstruction against the observation.
- **Visual:** none.
- **Notes:** Data: 1,223 Hofmeister-deconvolved timestamps, split by day into 917 train, 153 validation, 153 test. Cost: the solver runs one optimisation per pixel; both networks need one forward pass.
- **Status:** settled.

---

## Section 2: Finding a trainable objective

*(next)*

## Section 3: Architecture on small data

*(next)*

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
