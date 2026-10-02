# Advisor deck: slide outline

Working outline: one block per slide. Edit anything here; once a section is
settled it gets built into the NYU template. Styling comes last.

Each slide: **Title** (a takeaway sentence), **On slide** (the actual text, kept
short), **Visual** (what the image or diagram shows, if any), **Notes** (what to
say; goes into speaker notes, not onto the slide).

---

## Section 1: Problem and approach

### Slide 1 · Title

- **Title:** Predicting DEM label-free
- **On slide:** A neural DEM inversion trained on the solver's objective, not its labels · Progress update, October 2026
- **Visual:** NYU title layout, as in the template
- **Notes:** What we built since June, what each experiment taught us, how it compares with the supervised model, and the one failure that remains.

### Slide 2 · Why label-free

- **Title:** Goal: solver-quality DEMs, fast, without the solver's labels
- **On slide:**
  - BP and ElasticNet solve one optimisation per pixel: slow at full resolution, twice a day.
  - A supervised network is fast, but learns the solver's DEMs as labels.
  - We train on the solver's objective instead; its DEMs are used only to evaluate.
- **Visual:** two training loops side by side. Supervised: AIA, network, DEM, compared with the solver's DEM. Label-free: AIA, network, DEM, response R, compared with the observed AIA.
- **Notes:** The audience knows the inverse problem; this slide only says what is different about our training signal.

### Slide 3 · Model and loss

- **Title:** One small network, trained on the solver's own objective
- **On slide:**
  - Model: 6 log-intensities in; 4 hidden layers of 232, SiLU; 54 non-negative basis weights out; DEM = B·w over 18 bins, logT 5.5 to 7.2. 176k parameters, one pass per pixel.
  - BP loss: Σ_c [ max(0, |ŷ_c − o_c| − tσ_c) / tσ_c ]² + Σ_k w_k, with ŷ = R·B·w and t = 1.4: stay inside the noise band, then be sparse.
  - ElasticNet loss: the solver's objective, α = 0.001, l1 ratio 0.5.
  - Data: 1,223 Hofmeister-deconvolved timestamps, split by day into 917 train, 153 validation, 153 test.
- **Visual:** pipeline diagram: observed AIA, MLP, basis weights, DEM, response R, predicted AIA, with the loss closing the loop (the diagram from the current deck, slide 3).
- **Notes:** All results in this talk are on the 153 test days, never seen in training. The loss has no fit term inside the band: any DEM within noise is equally good, and the L1 term then picks the sparsest, which is exactly BP's criterion.

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
