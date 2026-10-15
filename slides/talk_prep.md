# Talk prep: advisor meeting, Wednesday 21 October 2026

Deck: `slides/DEM_deck_v2.pptx`, 35 slides. The main line is slides 1 to 26 (20 content
slides, 5 dividers, the title); the appendix starts at slide 27 and is pulled up only when
asked. Every main-line slide's speaker notes open with its time budget and say what to say;
uploading the pptx to Google Slides keeps the notes. The collaborators are "the supervised
side" and "the advisor" throughout.

## The 30-second version

We train a per-pixel network on the DEM solver's own objective instead of on its outputs,
so it never sees a solver DEM. A 400k-parameter MLP on one pixel's six intensities then
matches or beats the supervised model: ahead on every ElasticNet measure and on four of six
BP measures, three times closer on the typical BP pixel. The one place it lost, flare cores,
turned out to be the loss pricing hot plasma out, not training. A stiffer noise band put the
plasma back and more than halved the BP error. What remains is a training gap that is still
closing.

## Timing

| Act | Slides | Minutes | Running total |
|---|---|---|---|
| In one slide | 2 | 2 | 2 |
| Part 1: problem and approach | 4, 5, 6, 7 | 5.5 | 7.5 |
| Part 2: getting to the model | 9, 10 | 3 | 10.5 |
| Part 3: results | 12, 13 | 3.5 | 14 |
| Part 4: flare cores | 15 to 22 | 11 | 25 |
| Part 5: lessons, next | 24, 25 | 2 | 27 |
| For discussion | 26 | the rest | 30 |

If time runs short, skip in this order: slide 16 (the suggestions; say one sentence: "none of
them helped on the original loss"), slide 22 (what remains; fold into slide 21), slide 10
(choosing the model; say "width 336 by validation, bigger blows up").

## The arc in ten sentences

1. Same physics, different training signal: we minimise the solver's objective, and solver
   DEMs only score.
2. The network predicts the solver's 54 basis weights, so it searches the same space of DEMs;
   the basis and the response matrix are fixed.
3. The loss is the solver's objective; the one difference is that the noise band is a price
   (the band weight) rather than a rule.
4. Two years of AIA split in time, scored on the supervised side's protocol; the error is
   heavy-tailed, so medians and percentiles.
5. The patch CNN's edge was in-sample; at scale the per-pixel MLP wins; width 336 by
   validation; bigger blows up on rare pixels.
6. ElasticNet wins all six measures, BP four of six; the typical BP pixel is three times
   closer; the gap is the last 0.1% of pixels.
7. Flare cores held 90% of the BP error, and training-side fixes did not help.
8. The network already sat at its loss's optimum, and that optimum drops hot plasma: 616 in
   sparsity to include it against 85 in penalty to leave it out.
9. A stiffer band, chosen on validation, plus longer training: flare-core emission 40% to
   93%, DEM MSE 4.3 to 1.75.
10. Training is now the limit and still has room; three questions for the advisor.

## Likely questions, with answers

**The band weight is chosen on validation with solver DEMs. Is that not supervised?**
It is one scalar, chosen like any loss weight or early-stopping epoch; training never sees a
DEM. And the choice does not need solver DEMs: the 94 Å column of the screen (slide 19) is
the fit to observed AIA at flare cores, which is label-free, and it plateaus at a band weight
of 100 (26, 81, 89, 93, 94% for 1, 10, 30, 100, 300). What solver DEMs added is the cost on
the faint side (Quiet error rising beyond 30), and that could be replaced by a label-free
Quiet AIA fit. Source: `results/followup_recap_20261013.md`, Part 7.

**Why not a hard constraint, or the band weight to infinity?**
Stiffer helps bright pixels and costs faint ones: at faint pixels the band is mostly noise
and a stiff band makes the best answer chase it. 300 already raises Quiet error (0.048 vs
0.041) and trained behind 100 at matched epochs. A hard constraint is the solver's rule; in a
network it would be a projection or a Lagrangian multiplier per pixel, which is a next step,
not something tested.

**Why does longer training help now when it did not before?**
On the original loss the network sat 1 to 4% above its per-pixel optimum (8% at flare
cores): it had converged, and the optimum was wrong. On the stiff loss the optimum moved and
the network is far from it: 118% above it at flare cores after 40 epochs, 47% after 120. So
epochs now buy something. Source: slides 17 and 20.

**Why do bigger models blow up, and why exactly 336?**
A few flare-core pixels: the 1.43M stiff-band model over-predicts 171 Å three-fold there,
and its worst 1% of pixels carry 99.996% of its error; the 10M models fail the same way on
both tracks, and ElasticNet from 722k up. The mechanism is not understood; the sampled
validation set never saw these pixels, which is why "validate on every pixel" is on the next
list. 336 is the largest width that is stable on both tracks and the BP validation best.

**Is the remaining 94 Å gap (77% against the solver's 92%) loss or training?**
Training. The best answer under the stiff loss reproduces 93% of 94 Å at flare cores; the
network reaches 77% and it rises with epochs (68, 75, 77% at 40, 120, 120 + 120). 240 from
scratch cut the flare-core error another 13% but started to misfit a few pixels' AIA, so a
third continuation is the thing to try, with the AIA fit on rare pixels watched.

**Why does the supervised model get flare cores right?**
It copies the solver, and the solver treats the band as a rule. Under our loss the supervised
DEM scores worst of the four answers (1,878 against the network's 565) yet is closest to the
solver: the objective and the evaluation disagreed, which is the whole finding.

**What about ElasticNet's flare cores?**
Not the same failure: the ElasticNet fit term is quadratic, there is no band to price. At
flare cores the model holds 99% of the solver's emission and reproduces the channels as the
ElasticNet solver itself does (94 Å 53% against the solver's 55%). Flare cores still hold
44% of the ElasticNet error, and it beats the supervised model on all six measures. Backup:
slide 34.

**The bimodal DEMs from July?**
BP is multi-peaked on 13.5% of pixels. The new BP model flags 15.5% at 53% precision and
60% recall (old model 26% recall; supervised 72% recall at 24% precision). Where BP is
multi-peaked our gap from BP is the size of BP's own re-solve scatter under photon noise
(0.97×), so the deficit sits inside the solver's noise. Backup: slide 35.

**Why predict basis weights and not the 18 bins?**
The solver uses the same basis and fits flare cores, so the basis is not the limit, and
predicting the same weights keeps BP's sparsity meaningful for the network. Direct output is
on the next list, deferred.

**How fast is it against the solver?**
No measured number yet. The network is one forward pass per pixel (409k parameters); the BP
solver took about 9.5 minutes per 2048² image on a Torch CPU node. If the advisor wants the
number, it is one short job.

**What did training cost?**
About 145 s per epoch on one GPU: 40 epochs is about 1.6 hours, 120 + 120 about 10 hours.
Label generation for the supervised side's data was about 4 hours on about 90 nodes per
solver, 1.7 TB.

**Why did the median pixel error get worse (0.026 to 0.036) while Quiet MSE improved
(0.058 to 0.032)?**
Open. The stiff band moves the best answer at faint pixels, where the band is mostly noise;
the median is a faint-pixel statistic. The honest answer is that it was noticed and not yet
explained.

**Why does 240 from scratch misfit AIA when 120 + 120 does not?**
Observation only: 240 from scratch gets the same DEM gains but its band penalty at the
validation flare cores is 130 times the 120-epoch model's (the continuation: 13 times), and
AIA MSE is 591 against 103. The continuation warm-starts from the 120-epoch checkpoint with a
fresh cosine schedule. No mechanism established.

**Does the chronological split matter?**
Train Jan 2014 to Jun 2015, validation to September, test to December 2015, so every test
image is later than anything trained on. All of it is one phase of the cycle; a different
phase is untested.

**What is the label-free advantage if supervised works?**
No solver labels to generate (1.7 TB, hours on about 90 nodes per solver, for 1,223 images),
and any change of objective is a retrain, not a relabel. On the numbers it is ahead of the
supervised model on most measures and three times closer on the typical BP pixel.

**Is the 15% emission shortfall at ordinary pixels fixed?**
Yes: the best answer on the original loss held 82 to 86% of the solver's emission at
ordinary pixels; on the stiff band the model's total-emission error is 11.0% (old 19.6%) and
at quiet single-peaked pixels it holds 99% of the solver's emission.

**Architecture choices: SiLU, four layers, the Fourier input?**
The Fourier input is the supervised model's own input path, adopted by validation; the width
and depth come from the sweep and ablation; SiLU from the first design. None of them is a
claim of the paper.

**Next steps, in order?**
1. A third 120-epoch continuation, watching the AIA fit on rare pixels.
2. Validate on every pixel, not a sample.
3. Direct 18-bin output.
4. Uncertainty from the noisy re-solves; AIA + XRT if in scope.

## Numbers to have at hand

| | BP label-free | BP supervised | ENet label-free | ENet supervised |
|---|---|---|---|---|
| DEM MAE | 0.074 | 0.100 | 0.151 | 0.188 |
| DEM MSE | 1.75 | 0.91 | 0.31 | 0.32 |
| EM error | 11.0% | 14.3% | 12.2% | 18.0% |
| W1 (dex) | 0.068 | 0.133 | 0.103 | 0.121 |
| AIA MAE | 3.48 | 2.95 | 0.56 | 6.32 |
| AIA MSE | 103 | 157 | 161 | 598 |
| Median pixel error | 0.036 | 0.112 | 0.85 | 1.12 |
| p99 | 25 | 27 | 30 | 35 |

Flare cores (32× the Bright cutoff and above), BP track: 0.01% of pixels; old model held 90%
of the error, new 94% (supervised 90%); emission 40% → 93% (supervised 92%); 94 Å 23% →
77% (solver 92%, supervised 85%); extra peaks 83% → 25% (solver 18%).

The cause in numbers (validation flare cores, original loss): network 565 = 294 + 270, best
answer 463 = 220 + 243, solver's DEM 1,125 = 1,119 + 6, supervised DEM 1,878 = 1,107 + 771.
Worked example: hot plasma in costs 616, out costs 85; at a band weight of 100, 8,500.

## Honest unknowns

- Why the median rose while Quiet MSE fell.
- Why 240 from scratch misfits AIA on rare pixels and the continuation does not.
- The blow-up mechanism of wide models.
- Inference speed: not measured.
- A third continuation and a second seed: not run.

## Before the meeting

- Upload `slides/DEM_deck_v2.pptx` to Google Slides; check that the tables on slides 12, 20
  and 21 and the chart on slide 10 render.
- Copy the staged website preview to Triborough if the viewer will be shown.
- Rehearse once against the notes: 27 minutes without questions.
