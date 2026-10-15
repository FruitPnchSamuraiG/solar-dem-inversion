# Deck guide: the 30-minute talk, section by section

Read this with the deck open (`slides/DEM_deck_v2.pptx`); the numbers are the deck's slide
numbers. Each part gets a few lines on what it is for, then one entry per slide: what is
on it, and the one point to make. Time budgets in brackets add up to about 27 minutes.
The speaker notes inside the file say the same at more length; they can be ignored.

## Before Part 1 (slides 1 and 2)

**Slide 1 · Title.** Predicting DEM label-free.

**Slide 2 · In one slide** [2 min]. Four cards. No labels needed: the network trains on the
solver's own objective, solver DEMs only score it. Small is enough: a 400k-parameter MLP on
one pixel's six intensities, and bigger models fail on rare pixels. Ahead of supervised on
most measures: all six on ElasticNet, four of six on BP, the typical BP pixel three times
closer. Flare cores found and fixed in the loss: one pixel in ten thousand held 90% of the
BP error; the loss priced hot plasma out; a stiffer band put it back.
Point: these four claims are the talk; everything after is the evidence.

## Part 1 · Problem and approach (slides 3 to 7) [5.5 min]

What it is for: the setup. The advisor knows the inverse problem, so this part only says
what is different about our training signal, what the network predicts, which loss it
minimises, and what data and scoring everything below rests on.

**Slide 4 · Same physics, different training signal** [1 min]. Two training loops as
diagrams. Supervised: observed AIA, model, DEM, and the loss compares that DEM with the
solver's DEM. Label-free: observed AIA, model, DEM, predicted AIA through the response
matrix, and the loss compares predicted with observed AIA.
Point: we never train on a solver DEM; we use them only to evaluate.

**Slide 5 · The model** [1.5 min]. Pipeline: 6 intensities, MLP (409k parameters), 54 basis
weights, DEM over 18 bins, predicted AIA, with the loss closing the loop. The input is
encoded as the supervised model does it, square root then 12 Fourier frequencies. The 54
weights are the same weights both solvers solve for (a spike and two Gaussians at each bin);
the basis B and the response R are fixed, only the MLP learns.
Point: the network searches the same space of DEMs the solver does, so BP's sparsity means
the same thing for both.

**Slide 6 · The loss** [1.5 min]. BP track: the sum of the 54 weights plus μ times the band
penalty, which is zero inside the noise band (observation plus or minus 1.4σ) and quadratic
outside; the chart on the right shows that shape. ElasticNet track: the solver's own
objective with its settings. No fit term inside the band, so sparsity breaks the tie, as in
BP.
Point: the loss is the solver's objective; the one difference is that BP treats the band as
a rule and we treat it as a price, set by μ (1 originally, 100 in the final model, Part 4).

**Slide 7 · Data and scoring** [1.5 min]. Timeline of the split to scale: train to June
2015 (917 images), validation to September (153), test to December (153), so every test
image is later than training. 1,223 full-disk images, deconvolved, with a noise σ per pixel
and channel. Training sees AIA and σ only, fresh pixels every epoch. Test on the supervised
side's protocol: the same pixels and Bright cutoffs, five solver targets per image. Scores:
DEM against the solver, AIA reconstruction against the observation, and medians and
percentiles because the worst 1% of pixels hold 98% of the BP error.
Point: scored exactly as the supervised model is, and the error is heavy-tailed.

## Part 2 · Getting to the model (slides 8 to 10) [3 min]

What it is for: how the model was chosen, compressed to two slides. The history (which
loss, the first MLP, the patch CNN, the ablation, leave-one-day-out, the collapse) is in
the appendix if asked. The advisor's own position was the simplest model that scales; this
part says the data agreed.

**Slide 9 · At scale, the centre-pixel MLP beat the patch CNN** [1.5 min]. Table at 1.5M
parameters each on the test split: the MLP wins the training objective on both tracks and
at the median and p99; the CNN only reconstructs BP's AIA slightly better. The CNN's edge on
four images was in-sample and vanished on held-out days.
Point: the simplest model scales; no neighbourhood needed.

**Slide 10 · Choosing the model: width 336** [1.5 min]. Chart of BP median pixel error
against model size, 360k highlighted. Chosen by validation loss; 1.43M ties on validation
but only nudges the typical pixel and blows up on the stiff band; ElasticNet blows up from
722k on a few bright pixels; the square root plus Fourier input wins by validation on both
tracks.
Point: one MLP of width 336 per track; bigger is unstable on rare pixels.

## Part 3 · Results (slides 11 to 13) [3.5 min]

What it is for: the headline table and the shape of the error, shown once with the final
models. Part 4 then explains the one number that is behind.

**Slide 12 · Against supervised: all six on ElasticNet, four of six on BP** [2 min]. The
table, bold is better. ElasticNet: every column. BP: DEM MAE, EM error, W1 and AIA MSE;
supervised keeps DEM MSE (0.91 against our 1.75) and AIA MAE. Muted line: the old BP final
for reference.
Point: ahead on most measures; DEM MSE is set by flare cores.

**Slide 13 · The typical pixel and the average tell different stories** [1.5 min]. Two
callouts: median pixel error 0.036 against 0.112; 98% of our BP error sits in the worst 1%
of pixels (supervised 95%). Up to the 99th percentile we are level or ahead on BP; the gap
opens in the last 0.1%, the flare cores.
Point: the mean is a flare-core statistic.

## Part 4 · Flare cores: the failure and the fix (slides 14 to 22) [11 min]

What it is for: the centre of the talk and the new material since the last meeting. The
order is the story: the failure (a recap), the suggestions did not help, the cause, why the
loss does it, the fix, what training did on the new loss, flare cores now, what remains.

**Slide 15 · Flare cores, 1 pixel in 10,000, held 90% of the BP error** [1 min, recap]. The
old model: 0.01% of pixels, 90% of the DEM error (supervised 90% too). There it held 40% of
the solver's emission, peaked 23% as high, 0.2 dex too cool, with extra peaks on 83% of
curves. It fit 171, 193 and 211 Å to a few percent but reproduced only 23% of 94 Å. Figure:
the old model at random flare cores.
Point: the hot plasma was missing, and we read it as a training problem, which is where the
suggestions came from.

**Slide 16 · On the original loss, none of the suggestions helped** [1 min]. Table:
oversample flare cores, 120 epochs, 10M parameters, all on the original loss. Away from
flare cores all four score the same; at flare cores oversampling and 10M make it worse and
120 epochs moves emission from 35% to 37%.
Point: training harder does not help; the loss itself needs checking.

**Slide 17 · The cause: the network already sits at its loss's best answer** [2 min]. For
each pixel, solve the loss directly (it is convex): the best answer no network can beat. On
12,000 validation pixels the network sits 1 to 4% above it, 8% at flare cores. Table of four
answers under the loss: the network, the best answer, the solver's DEM, the supervised
model's DEM. The best answer itself reproduces only 26% of 94 Å, and the solver's DEM scores
2.4 times worse under our loss than a DEM without the hot plasma.
Point: the loss, not training, was the limit.

**Slide 18 · Why: hot plasma is expensive to include and cheap to leave out** [2 min].
Rule against price. Response table: a unit of warm plasma lights 171, 193 and 211 Å by tens
of DN/s, so it is in the DEM anyway and its 94 Å comes free; a unit of hot plasma lights no
channel above 0.4 DN/s, so a bright 94 Å reading needs hundreds of units, paid by the
sparsity term. Worked example at a flare core: hot plasma in costs 616, out costs 85.
Point: at μ = 100 leaving it out costs 8,500, and the plasma goes back in.

**Slide 19 · The fix: a stiffer band, chosen on validation** [1.5 min]. Screen table of
μ = 1 to 300, best answer per pixel with no network, scored on validation: stiffer helps
bright pixels and costs faint ones; μ = 100 takes most of the gain with Quiet error still
below the supervised model's. Muted line: μ is one scalar chosen on validation like any
loss weight, training never sees a DEM, and the 94 Å column needs no solver DEM.
Point: μ = 100.

**Slide 20 · On the stiff band, longer training is what helps** [1.5 min]. Runs table. The
band alone halves DEM MSE; the network then sits far above its best answer (118% at flare
cores), so epochs help; 120 epochs then 120 more continued from that checkpoint is the
final model (DEM MSE 1.75, AIA MSE 103); 240 from scratch misfits a few rare pixels' AIA;
oversampling still hurts; 10M and 680 wide blow up.
Point: the final BP model, and why that one.

**Slide 21 · Flare cores now: 93% of the solver's emission, 77% of its 94 Å** [1 min].
Table old / new / supervised / solver: emission 40% to 93%, 94 Å 23% to 77%, peak 23% to
66%, shift from 0.18 dex too cool to none, extra peaks 83% to 25%. Spurious peaks were 42%
of the old error, 3.5% now. Figure: the same random draw as slide 15, with the new model.
Point: fixed in the loss.

**Slide 22 · What remains: flare cores still set the BP DEM MSE** [1 min]. Error against
supervised by brightness: 0.55 on the faintest pixels, about 1 through the bright range,
2 at flare cores (the old model was at 18). Flare cores hold 94% of our BP error and the
DEM MSE gap. 94 Å at 77% against the solver's 92% is the one shortfall; longer training keeps
closing it, a third continuation is untested. Figure: reconstructed over observed AIA at
flare cores.
Point: the loss was the limit; training is now the limit, and it still has room.

## Part 5 · Next steps (slides 23 to 26) [2 min, then discussion]

**Slide 24 · What we learned** [1 min]. Six cards. The two new ones: check the loss first,
and size has a ceiling. The rest carry over: train on the objective, simple scales, heavy
tails, input encoding.

**Slide 25 · Next** [1 min]. Keep training BP (a third continuation, watching the AIA fit on
rare pixels); validate on every pixel; direct DEM output; uncertainty and AIA + XRT.

**Slide 26 · For discussion.** Three questions: which metric should lead the paper; train
the BP model longer or freeze at 1.9 times the supervised DEM MSE; is AIA + XRT in scope.

## Appendix (slides 27 to 35)

Pulled up only when asked.

**Slide 28 · Which loss.** Five losses optimised directly per pixel on 4 timestamps.
Fit-only losses reconstruct AIA better than BP yet land far from its DEM; BP's own terms
land on it. Why we train on the solver's objective, not on fit.

**Slide 29 · A per-pixel MLP missed BP's shape; a patch CNN tracked it.** The June story
on 4 images: the first 213k MLP was further from BP than a 1.5M patch CNN.

**Slide 30 · Ablation: the gap was capacity, not missing context.** Four variants at 1.45M
parameters; a centre-pixel MLP of that size closed the gap. Sparsity dot plot.

**Slide 31 · On unseen days, the patch CNN's edge disappeared.** Leave one image out, 4
folds; the flare fold was the weak spot. Why we carried the simpler MLP forward.

**Slide 32 · The first full-disk runs collapsed to an all-zero DEM.** Seven decades of
intensity, a dead softplus at step 0; the fix was a bounded input, warmup, a clamped
softplus and a guard.

**Slide 33 · How we score at scale.** The scoring protocol in full, with the block diagram
of one test day (64 random blocks per target, five targets).

**Slide 34 · ElasticNet: the final model, and its flare cores.** Width 336, square root plus
Fourier, 40 epochs, the solver's settings; ahead of supervised on all six. Flare cores are
not the BP kind of failure: no band to price, and it reproduces the channels as the
ElasticNet solver itself does (94 Å 53% against 55%). From 722k up it blows up.

**Slide 35 · Multi-peaked DEMs.** BP is multi-peaked on 13.5% of pixels; the new model flags
15.5% at 53% precision and 60% recall (old model 26% recall, supervised 72% at 24%
precision); where BP is multi-peaked our gap is the size of BP's own re-solve scatter.
