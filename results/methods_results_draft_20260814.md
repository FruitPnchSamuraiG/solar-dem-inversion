# Draft contribution: Unsupervised amortized DEM inversion

> This is an editable Methods and Results component for integration into a
> larger manuscript. It deliberately omits a general introduction, literature
> review, operational details, and claims outside the AIA-only experiments.
> Bracketed notes identify details that should be confirmed with Samuel before
> final manuscript submission.

## Methods

### Problem formulation

Differential emission measure (DEM) inversion estimates the thermal plasma
distribution along each line of sight from a small set of EUV observations. For
an AIA-only observation vector \(o\) with six channels, the forward model is
\(o = R x\), where \(R\) is the temperature-response matrix and \(x\) is the
non-negative DEM representation. We report the DEM on 18 temperature bins. The
inverse problem is underdetermined, so the selected DEM depends on the
regularization and feasibility rule used by the inversion method.

We consider two reference objectives. Basis Pursuit (BP) seeks a sparse,
non-negative solution subject to an observation-dependent tolerance band. When
the initial band is infeasible, its solver relaxes the tolerance. Elastic Net
(ENet) instead balances reconstruction fidelity with L1 and L2 penalties, which
generally favors smoother solutions. These solvers provide useful scientific
references, but solving an optimization problem independently at every image
pixel is expensive at full-disk cadence.

### Label-free amortized inversion

We train a neural network to map an AIA observation directly to a DEM while
minimizing a differentiable relaxation of the corresponding BP or ENet
objective. Crucially, solver-produced DEMs are **not** training targets. The
network sees observations, the response matrix, and the physical objective; BP
and ENet DEM solutions are retained only for evaluation. This is amortized
optimization: the cost of finding a valid solution is paid during training, and
one shared network subsequently supplies a DEM for each new pixel.

The BP-track loss penalizes violations of the lower and upper observational
tolerance bands together with the sparsity term. The ENet-track loss uses the
same forward model with the ENet regularization. Training the two objectives
separately is important: BP and ENet intentionally select different members of
the feasible DEM family.

### Data, split, and architecture

We used 1,223 Hofmeister-PSF-deconvolved SDO/AIA timestamps spanning two years.
The split was made by timestamp, not by pixel: 917 timestamps for training, 153
for validation, and a disjoint 153 for final testing. Thus the held-out test set
contains solar states and complete images never observed during training or
checkpoint selection.

The production model is a per-pixel MLP (`mlp6`) with 176,374 parameters
(hidden width 232). It receives six AIA values, applies an input lifting and
SiLU nonlinearities, and predicts 54 non-negative basis coefficients through a
softplus output head; these are converted to an 18-bin DEM. Inputs are
transformed with `log1p` for numerical conditioning, while the forward-model
loss and all physical metrics remain in the original data-number units. We also
tested a 9x9 patch CNN aligned to the lower-resolution DEM grid, allowing us to
ask whether spatial context is necessary.

### Evaluation

We evaluate agreement with the corresponding reference solver using DEM error,
Hoyer sparsity of the coefficient vector, and AIA resynthesis error. Resynthesis
error compares \(R\hat{x}\) directly with the observed AIA channels and does not
depend on a solver label. We additionally use the one-dimensional Wasserstein
distance (W1, in dex) between normalized DEM curves to emphasize agreement in
thermal shape rather than only amplitude.

Means are not sufficient for these data. The BP barrier objective has a
heavy-tailed error distribution, so model selection uses loss percentiles. For
the external Table-1-style evaluation, we report the requested mean DEM MSE but
also examine the per-pixel squared-error distribution and its upper-tail share.

## Results

### A small per-pixel network generalizes to unseen solar timestamps

The MLP generalized from the timestamp-disjoint training set to the untouched
153-timestamp test split: test metrics reproduced validation metrics to three
decimal places across the evaluated models. On this honest split, the patch CNN
did not show a consistent benefit over the per-pixel MLP. The MLP matches or
improves the objective percentiles and avoids the 81-input cost of a spatial
patch for every prediction. These results support the simpler per-pixel model as
the production architecture for the present AIA-only task.

### Capacity has a measurable trade-off, and 176k parameters is sufficient

We rescored a width sweep spanning approximately 10,000 to 11.2 million
parameters on the held-out test split. The 176k-parameter model is 8.1 times
smaller than the original 1.43M-parameter baseline and gives the best AIA
resynthesis error in both BP and ENet tracks. In the ENet track it also gives the
best solver-agreement values found in the sweep.

Increasing BP-model capacity improves coefficient sparsity toward the BP
reference (from 2.099 at 176k parameters to 1.826 at 11.2M), but slightly
worsens AIA resynthesis error (4.792 to 4.980). DEM error and high-percentile
loss are otherwise nearly flat above 176k parameters. We therefore selected the
176k model as the resynthesis-oriented production point, while making the
sparsity-versus-resynthesis trade-off explicit rather than treating scale as an
unqualified improvement.

### Multi-thermal structure is detected with high precision but limited recall

On a held-out BP sample, 14.17% of pixels had genuinely bimodal, interior DEM
solutions. The BP MLP identified multi-thermal pixels with approximately 80%
precision but about 30% recall. Thus it can flag many credible multi-thermal
locations, but it should not be described as reproducing all bimodal DEM curves.

The missed cases are not explained by insufficient model capacity. Across a
20k-to-11.2M parameter sweep, the DEM-error penalty at multimodal pixels stays
near threefold and recall saturates near 30%. The network performs best when two
peaks are well separated and have similar heights, and worst when a weak
component appears as a shoulder on a dominant peak.

To determine whether this remaining gap is a model failure or an ambiguity in
the reference, we re-solved BP under simulated photon-noise perturbations for 80
unimodal and 80 bimodal bright test pixels. At bimodal pixels, the average
network-to-BP discrepancy was 0.98 times BP's own perturbation-induced
self-scatter. BP self-scatter roughly doubled from 0.31 at unimodal pixels to
0.60 at bimodal pixels. Therefore, much of the apparent disagreement at
multi-thermal pixels is consistent with the reference solver's instability under
noise, rather than a capacity limitation of the network. A remaining, bounded
deficiency is an undershoot in the hot AIA channels (94 and 131 Å), accompanied
by DEM peaks about 0.1--0.15 lower in log-temperature at bright pixels.

### Full shared-test comparison with supervised models

Samuel Pérez-Díaz supplied a full spatial version of the same 153 held-out
timestamps used in our split. It contains 48,960 256x256 blocks per solver
track, compared with 9,792 blocks in our earlier staged evaluation. We evaluated
the final h232 networks over all finite pixels. The table below juxtaposes those
unsupervised results with the supervised values supplied by Samuel.

| Reference track | Method | DEM MSE | EM relative error (%) | W1 (dex) |
|---|---|---:|---:|---:|
| BP | Supervised model | 0.97 | 13.3 | 0.131 |
| BP | Unsupervised `mlp6` h232 | 4.191 | 15.29 | **0.0845** |
| ENet | Supervised model | 0.29 | 14.1 | 0.129 |
| ENet | Unsupervised `mlp6` h232 | 4.345 | 71.66 | 0.2997 |

For BP, the unsupervised model has similar relative EM error and lower W1, but a
higher mean DEM MSE. The MSE should be interpreted cautiously: it is strongly
tail-weighted. Its median per-pixel SSE (summed across the 18 DEM bins) is
0.0262, whereas the worst 1% of pixels contributes 97.80% of the total squared
error and the worst 0.01% contributes 89.71%. Removing only the single worst
pixel changes the reported MSE by 3.46%, so this is a rare population of hard
pixels rather than one corrupt value.

The current ENet model differs more broadly from its solver reference. Its
median per-pixel SSE is 4.04, its p90 is 38.4, and its worst 1% contributes
82.87% of total squared error. Removing its worst pixel changes mean MSE by only
1.10%. Thus the ENet gap is not explained by isolated outliers and is an
important target for further work.

The supervised table currently provides only mean metrics. A strict
typical-pixel comparison requires applying the same per-pixel percentile and
tail analysis to the supervised predictions. In addition, the exact Table-1
pixel mask and Bright/Quiet thresholds should be confirmed before this table is
described as a fully protocol-matched head-to-head comparison.

## Summary of contribution

The results show that a compact, label-free neural emulator can amortize BP and
ENet DEM inversion across previously unseen solar observations. For BP, it
matches the reference's thermal-shape metric closely while exposing that a mean
solver-distance metric is dominated by a narrow, scientifically difficult error
tail. The timestamp-disjoint tests, capacity sweep, and noise-resolve experiment
bound the principal limitations: spatial context is not required in this setting,
additional capacity does not resolve ambiguous multi-thermal solutions, and BP's
own noise sensitivity explains much of their apparent mismatch. The present ENet
result is weaker and should be reported as such.
