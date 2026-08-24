# ML4PS 2026 workshop paper — deferred plan

> **Status (2026-08-24): deferred.** Samuel and Hriday decided not to submit
> this workshop paper. Preserve this as a possible future direction; the active
> deliverable is now an advisor-facing presentation of the completed work and
> the next research questions.

## Submission decision

Submit to the **Research** track as high-quality work in progress. The official
deadline is 2026-09-12 23:59 AoE; the limit is four pages excluding references.

This paper must stand independently of Samuel Pérez-Díaz's supervised models and
his larger solar-physics manuscript. It will use our own label-free training,
timestamp-disjoint evaluation, architecture ablations, and robustness analyses.
A supervised comparison can appear only as a clearly labelled external baseline,
with Samuel's explicit approval; it is not necessary for the paper's central
claim.

**Authorship:** Hriday is first author. Confirm the remaining author list,
affiliations, and acknowledgements with Samuel before submission.

## Working title

**Label-Free Amortized Neural Inversion for Solar Differential Emission Measure
Estimation**

Alternative, more evaluation-focused:

**Evaluating Label-Free Neural Surrogates for Solar DEM Inversion Under Solver
Ambiguity**

## One-sentence thesis

A compact per-pixel neural network can amortize a physically constrained solar
DEM inversion objective without solver DEM labels, generalize across unseen solar
timestamps, and expose why mean agreement with an ill-posed reference solver is
not a sufficient quality measure.

## Claims to make

1. **Label-free amortized inversion works.** The network is trained directly
   from observations, response functions, and the BP/ENet objectives; solver
   DEMs are evaluation references, not training targets.
2. **The simple MLP is sufficient.** On timestamp-disjoint tests, a 176k
   parameter per-pixel MLP is a strong production choice; the patch CNN does not
   provide a consistent held-out advantage.
3. **Capacity is not the limiting explanation for multimodal mismatch.** The
   size sweep and BP noise re-solves show that close/unequal multi-thermal peaks
   are partly ambiguous in the reference solver itself.
4. **Evaluation must be robust to tails.** Mean BP barrier loss and per-pixel
   DEM MSE can be dominated by difficult pixels; percentiles, W1, and
   self-consistency reveal a materially different picture.

## What is deliberately out

- A claim that our model outperforms Samuel's supervised model.
- Full-disc web deployment details and cluster/SLURM engineering.
- AIA+XRT experiments, uncertainty heads, and unvalidated future work.
- A full ENet success story. ENet can be a compact stress-test/limitation result,
  not the paper's headline.
- The complete chronology of exploratory experiments.

## Four-page structure

### Page 1 — Motivation and method

- Solar DEM inversion: six AIA measurements, 18 temperature bins, per-pixel
  constrained optimization.
- BP and ENet objectives in compact notation.
- Label-free amortized objective and the 176k MLP.
- One small schematic: AIA observation -> network -> DEM -> response-model
  reconstruction/loss.

### Page 2 — Experimental design

- Hofmeister-deconvolved AIA data; timestamp-disjoint 917/153/153 split.
- MLP versus 9x9 patch CNN and width sweep.
- Metrics: resynthesis, W1 shape distance, solver agreement, loss percentiles.
- State plainly that solver labels are not used for training.

### Page 3 — Main quantitative results

- Table: MLP/CNN and selected width-sweep results on unseen timestamps.
- Figure: width/capacity trade-off or MLP-vs-CNN held-out comparison.
- Result: h232 (176k) is 8.1x smaller than the 1.43M baseline while retaining
  the relevant held-out performance.

### Page 4 — Ambiguity-aware evaluation and limitations

- Figure: BP self-consistency under photon-noise re-solves, or compact
  multimodal/capacity figure.
- Explain high-precision, low-recall multimodal detection and BP's
  self-scatter at those pixels.
- Tail-aware evaluation: means are not headline model-selection statistics.
- Brief limitations: hot-channel undershoot; current ENet mismatch; no claim of
  recovering all solver-specific multimodal structure.

## Figures and table to prepare

1. **Method diagram** — label-free objective and forward response.
2. **MLP vs CNN / width-sweep figure** — one clear held-out comparison.
3. **BP self-consistency or multimodal-quality figure** — choose the clearest,
   not both if space is tight.
4. **One compact results table** — production MLP, CNN baseline, and perhaps
   one larger/smaller width.

The full-disc interactive viewer is supplementary/demo material, not a required
paper figure. For a static full-disc figure, regenerate assets with Samuel's
visual conventions: turbo DEM maps, inferno mean/std maps, wavelength-specific
SunPy AIA maps, and shared AIA color limits within each comparison pair.

## Schedule to submission

| Date | Deliverable |
|---|---|
| Aug 24–27 | Freeze narrative, choose figures/table, match viewer rendering conventions. |
| Aug 28–31 | First complete four-page prose draft. |
| Sep 1–5 | Figures, numerical verification, coauthor review. |
| Sep 6–9 | Revision, references, formatting against the released submission template. |
| Sep 10–11 | Final author approval and upload buffer. |
| Sep 12 | Submit before 23:59 AoE. |

## Immediate next work

1. Add Samuel-style zoom/pan and visual rendering conventions to our viewer.
2. Regenerate the selected full-disc comparison assets with shared pairwise
   scales and add residual/difference views.
3. Convert this plan and the existing Methods/Results prose into a four-page
   manuscript source once the official template is released.
