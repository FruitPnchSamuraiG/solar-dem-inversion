# Methods & Results — draft outline for review

**Status**: review draft. The core internal results and the external full-test
comparison are now recorded below; the remaining task is to verify the few
open protocol details before turning this into prose.
**Scope**: this is a component of a larger paper, not a standalone one. No intro,
no literature review, no discussion of the field. Methods and Results, written so
they can be dropped into someone else's frame.
**Target**: outline agreed and filled by Monday 2026-08-10.

---

## The spine, in one paragraph

DEM inversion is a per-pixel constrained optimization, which is why full-disk
inversion at observing cadence is not currently done. We train a network to
minimize the solver's own objective directly against the observations, so the
solver's output is never a training target and exists only as a yardstick. The
resulting emulator reproduces both reference solvers on 153 entirely unseen
timestamps, needs only 176k parameters to do it, and requires no spatial context.
Where it appears to fail — multi-thermal pixels — most of the apparent error is
the reference solver's own irreproducibility under photon noise, which we measure
directly rather than assume.

**The one honest through-line**: the hard part was not making the network work,
it was establishing what the numbers mean. Four separate headline figures turned
out to be measuring artifacts. That is worth saying out loud, because a reader
reproducing this will hit the same four.

---

## The production model — final, as of 2026-08-07

**mlp6, hidden = 232, 176,374 parameters.** Per-pixel MLP, no spatial context,
one for each solver track. 8.11x smaller than the 1,430,774-parameter baseline
and 63x smaller than the largest model trained.

| | BP track | ENet track |
|---|---|---|
| `sp_coef` (BP reference 1.79) | 2.099 | 3.699 |
| `mae_dem` | 0.1266 | **0.0077** |
| `mae_aia` | **4.792** | **4.641** |
| loss p50 / p99 | 0.965 / 17.90 | 1.152 / 41.29 |
| multimodal precision / recall | 81.8% / 26.4% | 36.0% / 6.9% |
| `mae_dem` penalty at multimodal pixels | 3.02x | 2.63x |

Bolded values are the best in the entire 11-width sweep, in both directions.
`mae_aia` is the one metric with no solver in it, and 176k holds the sweep's best
value on both tracks; the ENet track's `mae_dem` and p99 are also outright sweep
bests, beating the 1.43M baseline (0.0104 / 42.85).

**The single honest cost**: barrier `sp_coef` 1.904 (1.43M) -> 2.099 (176k),
i.e. further from BP's 1.79. Everything else is flat or better.

**If BP sparsity fidelity is the headline claim, this decision flips** — and not
to the 360k fallback quoted before 2026-08-07. The upward sweep showed `sp_coef`
keeps improving past the baseline (1.904 -> 1.881 -> 1.856 -> **1.826** at 11.2M),
so the alternative is the *largest* model, bought at 4% worse resynthesis
(`mae_aia` 4.792 -> 4.980). One trade, two defensible ends, no third option: every
other metric is flat across that range.

---

## METHODS

### M1 — Problem setup
The forward model `o = Rx`: 6 AIA channels, 18 temperature bins, so the inversion
is underdetermined by construction and a regularizer chooses among a solution
family. Cost of the per-pixel solve is the motivation for everything that follows.
*Short. Half a page. This is context the larger paper may already carry — check
before writing.*

### M2 — The two reference objectives
Basis Pursuit and Elastic Net stated **as objectives, not as black boxes** — this
matters, because the network minimizes them rather than imitating their output.
The tolerance band `o − tσ ≤ Rx ≤ o + tσ` and BP's escalating-tolerance schedule
(t = 1.4 → 2.8 → …), which becomes relevant twice later.

### M3 — Training as amortized optimization *(the central methods claim)*
The network minimizes a smooth relaxation of the solver objective against the
observation. **No labels.** Solver outputs are computed for the same pixels but
used only for evaluation. Consequences worth stating: the network is not bounded
by label quality; it satisfies the tight tolerance band more consistently than BP,
which relaxes when the LP goes infeasible; and it is free to disagree with any
individual solve.
Define the barrier loss `Σ relu(violation)² / σ²` and the ENet loss. Note that the
barrier loss is heavy-tailed **by construction** — forward-reference M7.

### M4 — Architecture
Per-pixel MLP, 6 inputs → input lifting → SiLU body → softplus head over 54 basis
coefficients → 18 DEM bins. The patch CNN (9×9 neighborhood, stride-2 to match the
DEM grid) as the alternative we tested against it.
*Deliberately brief — the architecture is not the contribution, and R2 is going to
argue the neighborhood buys nothing.*

### M5 — Data and splits
1,223 timestamps over 2 years, twice-daily. Hofmeister PSF deconvolution
(validated channel-wise: 94/131/335 Å corrected 29–35% vs 8–12% for 171/193/211 Å,
the correct scattered-light signature). **Split by timestamp**, 917/153/153 — say
plainly that held-out *pixels* of seen images is not a test set, and that we made
that mistake early. Zarr staging and block sampling (chunks are `(…,1)`, so a
single-pixel read decompresses a 256×256 block).

### M6 — Numerical conditioning *(short but keep it)*
Raw DN spans ~6 decades full-disk vs ~1 decade in the on-disk crops the
architecture was validated on. Fed raw, `|Rx|` overshoots the band ~7× at
initialization, the barrier hits 5e13, and `softplus` underflows to a region where
its gradient is *exactly* zero — 98% of outputs dead by step 60, at which point the
run trains smoothly to a bit-identical loss forever.
Fix: `log1p` on the network input only (loss and all metrics stay in physical DN),
3,000-step LR warmup, clamped softplus.
**Argument for keeping this**: it is a silent failure that produces clean exit
codes and converged-looking logs, and anyone rebuilding this hits it. Two paragraphs.

### M7 — Evaluation metrics, and why each one is limited
Define once, so Results can just report:
- `sp_coef` — Hoyer sparsity on the 54 coefficients. **BP reference 1.79.** Note
  explicitly that it is meaningless as a quality measure on the ENet track, which
  trades sparsity for smoothness by design.
- `mae_dem` — distance to the solver's DEM. **Comparable within a track only**;
  BP's spikes are far harder to hit than ENet's smooth curves.
- `mae_aia` — resynthesis error against the real observation. No solver involved.
  **The only number comparable across tracks.**
- **Training-objective summaries:** use percentiles, not means. One pixel in
  5,013,504 was 94% of the *mean barrier loss* and inverted an architecture
  ranking. This warning applies to the heavy-tailed barrier objective.
- **External Table-1 protocol:** report its required mean DEM MSE alongside EM
  relative error and W1. Mean MSE is also tail-sensitive, so report the
  leave-one-worst-pixel check and add the full per-pixel percentile/tail summary
  before treating MSE as a typical-pixel claim.

---

## RESULTS

### R1 — The emulator reproduces both solvers on unseen timestamps
*Claim: it works, and the generalization is real.*
Four models on the untouched 153-timestamp test split (~5.0M pixels).
**Test reproduces validation to the third decimal** across every metric — no
overfitting across 153 timestamps never seen in training or in checkpoint
selection. Table: sp_coef, mae_aia, mae_dem, test/val side by side.

### R1b — External full-test comparison to supervised models
*Claim: on Samuel's full shared test set, the unsupervised BP emulator has
competitive curve-shape agreement and EM error, while the current ENet emulator
does not yet match the supervised ENet result.*

Samuel supplied a full version of the same 153 held-out timestamps (48,960
256x256 blocks per solver track, rather than our earlier 9,792-block staging).
Our h232 networks were evaluated once over every finite label in those shared
Zarr test sets. The supervised values below are Samuel's Table-1 values; the
unsupervised values are our rerun with the same named metrics. These are
comparison results, not training labels: our networks were trained directly from
the corresponding inverse objectives.

| Reference track | Method | DEM MSE | EM rel. err. (%) | W1 (dex) |
|---|---|---:|---:|---:|
| BP | Samuel supervised | 0.97 | 13.3 | 0.131 |
| BP | Ours, unsupervised mlp6 h232 | 4.191 | 15.29 | **0.0845** |
| ENet | Samuel supervised | 0.29 | 14.1 | 0.129 |
| ENet | Ours, unsupervised mlp6 h232 | 4.345 | 71.66 | 0.2997 |

The full-set MSE difference is **not** explained by one pathological pixel. On
BP, excluding the largest per-pixel squared DEM error (block 13,061; row 40;
column 71) changes MSE 4.191 to 4.046: 3.46% of total MSE. On ENet, excluding its
worst pixel (block 30,789; row 29; column 0) changes MSE 4.345 to 4.297: 1.10%.
The corresponding relative-error and W1 changes are negligible. Thus the high
mean MSE reflects a broader error tail, not the single-outlier failure already
known for the barrier training loss. We still need per-pixel p50/p90/p99/p99.9
and top-tail shares to say how broad that tail is.

**Protocol check before final prose:** confirm Samuel's exact Table-1 masking
and bright/quiet thresholds. Our current full row uses all finite labels; no
bright/quiet split is yet reported. Do not call this a strict head-to-head claim
until that masking is confirmed.

### R2 — Spatial context buys nothing
*Claim: the per-pixel MLP is not a compromise, it is the better model.*
mlp6 wins the training objective on both losses and **every loss percentile**
(median 0.984 vs 1.007 → p99.99 148 vs 255), with 2.5× the bimodal recall at equal
precision. The CNN's only win was in-sample, on held-out pixels of images it had
trained on, and it evaporated the moment the evaluation became honest — five
independent looks, four of them post-fix, all agreeing.
Practical corollary worth one line: the CNN reads 81 pixels per prediction to the
MLP's 1, at full-disk 4096² resolution.

### R3 — 176k parameters is enough
*Claim: the model can be an order of magnitude smaller than we built it.*
16-run width sweep, 1.43M → 10k, both losses, all rescored on the test split.
h232 = 176k, **8.1× smaller**, better `mae_aia` than the baseline on both tracks
and the sweep's best ENet numbers outright. Cost is one line of honesty: barrier
`sp_coef` drifts 1.904 → 2.099, i.e. away from BP's 1.79; h336 (360k, 1.981) is the
conservative fallback.
Two traps to report rather than hide: h48 posts the *closest* sparsity in the whole
sweep (1.494) while being completely broken, and the apparent recall cliff at h160
is substantially a detection-threshold artifact of the 15% prominence criterion,
not a capacity cliff — `mae_dem` moves only 13% across that step.
The axis is now bounded on both ends, 10k → 11.2M, and going upward exposes a
**tradeoff invisible in the downward sweep**: `sp_coef` improves monotonically
toward BP's 1.79 with size (2.099 → 1.826) while `mae_aia` degrades monotonically
(4.792 → 4.980), with `mae_dem` and p99 flat throughout. Sparsity fidelity and
resynthesis quality are bought against each other along the size axis, and 176k
sits at the resynthesis end of it.

### R4 — Multi-thermal plasma: detected, not reproduced
*Claim: state precisely and defensibly what the network does with bimodality.*
Census over 190,337 held-out pixels: **14.17%** genuinely bimodal (interior peaks;
only 11.2% of raw detections were boundary artifacts, a suspicion we tested and
rejected). Prevalence rises monotonically with brightness, 7.4% → 30.3%, and
concentrates at *tight* tolerance, so it is not an artifact of relaxed solving.
The network **flags** these at 79–81% precision against that 14.17% base rate —
5.6× better than chance, and contrary to the amortization argument that predicted
it impossible. But recall is only ~30%.
*(An earlier read of a filtered figure selection suggested ~14% on the strongest
double peaks. The systematic grading below supersedes it and points the other way
— do not carry the 14% figure forward.)*
**The sentence the whole section exists to earn**: peak co-occurrence is a binary
shape test, not curve agreement. Figure panels pass it while the network draws a
broad blob against BP's two spikes. We write "detects at high precision, low
recall", never "captures bimodality".

**Which multimodal structure survives** — the part that makes the recall figure
interpretable instead of just low. Grading all 82,863 BP-multimodal pixels by peak
separation and by weaker/stronger peak height ratio, recall runs *upward* in both:
12.6% at 2–5 bins of separation → **45.6%** at 6–7, and 20.8% at ratio 0.15–0.33 →
33.4% at 0.51–0.74. `mae_dem` agrees exactly (0.456 → 0.223). Recall on 3+ modes
(42.2%) exceeds recall on 2 (28.6%).
So the network is **best on well-separated, comparable-height components and worst
on shoulders of a dominant peak** — it misses the marginal end, which is also the
end where BP is most likely tie-breaking at the noise level. We predicted the
opposite before measuring; report that we tested it.

### R5 — Most of the residual gap belongs to the label
*Claim: the reframe. This is the most interesting result and should read that way.*
`mae_dem` is 3× worse at bimodal pixels (0.29 vs 0.10) — and **flat at 3.0–3.2×
across the entire size axis, 20k to 11.2M**, 500× the parameters with no movement.
Recall saturates too, peaking at 2.83M (29.9%) and *declining* above it. Capacity
is ruled out from both directions, which points at the target.
So we measured BP against itself: 80 unimodal + 80 bimodal bright test pixels,
30 re-solves each under simulated photon noise.

| | BP self-scatter | mean \|BP\| | noise share | \|NN−BP\| | ratio |
|---|---|---|---|---|---|
| unimodal | 0.3097 | 0.8351 | 37.1% | 0.2135 | **0.69×** |
| bimodal | 0.6027 | 1.3963 | 43.2% | 0.5895 | **0.98×** |

Re-solving one pixel under a single noise draw moves BP's own answer as much as
our prediction differs from it. At unimodal pixels the network is **closer to BP
than BP is to itself**. BP's self-scatter also roughly doubles at bimodal pixels
(0.31 → 0.60), which is essentially the entire 3× penalty.
Report the falsified hypothesis too: the network is *not* estimating the noise
expectation — `|NN−ens|` (1.10×) is worse than `|NN−BP|` (0.98×).

### R6 — What is actually still wrong
*Claim: one real deficiency, bounded.*
Hot-channel undershoot. At bright pixels the networks resynthesize 94/131 Å too
faint and their DEM peaks sit ~0.1–0.15 low in logT. It is the only quantity in R5
with a ratio above 1 (**1.17×**, hot bins at bimodal pixels), and it is the same
peak shift the leave-one-timestamp-out flare fold found at small scale. Unchanged
by scale, unchanged by capacity.

---

## In / out — needs your call

| Candidate | Rec. | Why |
|---|---|---|
| Zerochill / NaN fix (10.2% → 0.02%) | **IN**, M5 | Affects whether the labels are valid at exactly the bright active-region pixels the paper is about. Mechanism is interesting: σ ~ √counts shrinks the *relative* band until it is narrower than R's own accuracy, so the LP goes infeasible at **high** SNR. |
| Deconvolution positivity clamp | **IN**, one para in M5 | 4.4% of pixels carry a hard zero in 171/193 Å, the two brightest channels. A reader must know the label was fitted to a detector artifact there. |
| Collapse debugging (M6) | **IN**, short | Argued above. Two paragraphs, not four. |
| One-pixel-is-94%-of-the-mean | **IN**, M7 + R2 | It inverted a model ranking. It is a methods warning, not a result. |
| Leave-one-timestamp-out study | **OUT** | Superseded by the real test split. One sentence in R2 at most. |
| Crop-era experiments (loss comparison, optimizer, Wasserstein) | **OUT** | Pre-scale, different data regime. |
| SLURM / scheduling / staging costs | **OUT** | Reproducibility appendix if anywhere. |
| Uncertainty head | **OUT** | Not built. Mention as future work only if the larger paper wants it. |

---

## Blockers before this can be finished

1. **ENet hyperparameters unconfirmed with Samuel.** Generation used `alpha=1,
   lam=0.5, C=n_obs` (the defaults). If he validated different values, every ENet
   number in R1/R3 is against the wrong objective and the labels need regenerating
   (~2.5 h + 40 min restage). **This is the one that can actually cost us Monday —
   ask today.**
2. **AIA+XRT** — in scope for this write-up or not? Two of the four models in the
   original project framing do not exist yet.
3. `bp_self_consistency.py`'s `enet_*` rows compare ENet-trained models against
   *BP* labels. Not used in any conclusion above, but must be fixed or deleted
   before anyone reads the script.

---

## Figures — proposed set

1. Solver vs 1.43M vs 176k DEM curves, BP and ENet side by side *(exists,
   `results/plots/11_final_h232_20260803/`)*
2. Size axis 10k → 11.2M: detection, `mae_dem`, and recall by peak-quality tier
   *(exists, `results/plots/12_multimodal_size_20260807/multimodal_vs_size.png`)*
3. Bimodal examples, both groups: reproduced and missed *(exists)*
4. BP self-consistency: perturbation spaghetti with the network overlaid — the
   figure that makes R5 visual rather than tabular
5. **[new]** Full-disk per-logT-bin maps, solver vs network. See below.
