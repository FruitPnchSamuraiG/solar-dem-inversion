# 26: The solver's constrained problem and its unconstrained stand-ins (12 Nov 2026)

The solver (basis pursuit) solves a linear program at every pixel:

    min sum(x)  subject to  obs_c - h_c <= (D x)_c <= obs_c + h_c  for the six channels,  x >= 0,

with x the 54 basis weights, D the response of each channel to each basis element and
h_c = 1.4 sigma_c the band half-width. A network cannot be held to a rule on its output,
so training replaces the rule by a charge on breaking it. This folder compares the
textbook ways of doing that, first per pixel with no network (is the stand-in exact?),
then with a network trained on a fixed set of pixels (does training enforce the bands?).
Zeroed channels (the deconvolution clamp) keep their band around zero throughout: a
zeroed channel may receive at most h_c of light.

Scripts: `experiments/constrained_toy.py` (per pixel), `experiments/constrained_toy_net.py`
(network), `experiments/plot_constrained_toy.py` (figures and table), job files
`experiments/job_constrained_toy*.sbatch`, launcher `experiments/submit_constrained_toy.sh`.

## Terms

- **feasible pixel**: some DEM fits inside all six bands (the linear program has a solution).
- **half-width**: h_c; violations are measured in half-widths (1 = one full half-width outside).
- **dual price** of a channel: how much total weight the solver would save per half-width
  if that channel's band edge moved outward. Zero for a channel not at its edge.
- **single-channel price scale** P_c = h_c / max_k D[c,k]: the price when that channel
  alone is tight. Computable from the input (noise and response matrix), no solver.
- **price ratio** q = dual price / P_c, read from the solver's linear program per pixel.

## Per pixel, no network (`toy.json`, `toy.txt`; 2,000,000 training and 300,000 validation pixels)

- 90.3% of training pixels (92.1% of validation) are feasible; the solver's own flag agrees
  exactly (every pixel it solved at the tight band is feasible, every relaxed one is not).
- 38.7% of pixels have a zeroed channel; 86.7% of those are still feasible with the band
  kept at zero (92.6% of pixels without one).
- Infeasible pixels miss by little (`fig1_infeasible_band_multiple.png`): half become
  feasible with the band 1.27x as wide, 91% at 2x, 98.9% at 3x (the solver's first retry).
- The price ratio spans more than three decades: median 1.72, p90 9.6, p99 142, max 74,647
  (training pixels, largest channel per pixel).
- Each stand-in solved to its exact optimum on a validation sample balanced over
  brightness, share of feasible pixels inside every band within 0.001 half-widths
  (weighted to the population; `fig2_standins_per_pixel.png`):

  | stand-in | inside every band |
  |---|---|
  | quadratic penalty, mu = 100 (the loss trained so far) | 48.0% |
  | the same plus one multiplier update (augmented Lagrangian) | 95.2% |
  | multipliers, weight 10 raised when stuck, 7 updates | 100.0% (flare cores too) |
  | exact (linear) penalty at 16x the price scale | 96.5% |
  | exact penalty at 1,024x the price scale | 99.75% |

  The multiplier method reaches the solver's own total weight (ratio 1.0000).
- At infeasible pixels (balanced sample, 1,035 pixels) the multiplier method ends close to
  the smallest violation possible (largest channel, half-widths): excess over that minimum
  median 0.01 to 0.1 below flare cores, 0.68 at flare cores where the minimum itself is 3.3.
  The exact penalty at 1,024x ends further away: median excess 0.1 to 0.4, and 3.2 at flare
  cores. (The multiplier method drives the squared violation down, not the largest one, so
  it does not hit the minimum exactly.)

## With a network (`net_*.json`, `net_summary.txt`, `fig3_training_violations*.png`, `fig4_violation_vs_weight*.png`)

Setup: the final model's architecture (centre-pixel MLP, square root + 12 Fourier frequencies)
trained on one fixed set of 1,000,000 feasible training pixels under each stand-in, scored
on 276,177 held-out feasible validation pixels. Schedules and multiplier updates run over
the first 70% of training, then the learning rate decays to zero.

- **From scratch** (`net_<run>.json`): 336 wide (408,630 parameters), 200 epochs, LR 3e-4.
- **Warm start** (`net_<run>_warm.json`): from the final model (680 wide, 1,528,694 parameters,
  trained with zeroed channels free), 150 epochs, LR 1e-4. Every run here keeps the band at
  zero for zeroed channels.

Runs: `quad100` the loss used so far (mu = 100); `ramp` mu raised geometrically (1 or 100
to 10,000); `al` multipliers stored per pixel, mu doubled when the violation stalls (to
1,000); `alfix` the same with mu held at 100; `alnet` multipliers predicted by a second
network (primal-dual learning); `price` weight P_c / (2 eps) per pixel and channel with eps
lowered to 0.001; `huber` the exact penalty smoothed, slope 64 P_c, eps lowered to 0.001.

Warm start, held-out feasible pixels (DEM error: mean absolute difference from the
solver's own DEM, per bin):

| run | outside a band by > 0.01 half-widths | total weight / solver's (median) | DEM error |
|---|---|---|---|
| current final model, before training | 10.3% | 1.066 | 0.069 |
| `quad100` (loss used so far, zero band kept) | 7.7% | 1.075 | **0.060** |
| `price` (weight from the input, eps lowered) | 11.3% | 1.078 | 0.071 |
| `huber` (smoothed exact penalty) | 12.2% | 1.075 | 0.071 |
| `alfix` (multipliers, mu 100) | 5.3% | 1.118 | 0.080 |
| `alnet` (multiplier network, mu to 1,000) | 2.6% | 1.133 | 0.081 |
| `al` (multipliers, mu to 1,000) | 3.0% | 1.147 | 0.088 |
| `ramp` (mu 100 to 10,000) | **1.7%** | 1.171 | 0.092 |

From scratch the same order holds with larger costs: `quad100` 9.9%, 1.28x, 0.118;
`ramp` 3.7%, 3.33x, 0.389; `al` 5.7%, 3.55x, 0.460; `price` 13.3%, 1.31x, 0.118; `huber`
22.1%, 1.18x, 0.114; `alnet` diverged (67% outside).

What this shows:

- **A network does not reach zero violations on feasible pixels.** Violations fall during
  training and level off. Every way of pushing them lower moves the network along one
  trade-off: more total weight and a DEM further from the solver's (fig4). The solver's
  point (weight 1, no violation) is reached per pixel (fig2) but not by any network.
- **Why**: one network serves every pixel and cannot land exactly on each pixel's answer
  (the amortization gap). Pushed hard, it stays just inside the bands with a margin, and
  the margin costs weight. The learned multipliers show it: at the bands where the solver
  sits on the edge, the network's multipliers are zero (median learned / dual price 0 in
  every multiplier run that converged), so the multipliers never become the solver's dual prices.
- **Most accurate**: the loss used so far with zeroed channels keeping their band.
- **Zeroed channels**: the final model, trained with them free, gives 9.3% of zeroed
  channels more light than their band allows by over 0.1 half-widths (worst 1%: 5.4
  half-widths); keeping the band brings that to 0.2% and the DEM error to the solver's DEM
  from 0.064 to 0.056.
- **The weight from the input** (`price`, `huber`) is soft on the bright channels, whose
  price scale is tiny, and ends behind the fixed mu = 100 on both measures.
- **The multiplier network** diverges from scratch (multipliers overshoot while the
  network is far from any optimum) but works from the warm start: fewer violations than `al` at a smaller DEM error, and
  half the violations of `alfix` at the same error.

## No mu chosen by hand (`net_sched*.json`, `net_settle*.json`, `fig5_schedules.png`)

BP has no mu: it enforces the band exactly. mu exists only because training replaces the
rule by a charge. The method of multipliers takes the charge back to the rule: each pixel's
price is learned (price <- price + 2 mu x violation, gradient ascent on the dual), and mu is
raised by 10 only when a round fails to cut the violation measure (99th percentile of the
largest violation per training pixel) below 0.8 of its last value (Park and Van Hentenryck
2023; Nocedal and Wright 2006, Algorithm 17.4). All runs here start from scratch.

- **Textbook rule, a round every 10 epochs** (`sched0.1b`, `sched1`, `sched10`): the start does
  not matter (from 1 and 10 the same mu steps from epoch 40, from 0.1 one step later), but mu
  never stops rising: the network's violations level off near 0.2 half-widths (99th
  percentile) whatever mu is, so the rule keeps raising it to the ceiling (1e5 to 1e6). Held-out
  feasible pixels outside a band by > 0.01: 3.1 to 4.4%; weight 3.7 to 5.1x the solver's; DEM
  error 0.36 to 0.50. Nearly all the extra emission sits at logT 7.2, where AIA sees least
  (115x the solver's mean DEM there, against about 0.9x at logT 6.2 to 6.3).
- **Network-aware rule** (`settle0.1`, `settle1`, `settle10`, 400 epochs): a round ends once the
  training loss has settled (fell under 2% over 10 epochs), the textbook's condition that each
  round be solved first; mu stops rising once a 10x raise cut the measure by under 2x (theory
  predicts 10x), and that raise is undone. mu ended at 10, 10 and 100 (paths 0.1 -> 1 -> 10 ->
  100 -> 10; 1 -> 10 -> 100 -> 10; 10 -> 100 -> 1000 -> 100):

  | start | mu at the end | outside > 0.01 | weight / solver's | DEM error | flare-core DEM error |
  |---|---|---|---|---|---|
  | 0.1 | 10 | 19.5% | 1.18 | 0.190 | 100.9 |
  | 1 | 10 | 26.8% | 1.19 | 0.420 | 82.4 |
  | 10 | 100 | 9.9% | 1.37 | 0.146 | 44.6 |
  | fixed mu = 100 (by hand, 200 epochs) | 100 | 9.9% | 1.28 | 0.118 | 68.4 |

What this shows: the rule finds mu in the range validation had picked (10 to 100) without
being told. The reason is the violation law: the penalty's own drift is price / (2 mu) and
falls as 1/mu, while the network cannot resolve violations below its floor; mu of 10 to 100 is
where the drift drops below that floor, and past it a stiffer band only adds emission. Where
exactly the rule stops depends on one noisy comparison (the same 10 -> 100 raise cut the
measure 2.24x, 1.96x and 1.25x in the three runs, against a threshold of 2), and undoing the last raise brings
violations back up (fig5, middle); keeping it and only stopping further raises would avoid
that. The hand-picked mu = 100 is still slightly ahead on the overall DEM error.

Full-scale follow-up (all pixels, fresh pixels every epoch), queued 12 Nov on Torch:
`bp_h680_keep` (final recipe from scratch, zero band kept), `bp_h680_keep_warm` and
`bp_h680_keep_al` (60 epochs from the final model, loss used so far against multipliers
from frozen copies of the network), and `bp_h680_keep_settle` (the network-aware rule from
mu = 1, 200 epochs, multipliers capped at 256 x the price scale so that pixels with no DEM
inside the band get a finite push); see `experiments/submit_followup.sh`.
