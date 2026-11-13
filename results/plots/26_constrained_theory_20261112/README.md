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

## With a network (`net_*.json`, `net_summary.txt`, `fig3_training_violations*.png`)

Filled in below once the runs finished.
