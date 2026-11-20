# 30: Hard-pixel toy and figures of the method at work (20 Nov 2026)

## The set (`experiments/hard_toy_set.py`)

9,702 bright pixels from the training split that BP solves at the tight band, each with its LP
solved (optimum, DEM, dual prices): 3,870 at 3.2-10x the Bright threshold, 2,977 at 10-32x, 2,855
flare cores. Held-out: the earlier toy's validation sets cut to the same bands (3,015 and 4,391
pixels).

## Per pixel, no network (`per_pixel/`, `experiments/al_trajectory.py`)

The method of multipliers on single pixels: each round minimizes our loss exactly at fixed fines,
then updates the fines. At mu 10 the worst band miss reaches zero within 1 to 4 rounds at all six
pixels shown and the fines settle on BP's dual prices (the LP's exact multipliers); the DEM settles
on the LP's DEM. mu 100 gets there in 1 to 2 rounds; mu 1 has not after 10 rounds. mu sets the speed,
not the answer.

## Network toy runs (one network, 1.5M, trained only on the set; `experiments/submit_hard_toy.sh`)

Reference: method of multipliers mu 10, Adam lr 3e-4, a round every 20 epochs, 600 epochs, batch
512. One setting changed per run. Tables in `<set>/summary.txt`, curves in `<set>/compare_*.png`.

### Hard set (9,702 bright pixels BP solves at the tight band; `hard/`)
| run | train outside band % | worst-1% miss | weight / LP | DEM MAE vs LP |
|---|---|---|---|---|
| reference | 78.7 | 2.15 | 1.22 | 5.17 |
| mu 1 | 94.3 | 4.78 | 1.00 | 5.39 |
| mu 100 | 48.3 | 1.55 | 1.76 | 10.4 |
| no fines (penalty only) | 92.3 | 5.94 | 0.78 | 6.15 |
| lr 1e-4 | 83.2 | 5.52 | 1.54 | 8.79 |
| lr 1e-3 | 83.3 | 1.53 | 1.12 | 4.40 |
| round 5 / 50 epochs | 71.0 / 78.1 | 2.08 / 1.92 | 1.24 / 1.18 | 5.48 / 4.87 |
| reference, 3000 epochs | 89.5 | 0.71 | 1.02 | 2.16 |
| lr 1e-3, 3000 epochs | 95.1 | 1.09 | 1.04 | 2.67 |

- 600 epochs is far from converged: at 3000 the network approaches BP's exact answer (weight 1.02x,
  DEM error down 2.4x, worst-1% miss under one half-width) but almost every pixel sits slightly
  outside its band (a small miss everywhere instead of exactly inside).
- mu trades violations for weight, as before: mu 100 has the fewest pixels outside but 1.76x the
  weight and twice the DEM error; mu 1 is too slow; without fines the network drops plasma (0.78x).
- Learning rate: 1e-4 too slow; 1e-3 fastest at 600 epochs, behind 3e-4 at 3000.
- Round length (5 / 20 / 50 epochs) barely matters.
- `hard/pixels_ref.png`, `hard/pixel_*.png`: on a network the fines do not settle on BP's dual prices
  as they do per pixel (`per_pixel/`): they overshoot to several times the price and jump between
  rounds, and the band miss flickers between inside and about one half-width.

### 10k hard + 10k easy (`mix/`)
Same ranking. Reference: 57% of training pixels outside, worst-1% miss 1.32, weight 1.03x, DEM MAE
vs LP 1.74; mu 100 27% outside but weight 1.42x; mu 1 worst-1% miss 12.7; no fines weight 0.78x.

### Every pixel, no BP flags, truncated penalty (`mixall/`; the mix set + 2,000 pixels BP widened + 2,000 it could not solve)
Per class, training pixels, miss in tight half-widths:
| run | tight-band pixels inside % / DEM MAE vs LP | widened: inside BP's widened band % | unsolved: miss p50 / p90 |
|---|---|---|---|
| no flags, tight band everywhere | 43.4 / 1.95 | 98.2 | 5.9 / 8.4 |
| capped at 1 half-width | 37.7 / 11.3 (collapsed: weight 0.08x LP) | 54.8 | 19 / 53 |
| capped at 3 | 42.2 / 4.15 | 96.8 | 28 / 53 |
| capped at 10 | 45.4 / 2.11 | 97.6 | 17 / 100 |
| BP's flags (the final model's way) | pending | | |

- With no flags at all, the network lands 98% of the pixels BP widened inside BP's widened band by
  itself, and the unsolvable pixels miss by ~6 half-widths (what they need).
- The cap does what it says (past it a pixel exerts no pull), but on a network the abandoned pixels
  drift far (worst 10% miss 50 to 100 half-widths): wild resynthesis. A cap of 1 kills training
  (early on every pixel is past it, so there is no gradient and the output goes to zero); a cap needs
  a warm start or a cap that shrinks during training.
