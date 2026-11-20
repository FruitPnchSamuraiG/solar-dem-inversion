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
