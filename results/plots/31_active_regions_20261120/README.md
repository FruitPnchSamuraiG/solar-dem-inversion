# 31: 40 active regions, band maps and DEM line plots of hard pixels (20 Nov 2026)

`experiments/active_regions.py`, final BP model (1.5M, method of multipliers, seed 2), the supervised
10.1M model and BP's own DEM, test dates only.

Regions: on each of the 153 test dates, pixels at 3x the Bright threshold or above were grouped into
regions (2,186 found); one region per date, the 15 with the brightest peak (flaring) then the 25
largest. A 192 x 192 box (DEM grid, ~230 arcsec) centred on each region's densest core. Two picks
(20 Dec 05:59 and 06:00) are the same region a few seconds apart.

Files:
- `ar_NN.png`: per region, 171 A, BP's status, our worst channel and the supervised model's against
  the band BP used; the picked pixels numbered.
- `by_region.pdf`: each region's overview page, then its pixels. `by_case.pdf`: the same 322 pixel
  pages grouped by case (120 worst_dem, 40 worst_shape, 30 worst_band, 31 brightest, 39 widened,
  22 unsolved, 40 typical). Each page: the DEM as a line (BP, ours, supervised), the six readings
  with 1-sigma error bars and BP's band (grey) with each DEM's resynthesis in DN/s, and the
  resynthesis minus the reading in sigma.
- `summary.json`: per region, status counts, share inside, the picked pixels.
- Click viewer: `python3 experiments/band_viewer.py output/active_regions/viewer_arNN.npz` (local
  files, not tracked), now with error bars and resynthesis.

Numbers (1.48M pixels in the 40 boxes):
- BP solves 1.36M at the tight band. Ours inside 71.9% of them (median region; 64-79%), within 1.5
  half-widths 26.5%; supervised inside 41.6% (median), worse than ours in all 40 regions.
- Typical Bright pixel: our DEM differs from BP's by 7% (median relative difference); the three
  curves nearly coincide.
- worst_dem pixels are flare cores (median 95x Bright), relative difference 0.52.
- worst_shape pixels are ordinary (median 2x Bright): all three DEMs usually fit every band, but ours
  adds or moves plasma where the readings cannot tell (often hot plasma above logT 7 that BP leaves
  out): several DEMs fit equally well and BP picks the sparsest.
- Flaring regions: the brightest core is often unsolved by BP and shows saturation (region 13, 16 Oct
  2015: a vertical bleed streak in 171 A; estimated raw counts above 16,000). See
  `../29_pixel_analysis_20261120/flare_pixels/saturation.json`.
- Unsolved pixels on the disk away from flare cores: typically 131 A reads about zero while every DEM
  that fits the other channels needs 10-15 DN/s there.
