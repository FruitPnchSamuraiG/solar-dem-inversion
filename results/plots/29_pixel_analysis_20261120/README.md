# 29: Pixel-level look at the final BP model (20 Nov 2026)

Final BP model (1.5M, method of multipliers, seed 2), our staged test split (exact noise and BP's
per-pixel flag). Scripts: `experiments/flare_pixel_analysis.py`, `experiments/band_maps.py`,
`experiments/band_viewer.py` (click a pixel on a disk map to see its DEMs and resynthesis).

## Flare cores BP solves at the tight band (`flare_pixels/`)

Flare cores (32x the Bright threshold or above): 18,255 in the test split; BP solves 16,749 at the
tight band (549 at 3x, 151 at 5x, 806 unsolved). Each of the 16,749 is put in one type
(`summary.txt`; definitions in the script's docstring):

| type | pixels | share of our flare-core DEM error | supervised's |
|---|---|---|---|
| matches BP (relative error < 0.05) | 89.9% | 3.4% | 5.3% |
| inside every band, different DEM (tie / more emission) | 0.7% | 0.0% | 0.0% |
| smeared: same total, lower broader peak | 3.3% | 8.3% | 4.9% |
| too little emission | 0.3% | 30.9% | 41.9% |
| too much emission | 0.6% | 0.9% | 0.2% |
| peak shifted or extra peak | 1.1% | 53.7% | 47.4% |
| other, outside a band | 4.1% | 2.6% | 0.5% |

- The error is in very few pixels: the brightest 1% of these flare cores (168 pixels) hold 93% of
  it; 10 pixels where we miss most of BP's hot plasma (logT >= 6.8) hold 48%.
- The common failure is a smeared peak: BP puts the plasma in one sharp spike, we spread the same
  total over a lower, broader peak, and several channels miss their bands at once.
- At the brightest pixels (thousands of times the Bright threshold) BP's DEM reaches tens of
  thousands at logT 6.8 and at the top bin (7.2); ours and the supervised model's fall far short
  (emission 0.32x BP at the worst pixel) and miss every band by tens of half-widths.
- We are inside every band at 26% of these pixels, the supervised model at 4%; the median
  relative DEM error is ours 0.012, supervised 0.002.
- No two of these pixels have readings within 4.7% of each other in all six channels, so the
  look-alike question cannot be tested on flare cores this way.

## Saturation (`flare_pixels/saturation.json`, `experiments/saturation_check.py`)

Raw counts estimated as reading x the solver file's scale factor (exposure x degradation; before
deconvolution, so approximate); saturated if any channel reaches 15,000 (AIA saturates near 16,000).
- 9.2% of the 16,749 tight-band flare cores are saturated (193 A 8.2%, 171 A 3.4%, 211 A 2.3%).
- They hold 92% of our squared DEM error at flare cores and 99.3% of the supervised model's.
- Saturated share by type: too little emission 50%, peak shifted 46%, the brightest 1% 67%,
  matches BP 7.8%.
- Without them, flare-core DEM MSE is ours 2,025 vs supervised 91 (all: 24,187 vs 12,383), so
  saturation explains why a few pixels dominate the error, not our gap to the supervised model.
- Of all flare cores BP could not solve, 36% are saturated (tight band 9.2%, widened 7.7%).

## Disk maps (`band_maps/`)

Network's worst channel against the band BP used, beside BP's status, for all eight viewer dates
(`summary.json` two dates, `summary_6dates.json` the other six). Over the 30.6M pixels BP solves at
the tight band: inside 91.8% (91.0 to 93.0% per date), within 1.5 half-widths 7.8%, within 3 0.38%,
beyond 3 0.00%. At pixels BP had to widen to 3x, 98.8% are inside that band. Slight misses are spread
over the disk; the clear misses sit in active-region and flare cores. The worst channel when outside
is most often 211 A, then 193 A.

## Pixels BP cannot solve even at 5x (`unsolvable/`, `experiments/unsolvable_pixels.py`)

2,981 over the eight dates (33 to 1,932 per date). For each: the smallest band multiple that admits
any DEM, whether dropping one channel makes the tight band solvable, and its position.

- 90% (2,695) sit in the outermost 10 pixels of the image frame, far off the Sun, mostly on three
  dates (1,904 on 12 Nov 2015). Typically one channel holds a lone reading (211 A, sometimes 193 or
  335 A, hundreds of DN/s) while the others read about zero: image-edge artifacts, not plasma.
  Dropping 211 A alone makes 1,502 of them solvable.
- 286 lie on or just off the disk. There 131 A reads about zero (83 of 190 on disk are exactly
  zero) next to 600-700 DN/s in 171 and 193 A; no DEM gives that much 171/193 A without some
  131 A, and dropping 131 A alone makes most of them solvable. A zeroed or too-low 131 A reading
  (the deconvolution clamp) is the likely cause.
- They need a band of about 6x the tight one (median; p90 8x).

So BP's unsolvable pixels are almost all bad readings in one channel, not plasma no DEM can
explain. Cutouts (`cutout_*.png`) show six of them in every channel at full resolution.
