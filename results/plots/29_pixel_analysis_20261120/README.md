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

## Disk maps (`band_maps/`)

Network's worst channel against the band BP used, beside BP's status, for two test dates. At
pixels BP solves at the tight band: inside 91.0% / 91.2%, within 1.5 half-widths 8.7% / 8.5%,
within 3 0.36% / 0.31%, beyond 3 0.00% / 0.01%. Slight misses are spread over the disk;
the clear misses sit in active-region and flare cores. The worst channel when outside is most
often 211 A, then 193 A. All unsolved pixels (62 / 86) are far outside.
