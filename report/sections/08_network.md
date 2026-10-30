## 8. What the network learned

The advisor asked what the final layer looks like, sorted by eigenvector. The output layer maps the last 336 hidden units to the 54 basis weights, so it can be read three ways: the weight matrix itself, the 54 outputs it produces over many pixels, and the DEMs those outputs make. We analysed all three for the adopted BP model, the ElasticNet model and, for contrast, the old BP model (μ = 1, log1p input, 40 epochs), on 611,947 validation pixels: up to 100,000 per brightness band and every flare core of the split (`experiments/final_layer.py`).

### 8.1 The output layer is low-rank, and the stiff band widened it

Table 8.1. Effective dimension of the output layer and of what it produces. The participation ratio of a spectrum is (Σλ)²/Σλ²; "for 90%" is the number of components that hold 90% of it.

| | BP adopted | ElasticNet | BP old (μ = 1) |
| --- | --- | --- | --- |
| Weight matrix (54 × 336): participation ratio of the singular values | 9.5 | 7.4 | 2.9 |
| Weight matrix: components for 90% / 99% of the energy | 22 / 40 | 19 / 40 | 11 / 24 |
| The 54 outputs over pixels: participation ratio of their correlation | 13.5 | 6.7 | 5.7 |
| Last hidden layer (336 units): components for 90% / 99% of the variance | 14 / 90 | 11 / 57 | 6 / 34 |
| DEM shapes (unit sum): components for 90% / 99% | 8 / 12 | 10 / 15 | 4 / 8 |
| The solver's DEM shapes on the same pixels | 6 / 13 | | 6 / 13 |

The old model's output layer was close to rank three, and its DEMs lived in a family of four shapes where the solver's need six. The stiff band tripled the layer's effective rank, and the adopted model's DEMs span as many shape directions as the solver's (8 against 6 for 90%, 12 against 13 for 99%). The 336 hidden units are far from saturated: 14 directions carry 90% of their variance. Width is not what limits the typical pixel, which agrees with section 7.1: extra width buys freedom away from the data, not accuracy on it.

![Figure 8.1. Left: singular values of the output weight matrix, relative to the largest. Centre: correlation of the adopted BP model's 54 outputs over validation pixels, sorted by the leading eigenvector ("0.1 6.9" is the 0.1 dex Gaussian centred at logT 6.9). Right: where the weight goes, by the centre temperature of the basis element, for the adopted model and for the solver's own fewest-weights decomposition of its DEMs, on the same 20,000 validation pixels. (results/plots/23_meeting_followup_20261024/fig5_final_layer.png)](figures/fig5_final_layer.png)

### 8.2 Sorted by eigenvector: a hot-plasma mode

Sorting the 54 outputs by the leading eigenvector of their correlation (Figure 8.1, centre) does not produce a smooth sweep in temperature. It produces one tight block: the spikes and 0.1 dex Gaussians at logT 6.6 to 7.2, joined by the coolest spikes at logT 5.6 and 5.7, whose ten members move together with a mean correlation of 0.86. The leading direction is how much hot plasma the pixel has. The same block leads at every brightness from 0.03 to 32 times the Bright cutoff. At flare cores the leading direction shifts to the hottest elements (logT 7.0 to 7.2) together with cool 0.2 dex Gaussians, the variation that remains once hot plasma is always present. The second eigenvector separates the hottest end (7.0 to 7.2) from the cool end (5.5 to 6.2). The weight matrix's own singular vectors give the same picture: its first left singular vector orders the outputs from the spike and 0.1 dex Gaussian at logT 7.0 downward through the hot elements.

### 8.3 How the network uses the basis

Table 8.2. Weight by basis shape, and active elements per pixel (above 1% of the pixel's largest weight), on 20,000 validation pixels stratified by brightness. The solver's DEMs are decomposed into their fewest-weights representation.

| | Spike | 0.1 dex Gaussian | 0.2 dex Gaussian | Active elements per pixel, mean (median) |
| --- | --- | --- | --- | --- |
| BP adopted | 18% | 21% | 61% | 6.6 (6) |
| Solver | 22% | 30% | 49% | 5.7 (6) |
| BP old (μ = 1), all pixels | 2% | 6% | 93% | 4.2 (4) |

The adopted model is a little less sparse than the solver: it leans on the wide Gaussians (61% of its weight against 49%) and uses about one more element per pixel. The old model built nearly every DEM from a few wide Gaussians: 93% of its weight sat on the 0.2 dex shape, at logT 6.3 to 6.4 and at 7.0, with nothing between 6.6 and 6.9, which is where the solver puts its hot plasma (section 6).

The weights also show that the 54-weight representation is not unique. The adopted model puts 25% of its weight on elements centred at logT 6.8 and 36% at 7.0, and essentially none at 6.9; the solver puts 15%, 36% and 15% there (Figure 8.1, right). Both describe the same hot bump around logT 6.9: two Gaussians either side of a bin add up to a curve centred on it. The loss scores only the light a DEM makes and its total weight, so the network is free to pick either decomposition, and the DEM error compares curves, not weights.

The ElasticNet model spreads its weight over all three shapes (27%, 36% and 36%) and about 20 elements per pixel, as the L2 term asks; its weight matrix and outputs are lower-rank than the adopted BP model's (Table 8.1).

Sources: `results/plots/23_meeting_followup_20261024/final_layer.txt` and `final_layer.json` (all numbers of Tables 8.1 and 8.2 except the 20,000-pixel shares, which come from `final_layer.npz`, arrays `bp__sol_x_network` and `sol_x_solver`, summed by shape and by centre logT); `experiments/final_layer.py`; `experiments/plot_meeting_followup.py` (Figure 8.1). The solver comparison of the ElasticNet model is omitted because the script decomposed BP solver DEMs for all three models.
