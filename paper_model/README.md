# DEMNet — paper models (Wide, 10.13M parameters)

Neural networks that turn 6 SDO/AIA EUV intensities into a **Differential
Emission Measure (DEM)**, i.e. how much plasma sits at each coronal
temperature. They are trained to imitate a classical DEM solver, so they
come in two families, one per reference solver:

| family | imitates | targets (training data) |
|---|---|---|
| **BP** | Basis Pursuit (linear-programming DEM inversion) | `bp_AIA_hofdeconv_full_DS` |
| **ENet** | ElasticNet-regularized DEM inversion | `enet_AIA_hofdeconv_full_DS` |

Each model is the **best checkpoint** (lowest validation loss) of its run.

## The models

These are the two models used in the paper (`FreqClassNonReLUWide`, see
`models.py`), copied from `dem-dl/student_package` on 2026-11-06. Weights are
identical to the original training-run checkpoints (BP: `model_best.pth`;
ENet: `epoch_125.pth`). Metrics are on the full test set (all pixels); DEM errors are in
units of 1e26 cm⁻⁵ per bin, resynthesis errors in DN/s. Lower is better.

| file | family | class | params | best epoch | DEM RMSE | DEM MAE | EM rel. err. | W1 (dex) | resynth RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `checkpoints/BP_5_Wide_10.1M.pth` | BP   | `FreqClassNonReLUWide` | 10.13M | 132/140 | 0.623 | 0.091 | 12.9% | 0.113 |  5.07 |
| `checkpoints/EN_2_Wide_10.1M.pth` | ENet | `FreqClassNonReLUWide` | 10.13M | 125/140 | 0.483 | 0.156 | 12.6% | 0.112 | 12.86 |

Exact values plus provenance (source training run, data, loss, bin range) are in
`models.json` and also inside every checkpoint.

BP and ENet errors are each measured against their own solver's targets, so
compare sizes *within* a family, not BP against ENet.

## Quick start

```python
import torch
from models import load_model

model, meta = load_model('checkpoints/BP_5_Wide_10.1M.pth', device='cuda')
print(meta['label'], meta['n_params'])

aia = torch.rand(1, 6, 256, 256, device='cuda') * 100   # [B, 6, H, W], DN/s
with torch.no_grad():
    dem = model.forward_regression(aia)[:, :18].clamp(min=0)  # [B, 18, H, W]
```

Or from the command line, with a preprocessed cube or raw FITS:

```bash
python predict.py --model checkpoints/BP_5_Wide_10.1M.pth --aia_npy cube.npy --out pred.npz --ci
python predict.py --model checkpoints/EN_2_Wide_10.1M.pth --fits_dir /path/to/20170910_154359 --out pred.npz
```

See the docstring at the top of `predict.py` for the output contents.

## Inputs — things that matter

The model applies `sqrt` itself, so pass raw intensities. Do not take the
square root or normalize them yourself. Input must be `[B, 6, H, W]` float:

* **Channel order**: 94, 131, 171, 193, 211, 335 Å.
* **Units**: DN/s, i.e. divided by exposure time **and** by the AIA
  degradation factor (`aiapy.calibrate.degradation`, correction table
  `aia_corr.csv`).
* **Preprocessing used for training**: level-1 FITS → `update_pointing` →
  **Hofmeister PSF deconvolution** → `register` (0.6″/px, 4096×4096) → divide
  by exposure × degradation → clip negatives to 0. `predict.py --fits_dir`
  runs this pipeline (same code as the training data). Deconvolution needs a
  GPU, `cupy`, and the 6 PSF files (`psf_aia_{wl}.fits`, 537 MB each, not
  copied here; see `--psf_dir`). Skipping deconvolution (`--no_deconvolve`)
  gives inputs the models never saw.
* **Any image size works.** Every layer is a 1×1 convolution, so each pixel is
  predicted independently from its own 6 values. Training used 256×256
  patches, but that only affected batching.

## Outputs

`model(aia)` returns `(regression, classification)`:

* **`regression`** `[B, 26, H, W]` is the DEM. Use only **bins 0–17**
  (logT = 5.5, 5.6, …, 7.2; see `RData.npz['logT']`). Bins 18–25 are left
  over from a planned XRT extension, are always zero in the targets, and
  should be dropped. Values are emission measure per 0.1-dex temperature bin
  in units of 1e26 cm⁻⁵.
  **Clamp it at 0** (`dem.clamp(min=0)`). The regression head has no output
  activation, so a small fraction of raw values come out negative. All
  paper metrics and figures use the clamped DEM. `predict.py` clamps for you;
  `model(...)` / `forward_regression(...)` return the raw values.
* **`classification`** `[B, 26, 128, H, W]` gives logits over 128 DEM-value
  bins for each temperature bin. Softmax them to get a per-pixel distribution
  and read uncertainties (5/50/95% quantiles) off it, as
  `predict.py --ci` does. The bins are sqrt-spaced on `[0, vmax]`, and
  **vmax differs by family: 6500 for BP, 1100 for ENet**. Always build the
  edges with `models.dem_bin_edges(meta)`, which reads the right value from
  the checkpoint.
  This output is 128× larger than the regression. When you only need the DEM,
  call `model.forward_regression(x)`, which skips it.
* **Resynthesis check**: `aia_pred = (R * 1e26) @ dem[:18]` with
  `R = RData.npz['R']` (shape `[6, 18]`) should reproduce the input AIA
  (DN/s). `predict.py` saves this as `aia_synth`.

## Files

| | |
|---|---|
| `models.py` | standalone model definitions (torch only), `load_model`, `dem_bin_edges` |
| `predict.py` | inference on a FITS folder or a `.npy` cube; DEM, optional quantiles, resynthesis |
| `models.json` | index of the checkpoints with provenance and test metrics |
| `RData.npz` | AIA temperature response `R` `[6, 18]` (scale by 1e26) and `logT` `[18]` |
| `aia_corr.csv` | AIA degradation correction table |
| `hofmeister/deconvolve_image.py` | PSF deconvolution (© Stefan Hofmeister; needs cupy + GPU) |

Dependencies: `torch`, `numpy`. For `--fits_dir`, add `sunpy`, `aiapy`,
`astropy`, and `cupy`.

## Training recipe (for reference)

Both models were trained for 140 epochs with multi-GPU DDP, using AdamW at LR
4e-5. The loss is
`RegressionClassificationResynthLoss`, which combines regression +
classification. From epoch 100 on, an AIA-resynthesis χ² term (weight 0.01) is
added. Both did all 140 epochs in one run.
