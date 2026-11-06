"""
DEMNet model definitions -- standalone, depends only on torch.

Every model in this package is the same architecture (FreqClassNonReLU) at a
different width. It is purely pixel-wise: all layers are 1x1 convolutions, so
each pixel's DEM is predicted from that pixel's 6 AIA intensities alone, and
any input H x W works (we used 256 x 256 patches).

    AIA [B, 6, H, W]  (DN/s, exposure + degradation corrected)
      -> sqrt(clamp(x, 0))                     (done inside forward)
      -> Fourier positional encoding           6 -> 6*(2*12+1) = 150 channels
      -> backbone: 4 x (1x1 conv, SiLU, Dropout 0.2), widths W0, W1, W1, W1
      -> regression head      -> [B, 26, H, W]        DEM per logT bin
      -> classification head  -> [B, 26, 128, H, W]   logits over 128 DEM-value bins

State-dict keys match the training code (src/model.py in demdemo), so the
checkpoints in checkpoints/ load directly.

Quick use:
    from models import load_model
    model, meta = load_model('checkpoints/BP_3_W224_2.0M.pth')
    reg, cls = model(aia)          # aia: float tensor [B, 6, H, W]
"""

import torch
import torch.nn as nn


def posEncode(x, numFreq=16):
    """[B, C, H, W] -> [B, C*(2*numFreq+1), H, W]: sin/cos of each channel at
    numFreq frequencies 2*pi*2**(i/2), i = -4 .. numFreq-5, plus the raw value."""
    B, C, H, W = x.shape
    freqs = torch.tensor([6.28318530718*(2**(i/2.0)) for i in range(-4, numFreq-4)],
                         device=x.device, dtype=x.dtype)
    angle = x.unsqueeze(2) * freqs.view(1, 1, numFreq, 1, 1)             # [B, C, numFreq, H, W]
    periodic = torch.stack((torch.sin(angle), torch.cos(angle)), dim=3)  # [B, C, numFreq, 2, H, W]
    periodic = periodic.reshape(B, C*numFreq*2, H, W)
    return torch.cat((periodic, x), dim=1)


class DEMNet(nn.Module):
    """FreqClassNonReLU with configurable hidden widths (W0, W1)."""
    W0, W1 = 128, 256

    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=128, W0=None, W1=None):
        super().__init__()
        W0 = W0 or self.W0
        W1 = W1 or self.W1
        self.name = type(self).__name__
        self.nIn, self.nOut, self.nFreq, self.n_bins = nIn, nOut, nFreq, n_bins

        nChannels = nIn*(nFreq*2+1)
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, W0, 1), nn.SiLU(), nn.Dropout(p=0.2),
            nn.Conv2d(W0, W1, 1), nn.SiLU(), nn.Dropout(p=0.2),
            nn.Conv2d(W1, W1, 1), nn.SiLU(), nn.Dropout(p=0.2),
            nn.Conv2d(W1, W1, 1), nn.SiLU(), nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(nn.Conv2d(W1, nOut, 1))
        self.classification_head = nn.Conv2d(W1, nOut * n_bins, 1)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification

    def forward_regression(self, x):
        """Regression output only. Skips the classification head, whose output
        is 128x larger -- use this when you only need the DEM itself."""
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        return self.regression_head(self.backbone(posEncode(x, self.nFreq)))


# Named sizes, smallest to largest. Names match the `model_name` stored in
# each checkpoint. params = 163840*k^2 + 878720*k + 3354, k = W0/128.
class FreqClassNonReLUW72(DEMNet):
    """0.55M params (549,474)"""
    W0, W1 = 72, 144


class FreqClassNonReLU(DEMNet):
    """1.05M params (1,045,914) -- the original base size"""
    W0, W1 = 128, 256


class FreqClassNonReLUW224(DEMNet):
    """2.04M params (2,042,874)"""
    W0, W1 = 224, 448


class FreqClassNonReLUW400(DEMNet):
    """4.35M params (4,349,354)"""
    W0, W1 = 400, 800


class FreqClassNonReLUWide(DEMNet):
    """10.13M params (10,130,154) -- the paper model size"""
    W0, W1 = 720, 1440


MODELS = {cls.__name__: cls for cls in
          [FreqClassNonReLUW72, FreqClassNonReLU, FreqClassNonReLUW224,
           FreqClassNonReLUW400, FreqClassNonReLUWide]}


def load_model(path, device='cpu'):
    """Load a checkpoint from this package. Returns (model in eval mode, meta dict)."""
    pack = torch.load(path, map_location=device, weights_only=False)
    model = MODELS[pack['model_name']](n_bins=pack['n_bins'])
    model.load_state_dict(pack['model_state_dict'])
    model.to(device).eval()
    meta = {k: v for k, v in pack.items() if k != 'model_state_dict'}
    return model, meta


def dem_bin_edges(meta):
    """Edges (n_bins+1) of the DEM-value bins the classification head predicts
    over, for one temperature bin. Sqrt spacing on [vmin, vmax]; note vmax
    differs between BP (6500) and ENet (1100) models."""
    vmin, vmax, n = meta['dem_vmin'], meta['dem_vmax'], meta['n_bins']
    edges = torch.linspace(vmin ** 0.5, vmax ** 0.5, n + 1) ** 2
    edges[0], edges[-1] = 0.0, vmax
    return edges
