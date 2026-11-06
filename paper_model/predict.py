"""
predict.py -- run a DEMNet checkpoint on one AIA observation.

Two ways to give it data:

  1) A folder with the 6 raw level-1 AIA FITS files of one timestamp (JSOC
     naming, *.{94,131,171,193,211,335}.image_lev1.fits). The script does the
     same preprocessing the training data had:
        update pointing -> Hofmeister PSF deconvolution -> register (0.6"/px)
        -> divide by exposure time x degradation factor  => DN/s
     Deconvolution needs a GPU + cupy and the PSF files (--psf_dir).

        python predict.py --model checkpoints/BP_5_Wide_10.1M.pth \
            --fits_dir /path/to/20170910_154359 --out out.npz [--ci]

  2) An already-preprocessed cube saved as .npy, shape [6, H, W] in DN/s,
     channel order 94, 131, 171, 193, 211, 335 (any H, W -- the model is
     pixel-wise):

        python predict.py --model checkpoints/EN_2_Wide_10.1M.pth \
            --aia_npy cube.npy --out out.npz --ci

Output .npz:
    dem       [18, H, W]  regression DEM clamped at 0, EM per 0.1-dex logT bin in units of
                          1e26 cm^-5 (logT = 5.5 ... 7.2)
    logT      [18]
    aia_in    [6, H, W]   the preprocessed input (DN/s)
    aia_synth [6, H, W]   resynthesized AIA = (R * 1e26) @ dem   (needs RData.npz)
    ci_low, ci_med, ci_high [18, H, W]   5/50/95% quantiles from the
                          classification head (only with --ci)

The model outputs 26 temperature bins; bins 18-25 were reserved for an XRT
extension and are always zero in the training targets, so they are dropped.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import load_model, dem_bin_edges  # noqa: E402

WAVELENGTHS = [94, 131, 171, 193, 211, 335]
N_AIA_BINS = 18


def preprocess_fits(fits_dir, corr_table, pointing_file, psf_dir, deconvolve=True):
    """Raw lev1 FITS folder -> [6, 4096, 4096] DN/s cube. Mirrors
    processIndAIAData in demdemo/src/utils.py, which built the training data."""
    import astropy.units as u
    import sunpy.map
    from aiapy.calibrate import register, update_pointing, degradation
    from aiapy.calibrate.util import get_pointing_table, get_correction_table

    files = []
    for wl in WAVELENGTHS:
        match = [f for f in os.listdir(fits_dir) if f.endswith('%d.image_lev1.fits' % wl)]
        if not match:
            raise FileNotFoundError('no *%d.image_lev1.fits in %s' % (wl, fits_dir))
        files.append(os.path.join(fits_dir, match[0]))
    maps = [sunpy.map.Map(f) for f in files]

    # scale factor = exposure time x degradation; applied at the very end
    if os.path.isfile(corr_table):
        ctable = get_correction_table(corr_table)
    else:
        ctable = get_correction_table('JSOC')
    scale = np.array([m.exposure_time.to_value(u.s) *
                      float(u.Quantity(degradation(wl * u.angstrom, m.date, correction_table=ctable)).value)
                      for m, wl in zip(maps, WAVELENGTHS)], dtype=np.float64).reshape(-1, 1, 1)

    t0 = maps[0].date
    if pointing_file and os.path.isfile(pointing_file):
        from astropy.table import QTable
        pt = QTable.read(pointing_file, format='ascii.ecsv')
        pt = pt[(pt['T_START'] >= t0 - 12 * u.h) & (pt['T_START'] <= t0 + 12 * u.h)]
    else:
        pt = get_pointing_table('JSOC', time_range=(t0 - 12 * u.h, t0 + 12 * u.h))
    maps = [update_pointing(m, pointing_table=pt) for m in maps]

    if deconvolve:
        from astropy.io import fits
        sys.path.insert(0, os.path.join(HERE, 'hofmeister'))
        from deconvolve_image import deconvolve_bid
        out = []
        for m, wl in zip(maps, WAVELENGTHS):
            print('deconvolving %d A' % wl)
            psf = fits.getdata(os.path.join(psf_dir, 'psf_aia_%d.fits' % wl)).astype(np.float32)
            d = deconvolve_bid(m.data.astype(np.float32), psf, use_gpu=True, large_psf=True)
            out.append(sunpy.map.Map(d, m.meta))
        maps = out
    maps = [register(m) for m in maps]

    # register can return slightly less than 4096^2 -- zero-pad back
    cube = []
    for m in maps:
        d = m.data
        if d.shape[0] != 4096:
            d = np.pad(d, (4096 - d.shape[0]) // 2)
        cube.append(d)
    cube = np.stack(cube) / scale
    return np.maximum(cube, 0).astype(np.float32)


def quantiles_from_logits(logits, edges, qs=(0.05, 0.5, 0.95)):
    """logits [B, K, H, W] over K DEM-value bins with K+1 edges -> one [B, H, W]
    tensor per quantile, linearly interpolated within the bin. Same as
    quantiles_from_pmf(..., use_bin_edges=True) in demdemo/src/utils.py."""
    pmf = torch.softmax(logits, dim=1)
    cdf = pmf.cumsum(dim=1)
    K = pmf.shape[1]
    lo_e, hi_e = edges[:-1], edges[1:]
    out = []
    for q in qs:
        idx = (cdf < q).sum(dim=1).clamp(0, K - 1)
        Fk = cdf.gather(1, idx.unsqueeze(1)).squeeze(1)
        Fkm1 = cdf.gather(1, (idx - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        Fkm1 = torch.where(idx == 0, torch.zeros_like(Fkm1), Fkm1)
        denom = Fk - Fkm1
        alpha = torch.where(denom < 1e-12, torch.zeros_like(denom), (q - Fkm1) / denom).clamp(0, 1)
        out.append(lo_e[idx] + alpha * (hi_e[idx] - lo_e[idx]))
    # if the pmf is peaked (>20%) in the zero bin, the lower bound is exactly 0
    zero_peak = (pmf.argmax(dim=1) == 0) & (pmf.max(dim=1)[0] > 0.2)
    out[0] = torch.where(zero_peak, torch.zeros_like(out[0]), out[0])
    return out


@torch.no_grad()
def run(model, meta, aia, ci=False, chunk_rows=None, device='cpu'):
    """aia: [6, H, W] DN/s numpy -> dict of [18, H, W] arrays. The model is
    pixel-wise, so the image is processed in horizontal strips."""
    C, H, W = aia.shape
    if chunk_rows is None:
        # the classification output is 26*128 channels -- keep strips small with --ci
        chunk_rows = max(1, (2**22 if ci else 2**24) // W)
    edges = dem_bin_edges(meta).to(device)
    res = {'dem': np.zeros((N_AIA_BINS, H, W), np.float32)}
    if ci:
        for k in ('ci_low', 'ci_med', 'ci_high'):
            res[k] = np.zeros((N_AIA_BINS, H, W), np.float32)

    for r0 in range(0, H, chunk_rows):
        x = torch.from_numpy(aia[None, :, r0:r0 + chunk_rows]).float().to(device)
        if not ci:
            reg = model.forward_regression(x)
            res['dem'][:, r0:r0 + chunk_rows] = reg[0, :N_AIA_BINS].clamp(min=0).cpu().numpy()
            continue
        reg, cls = model(x)
        res['dem'][:, r0:r0 + chunk_rows] = reg[0, :N_AIA_BINS].clamp(min=0).cpu().numpy()
        for t in range(N_AIA_BINS):
            lo, med, hi = quantiles_from_logits(cls[:, t], edges)
            res['ci_low'][t, r0:r0 + chunk_rows] = lo[0].cpu().numpy()
            res['ci_med'][t, r0:r0 + chunk_rows] = med[0].cpu().numpy()
            res['ci_high'][t, r0:r0 + chunk_rows] = hi[0].cpu().numpy()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='checkpoint from checkpoints/')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--fits_dir', help='folder with the 6 raw lev1 AIA FITS files')
    src.add_argument('--aia_npy', help='preprocessed [6, H, W] DN/s cube (.npy)')
    p.add_argument('--out', required=True, help='output .npz path')
    p.add_argument('--ci', action='store_true', help='also save 5/50/95%% quantiles from the classification head')
    p.add_argument('--chunk_rows', type=int, default=None, help='rows per forward pass (lower if you run out of memory)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--rdata', default=os.path.join(HERE, 'RData.npz'), help='AIA response matrix + logT grid')
    # FITS preprocessing options
    p.add_argument('--corr_table', default=os.path.join(HERE, 'aia_corr.csv'), help='AIA degradation correction table')
    p.add_argument('--pointing_file', default='', help='cached pointing table (.ecsv); queries JSOC if empty')
    p.add_argument('--psf_dir', default='/scratch/vp2435/workspace/dem/demdemo/optimScripts/hofmeister_psf',
                   help='folder with psf_aia_{wl}.fits for the Hofmeister deconvolution')
    p.add_argument('--no_deconvolve', action='store_true',
                   help='skip PSF deconvolution (NOT how the models were trained -- for quick tests only)')
    args = p.parse_args()

    model, meta = load_model(args.model, args.device)
    print('model: %s  (%s, %s params, DEM bin range 0-%g)' %
          (meta['label'], meta['model_name'], f"{meta['n_params']:,}", meta['dem_vmax']))

    if args.fits_dir:
        aia = preprocess_fits(args.fits_dir, args.corr_table, args.pointing_file,
                              args.psf_dir, deconvolve=not args.no_deconvolve)
    else:
        aia = np.maximum(np.load(args.aia_npy), 0).astype(np.float32)
        assert aia.ndim == 3 and aia.shape[0] == 6, 'expected [6, H, W], got %s' % (aia.shape,)

    res = run(model, meta, aia, ci=args.ci, chunk_rows=args.chunk_rows, device=args.device)
    res['aia_in'] = aia
    if os.path.isfile(args.rdata):
        rd = np.load(args.rdata)
        R = rd['R'] * 1e26                                    # [6, 18]
        res['logT'] = rd['logT']
        res['aia_synth'] = np.tensordot(R, res['dem'], axes=([1], [0])).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **res)
    print('saved %s: %s' % (args.out, ', '.join('%s%s' % (k, list(v.shape)) for k, v in res.items())))


if __name__ == '__main__':
    main()
