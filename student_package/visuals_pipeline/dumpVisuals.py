"""
Dump visualizations for one DEM npz file
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
import scipy.ndimage
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

# hardcoded region coordinates by timestamp
# format: {timestamp: {region_type: (center_r, center_c), ...}}
# region types: 'hot', 'quiet', 'flaring', 'coronal_hole'
# regions will be 128x128 centered on the given pixel
HARDCODED_REGIONS = {
    '20170910_1548': {
        'hot':  (2380, 3540),
        'quiet': (1730, 2108),
        'flaring': (1813, 3635),
        'coronal_hole': (2775, 1372),
        'roi': (1813, 3635),  # region of interest (same as flaring)
    },
    '20110906_2217': {
        'hot':  (2695, 610),
        'quiet': (2805, 2058),
        'flaring': (2276, 2526),
        'coronal_hole': (2205, 1361),
        'roi': (2276, 2526),
    },
    '20140910_1731': {
        'hot':  (1800, 489),
        'quiet': (2824, 2500),
        'flaring': (2257, 1868),
        'coronal_hole': (756, 2042),
        'roi': (2257, 1868),
    },
    '20120603_0000': {
        'hot':  (1342, 3420),
        'quiet': (2344, 2009),
        'flaring': (2524, 739),
        'coronal_hole': (2569, 1787),
        'roi': (2569, 1787),
    },
    '20170908_1930': {
        'hot':  (2039, 2342),
        'quiet': (1590, 1457),
        'flaring': (1761, 3550),
        'coronal_hole': (2780, 1190),
        'roi': (1761, 3550),
    },
}

REGION_SIZE = 128  # size of region box

def extract_timestamp(dem_path):
    """extract timestamp from dem path (e.g., 'dem_20151005_1106.npz' -> '20151005_1106')"""
    basename = os.path.basename(dem_path)
    match = re.search(r'(\d{8}_\d{4})', basename)
    return match.group(1) if match else None

def center_to_bounds(center, region_size=REGION_SIZE):
    """convert center pixel (r, c) to bounds (r0, c0, r1, c1)"""
    r, c = center
    half = region_size // 2
    return (r - half, c - half, r + half, c + half)

def get_regions_for_timestamp(timestamp):
    """get hardcoded regions for a timestamp, converting centers to bounds"""
    centers = HARDCODED_REGIONS.get(timestamp, None)
    if centers is None:
        return None
    return {k: center_to_bounds(v) for k, v in centers.items()}

def parse_args():
    parser = argparse.ArgumentParser(description= "Create visualizations from DEMs")
    parser.add_argument("--dem_path", type=str, help="Path to the npz dem")
    parser.add_argument("--target", type=str, default=None,
                        help="Output directory for the visualizations (will be created if not exists)")
    parser.add_argument("--dem", action='store_true', default=False,
                       help="Create visualizations for the DEMs")
    parser.add_argument("--aia_resynth", action='store_true', default=False,
                          help="Create visualizations for the resynthesis AIA images")
    parser.add_argument("--dem_jpdfs", action='store_true', default=False,
                          help="Create visualizations for the DEM joint probability density functions")
    parser.add_argument("--dem_comparison_path", type=str, default=None,
                        help="Path to the npz dem for comparison (if provided, will create comparison visualizations)")
    parser.add_argument("--uncertainty", action='store_true', default=False,
                        help="Create uncertainty visualizations (requires ClassificationMaxProb and ClassificationEntropy in npz)")
    parser.add_argument("--dem_pixels", action='store_true', default=False,
                        help="Create DEM plots for representative pixels (50 hot, 50 quiet)")
    parser.add_argument("--n_hot", type=int, default=50,
                        help="Number of hot pixels to sample (default 50)")
    parser.add_argument("--n_quiet", type=int, default=50,
                        help="Number of quiet pixels to sample (default 50)")
    parser.add_argument("--hot_frac", type=float, default=0.005,
                        help="Top fraction of 193 for hot region selection")
    parser.add_argument("--quiet_lo", type=float, default=0.40,
                        help="Lower quantile for quiet region (default 0.40)")
    parser.add_argument("--quiet_hi", type=float, default=0.60,
                        help="Upper quantile for quiet region (default 0.60)")
    parser.add_argument("--disk_shrink", type=float, default=0.98,
                        help="Shrink factor for on-disk circle mask")
    parser.add_argument("--disk_mode", type=str, default='circle', choices=['circle', 'box'],
                        help="On-disk masking mode: circle (default) or box")
    parser.add_argument("--regions_jpdfs", action='store_true', default=False,
                        help="Create AIA vs AIA resynth JPDFs for each region (hot, quiet, flaring, coronal_hole)")
    parser.add_argument("--roi_dems", action='store_true', default=False,
                        help="Create DEM visualizations for the ROI region (flaring region by default)")

    return parser.parse_args()

def nnInterpNaN(X):
    """Given HxWxC image X, nearest neighbor interpolate all pixels with a nan
    in any channel. Not very efficient"""
    M = np.any(np.isnan(X),axis=0)

    distanceIndMulti = scipy.ndimage.distance_transform_edt(M, return_distances=False, return_indices=True)
    distanceInd = np.ravel_multi_index(distanceIndMulti, M.shape)

    X2 = X.copy()
    for c in range(X.shape[0]):
        Xc = X[c,:,:]
        X2[c,:,:] = Xc.ravel()[distanceInd]
    return X2 

def dumpDEMJointPDF(target, dem_truth, dem_pred, transformed=None):
    """
    Dump the joint PDF of the DEM predicted vs ground truth
    transformed: either name of transformation, or array of strings(e.g. ["sqrt", None])
    """

    if not os.path.exists(target):
        os.makedirs(target)

    # untransform the DEM cubes if necessary
    if transformed is not None:
        if transformed[0] == 'sqrt':
            dem_cube_truth = dem_cube_truth**2
        if transformed[1] == 'sqrt':
            dem_cube = dem_cube**2
    
    for i in range(dem_pred.shape[0]):
        gt = dem_truth[i].ravel()
        pd = dem_pred[i].ravel()

        # avoid zeros/nans before counting
        gt = np.maximum(gt, 1e-8)
        pd = np.maximum(pd, 1e-8)

        # mask 
        tr_mask = 1e-1
        mask = (gt > tr_mask) & (pd > tr_mask) & ~np.isnan(gt) & ~np.isnan(pd)
        gsize = 200
        # if no elements with mask, just plot everything
        if np.count_nonzero(gt[mask]) < gsize or np.count_nonzero(pd[mask]) < gsize:
            print("Warning: Not enough data points to create a meaningful joint PDF, using all data points.")
            mask = ~np.isnan(gt) & ~np.isnan(pd)
            tr_mask = 1e-8
            gsize = 40

        x = np.log10(gt[mask])
        y = np.log10(pd[mask])

        # R2
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            # if one of the stds is 0, just put 0
            r2 = 0
        else:
            r2 = np.corrcoef(x, y)[0,1]**2

        # count zeros: ground <=tr_mask region
        zero_mask = gt <= tr_mask
        tn = np.sum(zero_mask & (pd <= tr_mask))
        fp = np.sum(zero_mask & (pd > tr_mask))
        fn = np.sum((gt > tr_mask) & (pd <= tr_mask))
        
        # compute rates
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_negative_rate = fn / (fn + tn) if (fn + tn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        plt.figure(figsize=(4,4))
        plt.hexbin(gt[mask], pd[mask], gridsize=gsize, bins='log', xscale='log', yscale='log', cmap='turbo', extent=[np.log10(tr_mask), 2, np.log10(tr_mask), 2])
        plt.plot([tr_mask, 100], [tr_mask, 100], 'k--')

        # put and R^2 in the title
        plt.title(r"$R^2 = %.2f$" % r2)
        plt.xlabel(r"$DEM_{true}$")
        plt.ylabel(r"$DEM_{model}$")
        plt.axis('square')

        txt = f"tn={true_negative_rate*100:.2f}, fp={false_positive_rate*100:.2f}\nfn={false_negative_rate*100:.2f}"

        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
                 va='top', ha='left', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

        plt.savefig(os.path.join(target, f"dem_{i}_jpdf.png"), dpi=300, bbox_inches='tight')
        plt.close()
        

def dumpDEM(target, dem_cubes, transformed):
    dem_vmax = 30
    dem_cubes = np.array(dem_cubes)
    # threshold to remove numerical noise artifacts
    dem_cubes[dem_cubes < 0.01] = 0

    if not os.path.exists(target):
        os.makedirs(target)

    if transformed == 'sqrt':
        for i in range(dem_cubes.shape[0]):
            plt.imsave(f"{target}/dem_{i}.png", dem_cubes[i, :, :], vmin=0, vmax=dem_vmax, cmap='turbo')
    else:
        # save dem cubes
        for i in range(dem_cubes.shape[0]):
            plt.imsave(f"{target}/dem_{i}.png", np.maximum(0, dem_cubes[ i, :, :])**0.5, vmin=0, vmax=dem_vmax, cmap='turbo')

def dumpROIDEMs(target, dem_cube, aia_cube, dem_path=None):
    """create dem visualizations for roi region"""
    timestamp = extract_timestamp(dem_path) if dem_path else None
    if timestamp is None:
        print("warning: could not extract timestamp from dem_path, skipping roi dems")
        return
    
    regions = get_regions_for_timestamp(timestamp)
    if regions is None or 'roi' not in regions:
        print(f"warning: no roi region defined for timestamp {timestamp}, skipping")
        return
    
    r0, c0, r1, c1 = regions['roi']
    roi_dir = os.path.join(target, "roi_pixels")
    aia_dir = os.path.join(target, "roi_jpdfs")
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(aia_dir, exist_ok=True)
    
    print(f"creating roi dem visualizations for region ({r0}, {c0}) to ({r1}, {c1})")
    
    # extract roi region from dem cube
    dem_roi = dem_cube[:, r0:r1, c0:c1]
    # threshold to remove numerical noise artifacts
    dem_roi[dem_roi < 0.01] = 0
    
    # save each dem bin for the roi region
    dem_vmax = 30
    for i in range(dem_roi.shape[0]):
        filepath = os.path.join(roi_dir, f"dem_{i:03d}.png")
        plt.imsave(filepath, np.maximum(0, dem_roi[i, :, :])**0.5, vmin=0, vmax=dem_vmax, cmap='turbo')
    
    # save aia 193 region image
    I193 = aia_cube[3, :, :]  # 193 is channel 3
    I193_roi = I193[r0:r1, c0:c1]
    I193_full = I193
    vmin, vmax = np.nanpercentile(I193_full, [20, 99.9999])
    
    filepath = os.path.join(aia_dir, "aia_193_region.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.maximum(0, I193_roi)**0.5, cmap='sdoaia193',
              vmin=vmin**0.5, vmax=vmax**0.5, origin='upper')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # create full disk aia image with only roi region marked
    filepath_full = os.path.join(target, "aia193_roi.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.maximum(0, I193_full)**0.5, cmap='sdoaia193',
              vmin=vmin**0.5, vmax=vmax**0.5, origin='upper')
    
    # mark roi region with yellow box
    rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                         fill=False, edgecolor='yellow', linewidth=3, label='Region of Interest')
    ax.add_patch(rect)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath_full, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"saved {dem_roi.shape[0]} dem bins, aia 193 region, and full disk aia with roi")


def dumpDiagnostics(target, DEMCube, logT):
    """
    Plot some diagnostic information about the DEM cube, including:
    - mean of the logT distribution
    - std of the logT distribution
    """
    nLogT = DEMCube.shape[0]
    logTRange = np.max(logT) - np.min(logT)

    assert(nLogT == logT.size)
    # convert to a distribution, but some might be all 0, so add 1e-8 to stabilize
    DEMDistr = DEMCube / (1e-12 + np.sum(DEMCube, axis=0, keepdims=True))
    meanLogT = np.sum(logT.reshape(-1,1,1) * DEMDistr, axis=0, keepdims=True)
    stdLogT = np.sum((logT.reshape(-1,1,1) - meanLogT)**2 * DEMDistr, axis=0, keepdims=True)**0.5

    plt.imsave(os.path.join(target, "mean_logt.png"), meanLogT[0,:,:], vmin=np.min(logT), vmax=np.max(logT), cmap='inferno')
    plt.imsave(os.path.join(target, "std_logt.png"), stdLogT[0,:,:], vmin=0, vmax=0.5*logTRange, cmap='inferno')
    
    # save colorbar for meanLogT
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = matplotlib.colors.Normalize(vmin=np.min(logT), vmax=np.max(logT))
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap='inferno', norm=norm, orientation='horizontal')
    cbar.set_label(r'$\langle \log_{10} T \rangle$ [K]', fontsize=12)
    plt.savefig(os.path.join(target, "mean_logt_colorbar.png"), dpi=150, bbox_inches='tight')
    plt.close()

def dumpSynthesis(target, name, AIACube, DEMCube, R, wavelengths):
    """
    Resynthesize the AIA Data, compared to the AIA data

    target: where to dump the data
    name: identifier to append in the pngs
    AIACube: C x H x W
    DEMCube: nBins x H x W
    R: response function (C x nBins)
    wavelengths: the corresponding wavelength (in A), for the right colormap
    """
    resynth = np.zeros(AIACube.shape)
    # Vectorized computation to speed up resynthesis
    # AIACube: (C, H, W), DEMCube: (nBins, H, W), R: (C, nBins)
    # Want: resynth: (C, H, W) where resynth[:,i,j] = R @ DEMCube[:,i,j]
    resynth = np.tensordot(R, DEMCube, axes=([1], [0]))  # shape: (C, H, W)

    for i in range(AIACube.shape[0]):
        cm = plt.get_cmap("sdoaia%d" % wavelengths[i])
        
        # compute data-dependent vmin and vmax after sqrt transform
        vmin_percentile = 20
        vmax_percentile = 99.9999
        
        # apply sqrt transform first, then compute percentiles
        aia_data_sqrt = np.maximum(0, AIACube[i, :, :])**0.5
        aia_valid = aia_data_sqrt[~np.isnan(aia_data_sqrt)]
        if len(aia_valid) > 0:
            vmin_aia = np.percentile(aia_valid, vmin_percentile)
            vmax_aia = np.percentile(aia_valid, vmax_percentile)
        else:
            vmin_aia = 0
            vmax_aia = 1
        
        resynth_data_sqrt = np.maximum(0, resynth[i, :, :])**0.5
        resynth_valid = resynth_data_sqrt[~np.isnan(resynth_data_sqrt)]
        if len(resynth_valid) > 0:
            vmin_resynth = np.percentile(resynth_valid, vmin_percentile)
            vmax_resynth = np.percentile(resynth_valid, vmax_percentile)
        else:
            vmin_resynth = 0
            vmax_resynth = 1
        
        plt.imsave(os.path.join(target, "aia_%d_%s.png" % (i, name)), 
                    resynth_data_sqrt,
                    vmin=vmin_aia, vmax=vmax_aia, cmap=cm)
        plt.imsave(os.path.join(target, "aia_%d.png" % i), 
                    aia_data_sqrt,
                    vmin=vmin_aia, vmax=vmax_aia, cmap=cm)
        
        #plot a difference map
        vrange = np.nanmax(AIACube[i,:,:])*0.1
        diff = resynth[i,:,:] - AIACube[i,:,:]
        plt.imsave(os.path.join(target, "aia_%d_%s_diff.png" % (i, name)), 
                    diff, vmin=-vrange, vmax=vrange, cmap='bwr')


        x = np.maximum(AIACube[i,:,:].reshape(-1), 0.5)
        y = np.maximum(resynth[i,:,:].reshape(-1), 0.5)
       
        k = ~np.isnan(x) 
        clip = np.maximum(np.nanmax(np.log10(x[k])), np.nanmax(np.log10(y[k])))

        plt.figure(figsize=(4,4))
        plt.hexbin(x[k], y[k], xscale='log', yscale='log', extent=[1,clip,1,clip], bins='log', gridsize=200)
        plt.plot([10**1, 10**clip],[10**1, 10**clip], c='k', linestyle='--')
        # put and R^2 in the title
        plt.title(r"$R^2 = %.2f$" % np.corrcoef(np.log10(x[k]), np.log10(y[k]))[0,1]**2)
        plt.xlabel(r"$I_{AIA}$")
        plt.ylabel(r"$I_{resynth}$")
        plt.axis('square')
        plt.savefig(os.path.join(target, "aia_%d_%s_jpdf.png" % (i, name)), dpi=300)
        plt.close()


def on_disk_mask(H, W, shrink=0.98, mode='circle', edge_margin_px=128):
    """build on-disk mask (pixels only) using circle or box"""
    if mode == 'box':
        m = np.zeros((H, W), bool)
        m[edge_margin_px:H-edge_margin_px, edge_margin_px:W-edge_margin_px] = True
        return m
    # circle centered at image center
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r0 = 0.5 * min(H, W) * shrink
    return (xx - cx)**2 + (yy - cy)**2 <= r0**2

def select_hot_region_128(I193, region_size=128, top_frac=0.005):
    """select one 128x128 hot region with highest mean intensity in top percentile"""
    H, W = I193.shape
    x193 = np.log1p(np.clip(I193, 0, None))
    thr193 = np.nanquantile(x193, 1.0 - top_frac)
    
    # find regions with hot pixels
    best_mean = -np.inf
    best_region = None
    
    for r in range(0, H - region_size, region_size // 2):
        for c in range(0, W - region_size, region_size // 2):
            region = x193[r:r+region_size, c:c+region_size]
            hot_mask = region >= thr193
            if hot_mask.sum() > 10:  # require at least 10 hot pixels
                mean_intensity = np.nanmean(region[hot_mask])
                if mean_intensity > best_mean:
                    best_mean = mean_intensity
                    best_region = (r, c, r+region_size, c+region_size)
    
    return best_region

def select_quiet_region_128(I193, ondisk_mask, region_size=128, q_lo=0.40, q_hi=0.60):
    """select one 128x128 quiet region with most pixels in middle intensity band"""
    H, W = I193.shape
    x193 = np.log1p(np.clip(I193, 0, None))
    vals = x193[ondisk_mask]
    lo = np.nanquantile(vals, q_lo)
    hi = np.nanquantile(vals, q_hi)
    
    # find region with most quiet pixels
    best_count = 0
    best_region = None
    
    for r in range(0, H - region_size, region_size // 2):
        for c in range(0, W - region_size, region_size // 2):
            region = x193[r:r+region_size, c:c+region_size]
            region_ondisk = ondisk_mask[r:r+region_size, c:c+region_size]
            quiet_mask = (region >= lo) & (region <= hi) & region_ondisk
            count = quiet_mask.sum()
            if count > best_count:
                best_count = count
                best_region = (r, c, r+region_size, c+region_size)
    
    return best_region

def select_n_pixels_from_region(Iref, region_bounds, n_pixels, mask=None):
    """select n pixels uniformly spaced across intensity range within region"""
    r0, c0, r1, c1 = region_bounds
    region = Iref[r0:r1, c0:c1]

    if mask is not None:
        region_mask = mask[r0:r1, c0:c1]
        r_idx, c_idx = np.where(region_mask)
    else:
        r_idx, c_idx = np.where(~np.isnan(region))

    n_valid = r_idx.size
    if n_valid < n_pixels:
        print(f"warning: region has only {n_valid} valid pixels, sampling all")
        n_pixels = n_valid

    vals = region[r_idx, c_idx]
    order = np.argsort(vals)

    # select n pixels uniformly spaced across sorted intensity
    indices = np.linspace(0, n_valid - 1, n_pixels, dtype=int)
    
    pixels = []
    for idx in indices:
        k = order[idx]
        pixels.append((int(r0 + r_idx[k]), int(c0 + c_idx[k])))
    
    return pixels

def plot_aia_region_with_pixel(I193, pixel, region_bounds, filepath, marker_size=10):
    """plot aia 193 region (128x128) with single pixel marked by square"""
    r0, c0, r1, c1 = region_bounds
    I193_region = I193[r0:r1, c0:c1]
    
    vmin, vmax = np.nanpercentile(I193, [20, 99.9999])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.maximum(0, I193_region)**0.5, cmap='sdoaia193', 
              vmin=vmin**0.5, vmax=vmax**0.5, origin='upper')
    
    # mark pixel with simple square
    r, c = pixel
    r_local, c_local = r - r0, c - c0
    rect = plt.Rectangle((c_local - marker_size/2, r_local - marker_size/2), 
                         marker_size, marker_size, 
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_aia_full_with_regions(I193, regions_dict, filepath):
    """plot full aia 193 image with all regions marked"""
    vmin, vmax = np.nanpercentile(I193, [20, 99.9999])
    
    # colors for each region type
    region_colors = {
        'hot': 'red',
        'quiet': 'cyan', 
        'flaring': 'orange',
        'coronal_hole': 'magenta',
        'roi': 'yellow',
    }
    region_labels = {
        'hot': 'Hot Region',
        'quiet': 'Quiet Region',
        'flaring': 'Flaring Region',
        'coronal_hole': 'Coronal Hole',
        'roi': 'Region of Interest',
    }
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.maximum(0, I193)**0.5, cmap='sdoaia193', 
              vmin=vmin**0.5, vmax=vmax**0.5, origin='upper')
    
    for region_type, bounds in regions_dict.items():
        if bounds is None:
            continue
        r0, c0, r1, c1 = bounds
        color = region_colors.get(region_type, 'white')
        label = region_labels.get(region_type, region_type)
        rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, 
                             fill=False, edgecolor=color, linewidth=3, label=label)
        ax.add_patch(rect)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_region_aia_jpdf(aia_region, resynth_region, wavelength, filepath):
    """plot jpdf of aia original vs resynth for a region with free axis limits"""
    x = np.maximum(aia_region.ravel(), 0.5)
    y = np.maximum(resynth_region.ravel(), 0.5)
    
    k = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(k) < 10:
        print(f"warning: not enough valid pixels for jpdf at {wavelength}")
        return
    
    # compute axis limits from data
    x_valid, y_valid = np.log10(x[k]), np.log10(y[k])
    vmin = min(np.nanmin(x_valid), np.nanmin(y_valid))
    vmax = max(np.nanmax(x_valid), np.nanmax(y_valid))
    # add small margin
    margin = 0.05 * (vmax - vmin)
    vmin, vmax = vmin - margin, vmax + margin
    
    r2 = np.corrcoef(x_valid, y_valid)[0, 1]**2
    
    plt.figure(figsize=(4, 4))
    plt.hexbin(x[k], y[k], xscale='log', yscale='log', 
               extent=[vmin, vmax, vmin, vmax], bins='log', gridsize=100)
    plt.plot([10**vmin, 10**vmax], [10**vmin, 10**vmax], c='k', linestyle='--')
    plt.title(f"AIA {wavelength}Å  " + r"$R^2 = %.3f$" % r2)
    plt.xlabel(r"$I_{AIA}$")
    plt.ylabel(r"$I_{resynth}$")
    plt.axis('square')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_region_aia193(I193_region, I193_full, filepath):
    """save aia 193 image of a region with same color scale as full image"""
    vmin, vmax = np.nanpercentile(I193_full, [20, 99.9999])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.maximum(0, I193_region)**0.5, cmap='sdoaia193',
              vmin=vmin**0.5, vmax=vmax**0.5, origin='upper')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def dumpRegionJPDFs(target, dem_cube, aia_cube, dem_path=None, 
                    hot_frac=0.005, disk_shrink=0.98, disk_mode='circle'):
    """create aia vs resynth jpdfs for each region"""
    H, W = aia_cube.shape[1], aia_cube.shape[2]
    
    # load response function
    RData = np.load("/scratch/vp2435/workspace/dem/demdemo/RData.npz")
    R = RData["R"] * 1e26
    R = R.astype(np.float64)
    wavelengths = [94, 131, 171, 193, 211, 335]
    
    # handle xrt additional temperature bins
    if dem_cube.shape[0] > 18:
        logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
        R = np.hstack([R, np.zeros((R.shape[0], logTExpand.size))])
    
    # compute full resynth
    resynth = np.tensordot(R, dem_cube, axes=([1], [0]))
    
    # build on-disk mask and get regions
    ondisk = on_disk_mask(H, W, shrink=disk_shrink, mode=disk_mode)
    I193 = aia_cube[3, :, :]
    
    # check for hardcoded regions
    timestamp = extract_timestamp(dem_path) if dem_path else None
    hardcoded = get_regions_for_timestamp(timestamp) if timestamp else None
    
    if hardcoded:
        print(f"using hardcoded regions for timestamp {timestamp}")
        regions = hardcoded
    else:
        print("no hardcoded regions found, auto-detecting hot and quiet regions")
        regions = {
            'hot': select_hot_region_128(I193, region_size=128, top_frac=hot_frac),
            'quiet': select_quiet_region_128(I193, ondisk, region_size=128, q_lo=0.05, q_hi=0.10),
        }
    
    # process each region
    for region_type, bounds in regions.items():
        if bounds is None:
            continue
        
        r0, c0, r1, c1 = bounds
        region_dir = os.path.join(target, f"{region_type}_jpdfs")
        os.makedirs(region_dir, exist_ok=True)
        
        print(f"creating jpdfs for {region_type} region: ({r0}, {c0}) to ({r1}, {c1})")
        
        # save aia 193 region image (use full image for consistent color scale)
        plot_region_aia193(I193[r0:r1, c0:c1], I193, os.path.join(region_dir, "aia_193_region.png"))
        
        for i, wl in enumerate(wavelengths):
            aia_region = aia_cube[i, r0:r1, c0:c1]
            resynth_region = resynth[i, r0:r1, c0:c1]
            plot_region_aia_jpdf(aia_region, resynth_region, wl,
                                os.path.join(region_dir, f"aia_{wl}_jpdf.png"))
    
    # save overview image with regions
    regions_found = {k: v for k, v in regions.items() if v is not None}
    if regions_found:
        plot_aia_full_with_regions(I193, regions_found,
                                  os.path.join(target, "aia193_regions_jpdfs.png"))
    
    print(f"region jpdf visualizations saved to {target}")

def plot_single_pixel_dem(dem_cube, logT, pixel, filepath, aia_cube=None, R=None, ci_low=None, ci_high=None):
    """plot three-panel visualization: aia observations, aia resynthesis, and dem profile for a single pixel"""
    r, c = pixel
    aia_channels = ['94', '131', '171', '193', '211', '335']
    
    fig, subplots = plt.subplots(2, 1, figsize=(5, 10))
    
    # top panel: aia observations and resynthesis
    if aia_cube is not None and R is not None:
        aia_pixel = aia_cube[:, r, c]
        # compute resynth: R @ dem_pixel
        dem_pixel = dem_cube[:, r, c]
        aia_resynth = R @ dem_pixel
        
        x_positions = range(6)
        subplots[0].plot(x_positions, aia_pixel, 'o-', lw=2, label='AIA', markersize=6)
        subplots[0].plot(x_positions, aia_resynth, 's--', lw=2, label='AIA Resynth', markersize=6)
        subplots[0].set_xticks(x_positions)
        subplots[0].set_xticklabels(aia_channels)
        subplots[0].set_xlabel('AIA Channel [Å]')
        subplots[0].set_ylabel('Intensity [DN/s]')
        subplots[0].legend()
        subplots[0].grid(True, alpha=0.3)
    else:
        subplots[0].text(0.5, 0.5, 'AIA data not available', 
                        ha='center', va='center', transform=subplots[0].transAxes)
    
    # bottom panel: dem profile with optional confidence intervals
    dem_pixel = dem_cube[:, r, c]
    # threshold to remove numerical noise artifacts
    dem_pixel[dem_pixel < 0.01] = 0
    
    if ci_low is not None and ci_high is not None:
        # plot with shaded region showing confidence intervals on log scale
        ci_low_pixel = ci_low[:, r, c]
        ci_high_pixel = ci_high[:, r, c]
        
        # ensure all values are positive for log scale
        dem_pixel_plot = np.maximum(dem_pixel, 1e-3)
        ci_low_pixel_plot = np.maximum(ci_low_pixel, 1e-3)
        ci_high_pixel_plot = np.maximum(ci_high_pixel, 1e-3)
        
        # plot dem curve
        subplots[1].plot(logT, dem_pixel_plot, 'o-', lw=2, color='darkblue', 
                        markersize=4, label='DEM', zorder=3)
        
        # plot shaded confidence interval
        subplots[1].fill_between(logT, ci_low_pixel_plot, ci_high_pixel_plot, 
                                alpha=0.3, color='lightblue', label='90% CI', zorder=1)
        
        # plot ci bounds as thin lines
        subplots[1].plot(logT, ci_low_pixel_plot, '--', lw=1, color='steelblue', 
                        alpha=0.6, zorder=2)
        subplots[1].plot(logT, ci_high_pixel_plot, '--', lw=1, color='steelblue', 
                        alpha=0.6, zorder=2)
        
        subplots[1].set_yscale('log')
        subplots[1].legend()
    else:
        # plot without ci on log scale
        dem_pixel_plot = np.maximum(dem_pixel, 1e-3)
        subplots[1].plot(logT, dem_pixel_plot, 'o-', lw=2, color='darkblue', markersize=4)
        subplots[1].set_yscale('log')
    
    subplots[1].set_xlabel(r'$\log_{10} T$ [K]')
    subplots[1].set_ylabel(r'DEM [cm$^{-5}$ K$^{-1}$]')
    subplots[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def dumpPixelDEMs(target, dem_cube, aia_cube, logT, dem_path=None, n_hot=50, n_quiet=50, 
                  hot_frac=0.005, quiet_lo=0.40, quiet_hi=0.60, 
                  disk_shrink=0.98, disk_mode='circle', ci_low=None, ci_high=None):
    """dump dem plots for n pixels from each region (hot, quiet, flaring, coronal_hole)"""
    H, W = aia_cube.shape[1], aia_cube.shape[2]
    
    # load response function for resynth
    RData = np.load("/scratch/vp2435/workspace/dem/demdemo/RData.npz")
    R = RData["R"] * 1e26 
    R = R.astype(np.float64)
    
    # handle xrt additional temperature bins
    if dem_cube.shape[0] > 18:
        logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
        R = np.hstack([R, np.zeros((R.shape[0], logTExpand.size))])
    
    # build on-disk mask
    ondisk = on_disk_mask(H, W, shrink=disk_shrink, mode=disk_mode)
    I193 = aia_cube[3, :, :]  # channel 3 = 193
    
    # check for hardcoded regions
    timestamp = extract_timestamp(dem_path) if dem_path else None
    hardcoded = get_regions_for_timestamp(timestamp) if timestamp else None
    
    if hardcoded:
        print(f"using hardcoded regions for timestamp {timestamp}")
        regions = hardcoded
    else:
        # auto-detect hot and quiet only (flaring/coronal_hole require manual specification)
        print("no hardcoded regions found, auto-detecting hot and quiet regions")
        regions = {
            'hot': select_hot_region_128(I193, region_size=128, top_frac=hot_frac),
            'quiet': select_quiet_region_128(I193, ondisk, region_size=128, q_lo=0.05, q_hi=0.10),
        }
    
    # process each region type
    region_types = ['hot', 'quiet', 'flaring', 'coronal_hole']
    regions_found = {}
    
    for region_type in region_types:
        region_bounds = regions.get(region_type, None)
        if region_bounds is None:
            continue
        
        regions_found[region_type] = region_bounds
        region_dir = os.path.join(target, f"{region_type}_pixels")
        os.makedirs(region_dir, exist_ok=True)
        
        # select pixels from region
        pixels = select_n_pixels_from_region(I193, region_bounds, n_hot)
        print(f"selected {len(pixels)} {region_type} pixels")
        
        # create visualizations for each pixel
        for idx, pixel in enumerate(pixels):
            plot_aia_region_with_pixel(I193, pixel, region_bounds, 
                                      os.path.join(region_dir, f"region_{idx:03d}.png"))
            print(f"plotting {region_type} pixel {idx+1}/{len(pixels)} at {pixel}")
            plot_single_pixel_dem(dem_cube, logT, pixel, 
                                 os.path.join(region_dir, f"dem_{idx:03d}.png"),
                                 aia_cube=aia_cube, R=R, ci_low=ci_low, ci_high=ci_high)
    
    # full aia image with all regions marked
    if regions_found:
        plot_aia_full_with_regions(I193, regions_found,
                                  os.path.join(target, "aia193_regions.png"))
    
    print(f"pixel dem visualizations saved to {target}")

def dumpUncertainty(target, max_prob, entropy, most_likely_bin=None, expected_value=None, stdv=None, bins=None, 
                   dem_truth=None, ci_low=None, ci_high=None):
    """
    Create uncertainty visualizations from classification outputs
    max_prob: [26, H, W] - maximum probabilities 
    entropy: [26, H, W] - entropy values
    most_likely_bin: [26, H, W] - argmax bin indices (optional)
    expected_value: [26, H, W] - expected values from distribution (optional)
    stdv: [26, H, W] - standard deviation values (optional)
    bins: array - bin edges for converting bin indices to values (optional)
    dem_truth: [26, H, W] - ground truth dem for ci evaluation (optional)
    ci_low: [26, H, W] - lower confidence interval bound (optional)
    ci_high: [26, H, W] - upper confidence interval bound (optional)
    """
    # # save colorbars with numbers for reference
    # plt.figure(figsize=(4, 1))
    # plt.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap='viridis')
    # plt.axis('off')
    # plt.colorbar(ticks=np.linspace(0, 1, 6))
    # plt.savefig(os.path.join(target, "uncertainty_maxprob_colorbar.png"), dpi=300)  
    # plt.close()

    # plt.figure(figsize=(4, 1))
    # plt.imshow(np.linspace(0, np.log2(128), 256)[None, :], aspect='auto', cmap='plasma')
    # plt.axis('off')
    # plt.colorbar(ticks=np.linspace(0, np.log2(128), 6))
    # plt.savefig(os.path.join(target, "uncertainty_entropy_colorbar.png"), dpi=300)
    # plt.close()

    # # max probability maps for each temperature channel
    # for i in range(max_prob.shape[0]):
    #     plt.imsave(os.path.join(target, f"uncertainty_maxprob_{i:02d}.png"), 
    #                max_prob[i], cmap='viridis', vmin=0, vmax=1)
    
    # # entropy maps for each temperature channel  
    # for i in range(entropy.shape[0]):
    #     plt.imsave(os.path.join(target, f"uncertainty_entropy_{i:02d}.png"), 
    #                entropy[i], cmap='plasma', vmin=0, vmax=np.log2(127))
    
    # # most likely bin maps and their values
    # if most_likely_bin is not None:
    #     for i in range(most_likely_bin.shape[0]):
    #         plt.imsave(os.path.join(target, f"most_likely_bin_{i:02d}.png"), 
    #                    most_likely_bin[i], cmap='turbo', vmin=0, vmax=127)
            
    #         # convert to values if bins provided
    #         if bins is not None:
    #             bin_midpoints = np.array([(bins[j] + bins[j+1]) / 2 for j in range(len(bins)-1)])
    #             if len(bin_midpoints) < most_likely_bin.max() + 1:
    #                 bin_midpoints = np.append(bin_midpoints, bins[-1])
                
    #             bin_values_map = bin_midpoints[most_likely_bin[i]]
    #             plt.imsave(os.path.join(target, f"most_likely_values_{i:02d}.png"),
    #                       bin_values_map**0.5, cmap='turbo', vmin=bin_values_map.min()**0.5, vmax=10)
    
    # # expected value maps (mean of distribution)
    # if expected_value is not None:
    #     for i in range(expected_value.shape[0]):
    #         plt.imsave(os.path.join(target, f"expected_value_{i:02d}.png"),
    #                   expected_value[i]**0.5, cmap='turbo', vmin=expected_value[i].min()**0.5, vmax=10)
    
    # # standard deviation maps (uncertainty)
    # if stdv is not None:
    #     vmax_stdv = 25  # quarter of default vmax (100)
    #     for i in range(stdv.shape[0]):
    #         plt.imsave(os.path.join(target, f"stdv_{i:02d}.png"),
    #                   stdv[i], cmap='plasma', vmin=0, vmax=vmax_stdv)
    
    # confidence interval visualizations
    if dem_truth is not None and ci_low is not None and ci_high is not None:
        print("creating confidence interval visualizations...")
        
        # compute signed distance and coverage mask
        import torch
        import sys
        sys.path.append('../src')
        from utils import signed_distance_to_ci, ci_coverage
        
        # convert to torch tensors
        y = torch.from_numpy(dem_truth).float()
        lo = torch.from_numpy(ci_low).float()
        hi = torch.from_numpy(ci_high).float()
        
        # compute signed distance for each temperature bin
        signed_dist = signed_distance_to_ci(y, lo, hi)
        _, inside_mask = ci_coverage(y, lo, hi)
        
        # compute vmax for signed distance (95th percentile of absolute values)
        vmax_signed = np.nanpercentile(np.abs(signed_dist.numpy()), 95)
        vmax_signed = max(vmax_signed, 1.0)  # minimum vmax of 1
        
        # save signed distance maps (diverging colormap: blue=below, white=inside, red=above)
        for i in range(signed_dist.shape[0]):
            plt.imsave(os.path.join(target, f"ci_signed_distance_{i:02d}.png"),
                      signed_dist[i].numpy(), cmap='bwr', vmin=-vmax_signed, vmax=vmax_signed)
        
        # save coverage mask maps (binary: 1=inside, 0=outside)
        for i in range(inside_mask.shape[0]):
            plt.imsave(os.path.join(target, f"ci_coverage_mask_{i:02d}.png"),
                      inside_mask[i].float().numpy(), cmap='gray', vmin=0, vmax=1)
        
        # compute and save overall statistics
        coverage_rate = inside_mask.float().mean().item()
        with open(os.path.join(target, "ci_statistics.txt"), 'w') as f:
            f.write(f"confidence interval statistics\n")
            f.write(f"================================\n")
            f.write(f"overall coverage rate: {coverage_rate:.4f} ({coverage_rate*100:.2f}%)\n")
            f.write(f"signed distance vmax: {vmax_signed:.4f}\n\n")
            
            # per-temperature statistics
            f.write(f"per-temperature coverage rates:\n")
            for i in range(inside_mask.shape[0]):
                rate = inside_mask[i].float().mean().item()
                f.write(f"  temp bin {i:02d}: {rate:.4f} ({rate*100:.2f}%)\n")
        
        print(f"overall ci coverage: {coverage_rate*100:.2f}%")


def main():
    '''example usage:
    python dumpVisuals.py --dem_path ../preds/l1_barrier_lr-stable-1e-4/test/dem_20151005_1106.npz --target ../'''
    args = parse_args()
    print("Starting visualizations")

    # open the npz file
    if not os.path.exists(args.dem_path):
        raise FileNotFoundError(f"DEM file not found: {args.dem_path}")
    dem_data = np.load(args.dem_path, mmap_mode='r')
    DEMC = dem_data['DEMCube'] if 'DEMCube' in dem_data else dem_data['dem']
    AIAC = dem_data['AIACube'] if 'AIACube' in dem_data else dem_data['aia']

    # check if the data is compressed
    if DEMC.dtype != np.float32 and DEMC.dtype != np.float64:
        from numcodecs import Blosc
        # Load the npz file
        compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
        DEMData = np.frombuffer(compressor.decode(DEMC), dtype=np.float32).reshape(dem_data['DEMCubeShape'])
        dem = np.ones((DEMData.shape[0], 4096, 4096), dtype=np.float32)*np.nan
        decimationFactor = dem.shape[1] // DEMData.shape[1]
        dem[:, ::decimationFactor, ::decimationFactor] = DEMData
        dem = nnInterpNaN(dem)

        aia = np.frombuffer(compressor.decode(AIAC), dtype=np.float32).reshape(dem_data['AIACubeShape'])
        aia = aia[:6] # assuming the first 6 channels are needed
    else:
        dem = DEMC
        aia = AIAC

    if args.target is None:
        # if target is not provided, create a target directory based on the dem_path
        if args.dem_path.endswith('.npz'):
            base_name = os.path.basename(args.dem_path).replace('.npz', '')
        else:
            base_name = os.path.basename(args.dem_path)
        target_dir = os.path.join("../results", base_name)
    else:
        target_dir = args.target
    print(f"Target directory: {target_dir}")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if args.dem:
        print("Creating DEM visualizations...")
        dumpDEM(target_dir, dem, transformed=None)

    if args.aia_resynth:
        print("Creating AIA visualizations and resynthesis...")
        if aia is None:
            print("No AIA data found in the provided DEM file. Attempting to load from comparison path...")
            # try to infer the AIA data from comparison path if provided
            if args.dem_comparison_path is not None:
                if not os.path.exists(args.dem_comparison_path):
                    raise FileNotFoundError(f"Comparison AIA file not found: {args.dem_comparison_path}")
                aia_data = np.load(args.dem_comparison_path, mmap_mode='r')
                # either "aia" or "AIACube" key
                if 'AIACube' in aia_data:
                    aia = aia_data['AIACube']
                elif 'aia' in aia_data:
                    aia = aia_data['aia']
                else:
                    raise KeyError("No AIA data found in the comparison file.")
                
            else:
                raise ValueError("AIA data is required for resynthesis, but not provided.")

        # load the response function and wavelengths
        if aia is not None:
            RData = np.load("/scratch/vp2435/workspace/dem/demdemo/RData.npz")
            R, logT = RData["R"] * 1e26, RData["logT"]
            R = R.astype(np.float64)
            wavelengths = [94, 131, 171, 193, 211, 335]

            # handle xrt additional temperature bins
            if dem.shape[0] > 18:
                logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
                R = np.hstack([R, np.zeros((R.shape[0], logTExpand.size))])
                logT = np.hstack([logT, logTExpand])
            dumpSynthesis(target_dir, "synth", aia, dem, R, wavelengths)
            dumpDiagnostics(target_dir, dem, logT)
    
    if args.dem_jpdfs:
        print("Creating DEM joint probability density function visualizations...")
        # open npz froma args.dem_comparison_path if provided
        if args.dem_comparison_path is not None:
            if not os.path.exists(args.dem_comparison_path):
                raise FileNotFoundError(f"Comparison DEM file not found: {args.dem_comparison_path}")
            
            dem_comp_data = np.load(args.dem_comparison_path, mmap_mode='r')
            # either "dem" or "DEMCube" key
            if 'DEMCube' in dem_comp_data:
                DEMC = dem_comp_data['DEMCube']
                # check if it is compressed
                if DEMC.dtype != np.float32 or DEMC.dtype != np.float64:
                    from numcodecs import Blosc
                    # Load the npz file
                    compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
                    DEMData = np.frombuffer(compressor.decode(DEMC), dtype=np.float32).reshape(dem_comp_data['DEMCubeShape'])
                    dem_comp = np.ones((DEMData.shape[0], 4096, 4096), dtype=np.float32)*np.nan
                    decimationFactor = dem_comp.shape[1] // DEMData.shape[1]
                    dem_comp[:, ::decimationFactor, ::decimationFactor] = DEMData
                else:
                    dem_comp = DEMC
            elif 'dem' in dem_comp_data:
                dem_comp = dem_comp_data['dem']
            print("Creating joint PDF visualizations with comparison DEM...")
            dumpDEMJointPDF(target_dir, dem_comp, dem, transformed=None)
        else:
            print("No comparison DEM provided, not creating joint PDF visualizations.")
    
    if args.uncertainty:
        print("Creating uncertainty visualizations...")
        
        # check for classification outputs in the data
        if 'ClassificationMaxProb' in dem_data and 'ClassificationEntropy' in dem_data:
            max_prob = dem_data['ClassificationMaxProb']
            entropy = dem_data['ClassificationEntropy']
            
            # optional additional maps
            most_likely_bin = dem_data.get('ClassificationMostLikelyBin', None)
            expected_value = dem_data.get('ClassificationExpectedValue', None)
            stdv = dem_data.get('ClassificationStdv', None)
            bins = dem_data.get('Bins', None)
            
            # confidence interval data
            ci_low = dem_data.get('ClassificationCI_Low', None)
            ci_high = dem_data.get('ClassificationCI_High', None)
            dem_truth = dem  # use loaded ground truth DEM

            # handle compressed data if needed
            if max_prob.dtype not in [np.float16, np.float32, np.float64]:
                from numcodecs import Blosc
                compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
                max_prob = np.frombuffer(compressor.decode(max_prob), dtype=np.float32).reshape(dem_data['ClassificationMaxProbShape'])
                entropy = np.frombuffer(compressor.decode(entropy), dtype=np.float32).reshape(dem_data['ClassificationEntropyShape'])
                
                if most_likely_bin is not None and most_likely_bin.dtype not in [np.int16, np.int32]:
                    most_likely_bin = np.frombuffer(compressor.decode(most_likely_bin), dtype=np.int16).reshape(dem_data['ClassificationMostLikelyBinShape'])
                
                if expected_value is not None and expected_value.dtype not in [np.float16, np.float32, np.float64]:
                    expected_value = np.frombuffer(compressor.decode(expected_value), dtype=np.float32).reshape(dem_data['ClassificationExpectedValueShape'])
                
                if stdv is not None and stdv.dtype not in [np.float16, np.float32, np.float64]:
                    stdv = np.frombuffer(compressor.decode(stdv), dtype=np.float32).reshape(dem_data['ClassificationStdvShape'])
                
                if ci_low is not None and ci_low.dtype not in [np.float16, np.float32, np.float64]:
                    ci_low = np.frombuffer(compressor.decode(ci_low), dtype=np.float32).reshape(dem_data['ClassificationCI_LowShape'])
                
                if ci_high is not None and ci_high.dtype not in [np.float16, np.float32, np.float64]:
                    ci_high = np.frombuffer(compressor.decode(ci_high), dtype=np.float32).reshape(dem_data['ClassificationCI_HighShape'])

            dumpUncertainty(target_dir, max_prob, entropy, most_likely_bin, expected_value, stdv, bins,
                          dem_truth=dem_truth, ci_low=ci_low, ci_high=ci_high)
        else:
            print("No uncertainty data found in npz file. Expected 'ClassificationMaxProb' and 'ClassificationEntropy' keys.")
    
    if args.dem_pixels:
        print("Creating pixel DEM visualizations...")
        # need logT from response function
        logT = [5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2]
        
        # handle xrt additional temperature bins
        if dem.shape[0] > 18:
            logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
            logT = np.hstack([logT, logTExpand])
        
        # load confidence interval data if available
        ci_low = dem_data.get('ClassificationCI_Low', None)
        ci_high = dem_data.get('ClassificationCI_High', None)
        
        if ci_low is not None and ci_high is not None:
            print("confidence interval data found, will plot error bars")
        else:
            print("no confidence interval data found, plotting without error bars")
        
        dumpPixelDEMs(target_dir, dem, aia, logT, dem_path=args.dem_path,
                     n_hot=args.n_hot, n_quiet=args.n_quiet,
                     hot_frac=args.hot_frac, quiet_lo=args.quiet_lo, quiet_hi=args.quiet_hi,
                     disk_shrink=args.disk_shrink, disk_mode=args.disk_mode,
                     ci_low=ci_low, ci_high=ci_high)
    
    if args.regions_jpdfs:
        print("Creating region AIA vs resynth JPDFs...")
        dumpRegionJPDFs(target_dir, dem, aia, dem_path=args.dem_path,
                       hot_frac=args.hot_frac, disk_shrink=args.disk_shrink, 
                       disk_mode=args.disk_mode)
    
    if args.roi_dems:
        print("Creating ROI DEM visualizations...")
        dumpROIDEMs(target_dir, dem, aia, dem_path=args.dem_path)


if __name__ == "__main__":
    main()