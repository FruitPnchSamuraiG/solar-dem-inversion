"""
predict from the new models with regression by classification
very specific script for the npz in the current dataset
"""

import argparse
import os
import re
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from numcodecs import Blosc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import SimpleAIAData
from src.model import BasicNetworkFreqClass, BasicNetworkFreqClassConv, BasicNetworkFreqClassSmall, NoNormMixer, FreqClassNonReLU, FreqClassNonReLUNoPos
from src.utils import unfold_tensor, reconstruct_cube, create_dem_bins, quantiles_from_pmf, processIndAIAData

_models = {
    'BasicNetworkFreqClass': BasicNetworkFreqClass,
    'BasicNetworkFreqClassConv': BasicNetworkFreqClassConv,
    'BasicNetworkFreqClassSmall': BasicNetworkFreqClassSmall,
    'NoNormMixer': NoNormMixer,
    'FreqClassNonReLU': FreqClassNonReLU,
    'FreqClassNonReLUNoPos': FreqClassNonReLUNoPos
}

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with dual-head classification models")
    parser.add_argument("--mode", choices=["raw_aia", "npz_full"],
                        default="npz_full",
                        help="input type: raw AIA directory or npz file (default: npz_full)")
    parser.add_argument("--input", required=True, help="path to input .npz file or raw AIA directory")
    parser.add_argument("--model", required=True, help="path to trained model .pth file")
    parser.add_argument("--target", default=None,
                        help="output directory (will be created if not exists)")
    parser.add_argument("--output_mode", choices=["regression", "classification", "both"],
                        default="both", help="which head output to save")
    parser.add_argument("--n_bins", type=int, default=128, help="number of classification bins")
    parser.add_argument("--vmin", type=float, default=None, help="min DEM value for binning (default: read from checkpoint, else 0)")
    parser.add_argument("--vmax", type=float, default=None, help="max DEM value for binning (default: read from checkpoint, else 2000)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument('--bin_spacing', type=str, default=None, choices=['linear', 'sqrt', 'log'],
                        help="bin spacing (default: read from checkpoint, else linear)")
    parser.add_argument("--corr_table", type=str, default="aia_corr.csv",
                        help="correlation table for raw_aia mode")
    parser.add_argument("--deconvolve", type=str, default="none",
                        choices=["none", "hofmeister"],
                        help="deconvolution method applied to AIA inputs before inference")
    parser.add_argument("--apply_relu", action="store_true",
                        help="apply relu to regression output (for nonrelu models)")
    parser.add_argument("--aia_only", action="store_true",
                        help="zero out xrt bin weights (bins 18-25) for aia-only models")

    return parser.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("DEM Classification Prediction (Memory-Optimized)")
    print("="*60)
    print(f"configuration:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_bins: {args.n_bins}")
    print(f"  output_mode: {args.output_mode}")
    print("="*60)
    
    full_path = args.input
    timestamp_pattern = r"(\d{8}_\d{6})"
    match = re.search(timestamp_pattern, full_path)
    if match:
        aia_timestamp = match.group(1)
    # else if try to find YYYYMMDD_HHMM only
    else:
        timestamp_pattern_short = r"(\d{8}_\d{4})"
        match_short = re.search(timestamp_pattern_short, full_path)
        if match_short:
            aia_timestamp = match_short.group(1) + "00"  # add 00 seconds to normalize
        else:
            raise ValueError("aia_timestamp not found in input path")

    print(f"AIA timestamp: {aia_timestamp}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load model
    print(f"Loading model from {args.model}")
    model_pack = torch.load(args.model, map_location=device)
    model_state_dict = model_pack['model_state_dict']
    model_name_internal = model_pack['model_name']
    loss_name = model_pack['loss_name']
    run_name = model_pack['experiment_name']
    timestamp_model = model_pack['timestamp']
    
    # extract model folder name from path to match checkpoint filename
    # e.g., ../results/models/FreqClassNonReLU-RegressionClassificationLoss_jnonrelu.../model_best.pth
    # -> FreqClassNonReLU-RegressionClassificationLoss_jnonrelu...
    model_folder = os.path.basename(os.path.dirname(args.model))
    
    # parse model name from folder (first part before first -)
    if '-' in model_folder:
        model_name = model_folder.split('-')[0]
    else:
        # fallback to internal name if parsing fails
        model_name = model_name_internal
    
    print(f"Using model: {model_name}, loss: {loss_name}")

    # read vmin/vmax/bin_spacing from checkpoint args when not overridden on CLI
    ckpt_args = model_pack.get('args', {})
    if args.vmin is None:
        args.vmin = ckpt_args.get('vmin', 0)
        print(f"vmin from checkpoint: {args.vmin}")
    if args.vmax is None:
        args.vmax = ckpt_args.get('vmax', 2000)
        print(f"vmax from checkpoint: {args.vmax}")
    if args.bin_spacing is None:
        args.bin_spacing = ckpt_args.get('bin_spacing', 'linear')
        print(f"bin_spacing from checkpoint: {args.bin_spacing}")

    # infer n_bins from checkpoint if not matching
    # classification_head.weight shape is [nOut * n_bins, 256, 1, 1]
    classification_head_shape = model_state_dict['classification_head.weight'].shape[0]
    n_bins_from_checkpoint = classification_head_shape // 26  # nOut = 26
    
    if n_bins_from_checkpoint != args.n_bins:
        print(f"Warning: Checkpoint was trained with {n_bins_from_checkpoint} bins, but --n_bins={args.n_bins} specified")
        print(f"Using n_bins={n_bins_from_checkpoint} from checkpoint")
        args.n_bins = n_bins_from_checkpoint
    
    # init model
    model = _models[model_name_internal](n_bins=args.n_bins)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    # zero out xrt bin weights (18-25) for aia-only models
    if args.aia_only:
        print("zeroing xrt bin weights (18-25) in regression and classification heads")
        with torch.no_grad():
            # regression: zero weights and bias
            model.regression_head[0].weight[18:26, :, :, :] = 0
            if model.regression_head[0].bias is not None:
                model.regression_head[0].bias[18:26] = 0
            
            # classification: for each xrt temperature bin, create peaked distribution at dem bin 0
            # classification_head outputs [nOut * n_bins] channels, reshaped to [nOut, n_bins, H, W]
            for temp_bin in range(18, 26):
                start_ch = temp_bin * args.n_bins
                end_ch = (temp_bin + 1) * args.n_bins
                
                # zero all weights (no input dependency)
                model.classification_head.weight[start_ch:end_ch, :, :, :] = 0
                
                if model.classification_head.bias is not None:
                    # set dem bin 0 to high value (peaked distribution)
                    model.classification_head.bias[start_ch] = 10.0
                    # set all other dem bins to very negative
                    model.classification_head.bias[start_ch+1:end_ch] = -1e10
    
    model.eval()
    print(f"Model loaded successfully with n_bins={args.n_bins}")
    
    bins = create_dem_bins(vmin=args.vmin, vmax=args.vmax, n_bins=args.n_bins, spacing=args.bin_spacing)
    
    # load data based on mode
    if args.mode == "raw_aia":
        print(f"Loading raw AIA data from {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        
        # add preprocess attribute if not present
        if not hasattr(args, 'preprocess'):
            args.preprocess = ''
        
        AIACube, aia_errors, scale_factor = processIndAIAData(args.input, args)
        print(f"AIA cube shape: {AIACube.shape}")
        print(f"AIA errors shape: {aia_errors.shape}")
        
    elif args.mode == "npz_full":
        print(f"Loading NPZ data from {args.input}")
        compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
        data = np.load(args.input, mmap_mode='r')
        
        if 'AIACube' in data:
            AIA = data['AIACube']
            AIACube = np.frombuffer(compressor.decode(AIA), dtype=np.float32).reshape(data['AIACubeShape'])
            AIACube = AIACube[:6]  # first 6 channels
        else:
            raise ValueError("Input npz file must contain 'AIACube'")
        
        if 'DEMCube' in data:    
            DEM = data['DEMCube']
            DEMCube = np.frombuffer(compressor.decode(DEM), dtype=np.float32).reshape(data['DEMCubeShape'])
            print(f"DEM cube shape: {DEMCube.shape}")
        else:
            print("No DEM cube in input")
        
        print(f"AIA cube shape: {AIACube.shape}")
    
    # unfold into patches
    aia_patches = unfold_tensor(AIACube, 256, 256)
    err_patches = torch.randn(aia_patches.shape, dtype=torch.float32).numpy()  # dummy errors

    # create dataset and loader
    dataset = SimpleAIAData((aia_patches, err_patches))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Processing {len(aia_patches)} patches...")
    print(f"Input shape: {aia_patches.shape}, Errors shape: {err_patches.shape}")
    
    # compute bin midpoints for expected value calculation
    bin_midpoints = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    if len(bin_midpoints) < args.n_bins:
        bin_midpoints = np.append(bin_midpoints, bins[-1])
    bin_midpoints_tensor = torch.tensor(bin_midpoints, dtype=torch.float32).to(device)
    
    # pre-allocate tensors on CPU to save GPU memory
    # only keep batch-sized tensors on GPU during processing
    num_patches = len(aia_patches)
    print(f"allocating output tensors on CPU (num_patches={num_patches})")
    
    reg_cube = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_max_prob = torch.empty((num_patches, 26, 256, 256), dtype=torch.float16)
    cls_entropy = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_most_likely_bin = torch.empty((num_patches, 26, 256, 256), dtype=torch.int16)
    cls_expected_value = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_stdv = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_ci_low = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_ci_med = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    cls_ci_high = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
    
    print(f"total CPU memory allocated: ~{(num_patches * 26 * 256 * 256 * (4*7 + 2*2)) / 1e9:.2f} GB")
    
    cube_idx = 0
    with torch.no_grad():
        for batch_idx, (aia_batch, _) in enumerate(loader):
            aia_batch = aia_batch.to(device)
            
            # forward pass
            reg_out, cls_out = model(aia_batch)
            
            # apply relu if requested (for nonrelu models)
            if args.apply_relu:
                reg_out = torch.clamp(reg_out, min=0)
            
            # store regression directly (move to CPU immediately to save GPU memory)
            batch_size = reg_out.shape[0]
            reg_cube[cube_idx:cube_idx + batch_size] = reg_out.cpu()
            
            # compute probabilities - this is the memory bottleneck
            # cls_out: [B, 26, n_bins, H, W] where n_bins=128 is large
            # instead of keeping cls_probs in memory, process temperature bins one at a time
            
            # first pass: compute metrics that need argmax/max (don't need full probs)
            with torch.no_grad():
                # max probability (computed from logits directly)
                max_logit = torch.max(cls_out, dim=2)[0]  # [B, 26, H, W]
                max_prob = torch.softmax(cls_out, dim=2).max(dim=2)[0]  # [B, 26, H, W]
                cls_max_prob[cube_idx:cube_idx + batch_size] = max_prob.cpu().to(torch.float16)
                del max_prob, max_logit
                
                # most likely bin (argmax on logits, no softmax needed)
                most_likely_bin = torch.argmax(cls_out, dim=2)  # [B, 26, H, W]
                cls_most_likely_bin[cube_idx:cube_idx + batch_size] = most_likely_bin.cpu().to(torch.int16)
                del most_likely_bin
            
            # second pass: process each temperature bin separately to save memory
            # allocate space for accumulated results
            entropy_accum = torch.zeros((batch_size, 26, 256, 256), device=device)
            expected_accum = torch.zeros((batch_size, 26, 256, 256), device=device)
            expected_sq_accum = torch.zeros((batch_size, 26, 256, 256), device=device)
            
            # process in temperature-bin batches
            for temp_idx in range(26):
                logits_temp = cls_out[:, temp_idx, :, :, :]  # [B, n_bins, H, W]
                probs_temp = torch.softmax(logits_temp, dim=1)  # [B, n_bins, H, W]
                
                # entropy for this temperature
                entropy_temp = -torch.sum(probs_temp * torch.log(probs_temp + 1e-8), dim=1)  # [B, H, W]
                entropy_accum[:, temp_idx, :, :] = entropy_temp
                
                # expected value for this temperature
                expected_temp = torch.sum(probs_temp * bin_midpoints_tensor[None, :, None, None], dim=1)
                expected_accum[:, temp_idx, :, :] = expected_temp
                
                # expected x^2 for this temperature
                expected_sq_temp = torch.sum(probs_temp * (bin_midpoints_tensor[None, :, None, None] ** 2), dim=1)
                expected_sq_accum[:, temp_idx, :, :] = expected_sq_temp
                
                del probs_temp, entropy_temp, expected_temp, expected_sq_temp
            
            # store results
            cls_entropy[cube_idx:cube_idx + batch_size] = entropy_accum.cpu()
            cls_expected_value[cube_idx:cube_idx + batch_size] = expected_accum.cpu()
            
            # compute stddev
            variance = expected_sq_accum - (expected_accum ** 2)
            stdv = torch.sqrt(torch.clamp(variance, min=0))
            cls_stdv[cube_idx:cube_idx + batch_size] = stdv.cpu()
            
            del entropy_accum, expected_accum, expected_sq_accum, variance, stdv
            
            # third pass: compute confidence interval quantiles for each temperature bin
            # use_bin_edges=True to allow CIs to reach bin boundaries for peaked distributions (e.g., background pixels)
            for temp_idx in range(26):
                logits_temp = cls_out[:, temp_idx, :, :, :]  # [B, n_bins, H, W]
                low, med, high = quantiles_from_pmf(logits_temp, bins, q_low=0.05, q_med=0.50, q_high=0.95, use_bin_edges=True)
                cls_ci_low[cube_idx:cube_idx + batch_size, temp_idx, :, :] = low.cpu()
                cls_ci_med[cube_idx:cube_idx + batch_size, temp_idx, :, :] = med.cpu()
                cls_ci_high[cube_idx:cube_idx + batch_size, temp_idx, :, :] = high.cpu()
                del low, med, high
            
            # free large GPU tensors before next iteration
            del cls_out, reg_out
            torch.cuda.empty_cache()  # explicitly free GPU memory
            
            cube_idx += batch_size
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(loader)}")
    
    print("All batches processed, reconstructing full cubes...")
    print("(tensors are on CPU, reconstruction will be memory-efficient)")
    
    print(f"Regression output shape: {reg_cube.shape}")
    print(f"Classification max probability shape: {cls_max_prob.shape}")
    print(f"Classification most likely bin shape: {cls_most_likely_bin.shape}")
    print(f"Classification entropy shape: {cls_entropy.shape}")
    print(f"Classification expected value shape: {cls_expected_value.shape}")
    print(f"Classification stdv shape: {cls_stdv.shape}")
    print(f"Classification CI low shape: {cls_ci_low.shape}")
    print(f"Classification CI med shape: {cls_ci_med.shape}")
    print(f"Classification CI high shape: {cls_ci_high.shape}")
    
    # reconstruct full cubes (on CPU)
    print("reconstructing regression...")
    reg_full_cube = reconstruct_cube(reg_cube, (26, 4096, 4096), numpy=False)
    del reg_cube  # free memory
    
    print("reconstructing classification outputs...")
    cls_max_prob_full = reconstruct_cube(cls_max_prob, (26, 4096, 4096), numpy=False)
    del cls_max_prob
    
    # convert int16 to float32 for reconstruction, then back to int16
    cls_most_likely_bin_float = cls_most_likely_bin.to(torch.float32)
    del cls_most_likely_bin
    cls_most_likely_bin_full = reconstruct_cube(cls_most_likely_bin_float, (26, 4096, 4096), numpy=False)
    del cls_most_likely_bin_float
    cls_most_likely_bin_full = cls_most_likely_bin_full.to(torch.int16)
    
    cls_entropy_full = reconstruct_cube(cls_entropy, (26, 4096, 4096), numpy=False)
    del cls_entropy
    
    cls_expected_value_full = reconstruct_cube(cls_expected_value, (26, 4096, 4096), numpy=False)
    del cls_expected_value
    
    cls_stdv_full = reconstruct_cube(cls_stdv, (26, 4096, 4096), numpy=False)
    del cls_stdv
    
    # reconstruct confidence intervals
    print("reconstructing confidence intervals...")
    cls_ci_low_full = reconstruct_cube(cls_ci_low, (26, 4096, 4096), numpy=False)
    del cls_ci_low
    
    cls_ci_med_full = reconstruct_cube(cls_ci_med, (26, 4096, 4096), numpy=False)
    del cls_ci_med
    
    cls_ci_high_full = reconstruct_cube(cls_ci_high, (26, 4096, 4096), numpy=False)
    del cls_ci_high

    # print out a row of each for verification
    print("Sample outputs at [6, 2048, 2040:2050]:")
    print("Regression:", reg_full_cube[6, 2048, 2040:2050])
    print("Classification Max Probability:", cls_max_prob_full[6, 2048, 2040:2050])
    print("Classification Most Likely Bin:", cls_most_likely_bin_full[6, 2048, 2040:2050])
    print("Classification Entropy:", cls_entropy_full[6, 2048, 2040:2050])
    print("Classification Expected Value:", cls_expected_value_full[6, 2048, 2040:2050])
    print("Classification Stdv:", cls_stdv_full[6, 2048, 2040:2050])
    print("Classification CI Low:", cls_ci_low_full[6, 2048, 2040:2050])
    print("Classification CI Med:", cls_ci_med_full[6, 2048, 2040:2050])
    print("Classification CI High:", cls_ci_high_full[6, 2048, 2040:2050])

    # prepare outputs
    output_dict = {}
    
    if args.output_mode in ["regression", "both"]:
        output_dict['DEMCube'] = reg_full_cube.cpu().numpy()
    
    if args.output_mode in ["classification", "both"]:
        output_dict['ClassificationMaxProb'] = cls_max_prob_full.cpu().numpy()
        output_dict['ClassificationMostLikelyBin'] = cls_most_likely_bin_full.cpu().numpy()
        output_dict['ClassificationEntropy'] = cls_entropy_full.cpu().numpy()
        output_dict['ClassificationExpectedValue'] = cls_expected_value_full.cpu().numpy()
        output_dict['ClassificationStdv'] = cls_stdv_full.cpu().numpy()
        output_dict['ClassificationCI_Low'] = cls_ci_low_full.cpu().numpy()
        output_dict['ClassificationCI_Med'] = cls_ci_med_full.cpu().numpy()
        output_dict['ClassificationCI_High'] = cls_ci_high_full.cpu().numpy()
        output_dict['Bins'] = bins
    
    # include original aia
    output_dict['AIACube'] = AIACube
    
    # save
    if args.target:
        out_dir = args.target
    else:
        # use absolute path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "../preds")
    out_dir = os.path.join(out_dir, f"{model_name}-{loss_name}_{run_name}_{timestamp_model}")
    print(f"Output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    dem_out_name = f"dem_{aia_timestamp}.npz"
    dem_out_path = os.path.join(out_dir, dem_out_name)
    print(f"Saving to {dem_out_path}")
    np.savez(dem_out_path, **output_dict)
    print("Prediction complete!")


if __name__ == "__main__":
    main()