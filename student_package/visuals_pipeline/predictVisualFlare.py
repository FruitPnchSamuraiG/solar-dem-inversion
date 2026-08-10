#!/usr/bin/env python3
"""
predict and visualize dem results for a single flare timestamp
single-case version for array job parallelization
"""

import os
import sys
import argparse
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def process_single_case(model_name, input_path, timestamp, skip_existing=True, just_predict=False):
    """process a single model-timestamp combination"""
    print(f"\n=== processing model: {model_name} ===")
    print(f"timestamp: {timestamp}")
    print(f"input_path: {input_path}")
    
    # validate model name
    if not model_name or model_name.strip() == "":
        print("error: empty model name provided")
        sys.exit(1)

    # handle special models: lp, qp, enet (skip prediction, only create visuals)
    if model_name in ["lp", "qp", "enet", "lp_AIA", "lp_AIA_notrunc", "lp_AIA_notrunc_nodiracs"]:
        print("skipping prediction, only creating visuals")
        if model_name == "enet":
            npz_folder = f"/scratch/vp2435/workspace/dem/data/xrtData_enet_noisy_half_a1.0_l10.5/{timestamp}.npz"
        else:
            #npz_folder = f"/home/vp2435/workspace/dem/data/xrtData_{model_name}_AIA_noisy/{timestamp}.npz"
            npz_folder = f"/scratch/vp2435/workspace/dem/data/{model_name}/{timestamp}.npz"
        visual_cmd = (
            f"python3 -u {SCRIPT_DIR}/createTestVisuals.py "
            f"--input_folder {npz_folder} "
            f"--run_name {model_name} "
        )
        print(f"running visualization: {visual_cmd}")
        result = os.system(visual_cmd)
        if result != 0:
            print(f"error: visualization failed for {timestamp}")
            sys.exit(1)
        print("\ndone! check results in:")
        print(f"  visualizations: /scratch/vp2435/workspace/dem/demdemo/results/vis/{model_name}/")
        return
    
    # parse model name
    model_split = model_name.replace("_", "-").split("-")
    if len(model_split) < 2:
        print(f"error: invalid model name format '{model_name}' - expected format: ModelName-LossName_run_details_timestamp")
        sys.exit(1)
    
    name_model = model_split[0]
    name_loss = model_split[1]
    name_run = "-".join(model_split[2:-2]) if len(model_split) > 2 else ""
    print(f"using model: {name_model}, loss: {name_loss}")
    
    model_dir = f"/scratch/vp2435/workspace/dem/demdemo/results/models/{model_name}"
    
    # use model_best, if not found, use latest epoch
    model_best_path = f"{model_dir}/model_best.pth"
    if os.path.exists(model_best_path):
        model_path = model_best_path
        print(f"using best model: {model_path}")
    else:
        print(f"best model not found, searching for latest epoch in {model_dir}")
        epoch_files = [f for f in os.listdir(model_dir) if f.startswith("epoch_") and f.endswith(".pth")]
        n_epochs = len(epoch_files)
        model_path = f"{model_dir}/epoch_{n_epochs}.pth"
    
    # heuristic for bin spacing: use 'log' when model/run name mentions log/logbins, else 'linear'
    lower_model_name = model_name.lower() if model_name else ""
    lower_run = name_run.lower() if name_run else ""
    if "logbins" in lower_model_name or "logbins" in lower_run or "log" in lower_model_name or "log" in lower_run:
        bin_spacing = "log"
    elif "sqrt" in lower_model_name or "sqrt" in lower_run:
        bin_spacing = "sqrt"
    else:
        bin_spacing = "linear"
    print(f"bin spacing heuristic: {bin_spacing}")
    
    # heuristic for apply_relu: use when model name contains 'jnonrelu' or 'nonrelu'
    apply_relu = "jnonrelu" in lower_model_name or "jnonrelu" in lower_run or "nonrelu" in lower_model_name or "nonrelu" in lower_run
    print(f"apply relu: {apply_relu}")
    
    # heuristic for aia_only: use when model name contains 'AIA' but not 'AIAXRT' or 'XRT'
    aia_only = ("aia" in lower_model_name or "aia" in lower_run) and not ("aiaxrt" in lower_model_name or "xrt" in lower_model_name or "aiaxrt" in lower_run or "xrt" in lower_run)
    print(f"aia only (zero xrt weights): {aia_only}")

    # heuristic for deconvolution: use hofmeister when model name contains 'hofdeconv'
    deconvolve = "hofdeconv" in lower_model_name or "hofdeconv" in lower_run
    print(f"deconvolve (hofmeister): {deconvolve}")
    
    # process single timestamp
    input_folder = f"/scratch/vp2435/workspace/dem/demdemo/data/flares/{timestamp}"

    # when non-flares (this is not ideally hard coded, lol)
    #input_folder = f"/scratch/vp2435/workspace/dem/data/xrtSource/{timestamp}"
    #input_folder = f"/scratch/vp2435/workspace/dem/data/flares/{model_name}/{timestamp}"

    output_folder = f"/scratch/vp2435/workspace/dem/demdemo/preds/{model_name}/"
    
    if not os.path.exists(input_folder):
        print(f"error: input folder {input_folder} not found")
        sys.exit(1)
    
    # normalize timestamp to YYYYMMDD_HHMMSS format
    # timestamps can be: YYYYMMDD_HHMM (4 digits) or YYYYMMDD_HHMMSS (6 digits)
    timestamp_parts = timestamp.split('_')
    if len(timestamp_parts) == 2:
        time_part = timestamp_parts[1]
        if len(time_part) == 4:
            # HHMM format, add 00 seconds
            normalized_timestamp = timestamp + "00"
        elif len(time_part) == 6:
            # HHMMSS format, already complete
            normalized_timestamp = timestamp
        else:
            print(f"warning: unexpected timestamp format '{timestamp}', using as-is")
            normalized_timestamp = timestamp
    else:
        print(f"warning: unexpected timestamp format '{timestamp}', using as-is")
        normalized_timestamp = timestamp
    
    print(f"normalized timestamp: {normalized_timestamp}")
    
    prediction_npz = f"{output_folder}/dem_{normalized_timestamp}.npz"

    if skip_existing and os.path.exists(prediction_npz):
        print(f"prediction already exists: {prediction_npz}, skipping prediction step")
    else:
        # prediction - use raw_aia mode for fits data, use actual input_folder
        predict_cmd = (
            f"python3 -u {SCRIPT_DIR}/predict_classification.py "
            f"--model {model_path} "
            f"--input {input_folder} "
            f"--mode raw_aia "
            f"--bin_spacing {bin_spacing}"
        )
        
        # add apply_relu flag for nonrelu models
        if apply_relu:
            predict_cmd += " --apply_relu"

        # add aia_only flag to zero xrt bin weights
        if aia_only:
            predict_cmd += " --aia_only"

        # add deconvolution flag for hofdeconv models
        if deconvolve:
            predict_cmd += " --deconvolve hofmeister"
        
        print(f"running prediction: {predict_cmd}")
        result = os.system(predict_cmd)
    
        if result != 0:
            print(f"error: prediction failed for {timestamp}")
            sys.exit(1)
    
    # use normalized timestamp (YYYYMMDD_HHMMSS format)
    prediction_npz = f"{output_folder}/dem_{normalized_timestamp}.npz"
    
    # verify prediction exists before proceeding
    if not os.path.exists(prediction_npz):
        print(f"error: prediction file not found: {prediction_npz}")
        sys.exit(1)
    
    print(f"using prediction: {prediction_npz}")
    
    # visualization - createTestVisuals expects prediction npz files
    # create dummy comparison folder to avoid crashes
    dummy_dem_folder = "/scratch/vp2435/workspace/dem/demdemo/data/dummy_out"
    os.makedirs(dummy_dem_folder, exist_ok=True)
    
    pred_npz = prediction_npz

    if just_predict:
        print("skipping visualization")
        exit(0)

    visual_cmd = (
        f"python3 -u {SCRIPT_DIR}/createTestVisuals.py "
        f"--input_folder {pred_npz} "
        f"--run_name {model_name} "
        f"--dem_folder {dummy_dem_folder}"
    )
    
    print(f"running visualization: {visual_cmd}")
    result = os.system(visual_cmd)
    
    if result != 0:
        print(f"error: visualization failed for {timestamp}")
        sys.exit(1)
    
    print(f"\ndone! check results in:")
    print(f"  predictions: /scratch/vp2435/workspace/dem/demdemo/preds/{model_name}/")
    print(f"  pixel dem visualizations: /scratch/vp2435/workspace/dem/demdemo/preds/{model_name}/vis_*/")
    print(f"  standard visualizations: /scratch/vp2435/workspace/dem/demdemo/results/vis/{model_name}/")

def main():
    parser = argparse.ArgumentParser(
        description="predict and visualize dem for single flare timestamp"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="full model name (e.g., BasicNetworkFreqClass-RegressionClassificationLoss_joint_lp_noisy_1e4sch_logbins_20251028_191347)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="input data path identifier (e.g., xrtData_lp_logbins)"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="flare timestamp in YYYYMMDD_HHMM format (e.g., 20110101_2330)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip if prediction already exists"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force regeneration even if prediction exists"
    )
    parser.add_argument(
        "--just_predict",
        action="store_true",
        help="only run prediction, skip visualization"
    )
    
    args = parser.parse_args()
    
    skip_existing = args.skip_existing and not args.force
    just_predict = args.just_predict

    process_single_case(
        model_name=args.model_name,
        input_path=args.input_path,
        timestamp=args.timestamp,
        skip_existing=skip_existing,
        just_predict=just_predict
    )

if __name__ == "__main__":
    main()
