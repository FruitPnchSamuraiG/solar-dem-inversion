#!/usr/bin/env python3
"""
predict and visualize dem results for flare timestamps - quick and dirty version
"""

import os
import sys

# hardcoded flag: set to True to skip existing predictions, False to redo everything
SKIP_EXISTING_PREDICTIONS = True

def get_flare_timestamps():
    """get all timestamps from data/flares directory"""
    flare_dir = "../data/flares"
    if not os.path.exists(flare_dir):
        print(f"error: flare directory {flare_dir} not found")
        return []
    
    # get all timestamp folders
    timestamps = []
    for item in os.listdir(flare_dir):
        item_path = os.path.join(flare_dir, item)
        if os.path.isdir(item_path) and len(item) == 13:  # YYYYMMDD_HHMM 
            timestamps.append(item)
    
    return sorted(timestamps)

def process_model(model_name, input_path):
    """process a single model"""
    print(f"\n=== processing model: {model_name} ===")
    
    # get timestamps
    timestamps = get_flare_timestamps()
    if not timestamps:
        print("no flare timestamps found in data/flares")
        return
    
    print(f"processing {len(timestamps)} timestamps: {timestamps[:5]}...")
    
    # parse model name
    model_split = model_name.replace("_", "-").split("-")
    name_model = model_split[0]
    name_loss = model_split[1]
    name_run = "-".join(model_split[2:-2])
    print(f"using model: {name_model}, loss: {name_loss}")
    
    model_dir = f"../results/models/{model_name}"
    
    # use latest epoch
    epoch_files = [f for f in os.listdir(model_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    n_epochs = len(epoch_files)
    model_path = f"{model_dir}/epoch_{n_epochs}.pth"
    
    # process each timestamp
    base_method = input_path.split("_")[1]  # extract "lp" or "qp" from input path
    base_path = "/scratch/projects/fouheylab/solar_dem/"

    # heuristic for bin spacing: use 'log' when model/run name mentions log/logbins, else 'linear'
    lower_model_name = model_name.lower() if model_name else ""
    lower_run = name_run.lower() if name_run else ""
    if "logbins" in lower_model_name or "logbins" in lower_run or "log" in lower_model_name or "log" in lower_run:
        bin_spacing = "log"
    elif "sqrt" in lower_model_name or "sqrt" in lower_run:
        bin_spacing = "sqrt"
    else:
        bin_spacing = "linear"
    print(f"  bin spacing heuristic: {bin_spacing}")
    
    for timestamp in timestamps[0:1]:
        print(f"\nprocessing {timestamp}...")
        
        input_folder = f"../data/flares/{timestamp}"
        output_folder = f"../preds/{model_name}/"
        
        if not os.path.exists(input_folder):
            print(f"warning: input folder {input_folder} not found, skipping")
            continue
        
        # check if prediction already exists
        temp_timestamp = timestamp + "00"  # add 00 seconds
        prediction_npz = f"{output_folder}/dem_{temp_timestamp}.npz"

        if SKIP_EXISTING_PREDICTIONS and os.path.exists(prediction_npz):
            print(f"prediction already exists: {prediction_npz}, skipping prediction step")
        else:
            # create temp symlink with expected timestamp format (add :00 seconds)
            # predict.py expects YYYYMMDD_HHMMSS but flares have YYYYMMDD_HHMM
            temp_timestamp = timestamp + "00"  # add 00 seconds
            temp_folder = f"../data/flares/{temp_timestamp}"
        
            # create symlink if it doesn't exist
            if not os.path.exists(temp_folder):
                os.symlink(timestamp, temp_folder)
            
            # prediction - use raw_aia mode for fits data
            predict_cmd = (
                f"python3 -u predict_classification.py "
                f"--model {model_path} "
                f"--input {temp_folder} "
                f"--mode raw_aia "
                f"--bin_spacing {bin_spacing}"
            )
            
            print(f"running prediction: {predict_cmd}")
            result = os.system(predict_cmd)
            
            # cleanup temp symlink
            if os.path.islink(temp_folder):
                os.unlink(temp_folder)
        
            if result != 0:
                print(f"error: prediction failed for {timestamp}")
                continue
        
        # keep the prediction with full timestamp (YYYYMMDD_HHMMSS format)
        # the file created by predict_classification.py has correct CI values
        prediction_npz = f"{output_folder}/dem_{temp_timestamp}.npz"
        
        # verify prediction exists before proceeding
        if not os.path.exists(prediction_npz):
            print(f"error: prediction file not found: {prediction_npz}")
            continue
        
        print(f"using prediction: {prediction_npz}")
        # visualization - createTestVisuals expects prediction npz files
        # create dummy comparison folder to avoid crashes
        dummy_dem_folder = "../data/dummy_out"
        os.makedirs(dummy_dem_folder, exist_ok=True)
        
        pred_npz = prediction_npz
        visual_cmd = (
            f"python3 -u createTestVisuals.py "
            f"--input_folder {pred_npz} "
            f"--run_name {model_name} "
            f"--dem_folder {dummy_dem_folder}"
        )
        
        print(f"running visualization: {visual_cmd}")
        result = os.system(visual_cmd)
        
        if result != 0:
            print(f"error: visualization failed for {timestamp}")
            continue
    
    print(f"\ndone with {model_name}! check results in:")
    print(f"  predictions: ../preds/{model_name}/")
    print(f"  pixel dem visualizations: ../preds/{model_name}/vis_*/")
    print(f"  standard visualizations: ../results/vis/{model_name}/")

def main():
    # hardcoded models - the two specific ones you want
    # models = [
    #     ("BasicNetworkFreq-MaskedMSELoss_xrt_lp_long_slr_20250814_015419", "xrtData_lp_full"),
    #     ("BasicNetworkFreq-MaskedMSELoss_xrt_qp_long_slr_20250814_015750", "xrtData_qp_full")
    # ]

    models = [
        ("BasicNetworkFreqClass-RegressionClassificationLoss_joint_lp_noisy_1e4sch_logbins_20251028_191347", "xrtData_lp_logbins"),
        ("BasicNetworkFreqClass-RegressionClassificationLoss_joint_lp_noisy_1e4sch_sqrtbins_20251029_222912", "xrtData_lp_sqrtbins"),
    ]
    
    for model_name, input_path in models:
        process_model(model_name, input_path)

if __name__ == "__main__":
    main()