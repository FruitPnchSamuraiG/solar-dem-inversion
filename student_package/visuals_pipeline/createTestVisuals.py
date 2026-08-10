"""
Create test visuals by processing all .npz files in a given folder.
"""

import os
import subprocess
import argparse
import re
import numpy as np

def process_folder(input_folder, output_folder, run_name=None, dem_folder=None):
    """
    Process all .npz files in the input folder by calling dumpVisuals.py for each file.
    Organize outputs in ../results/vis/run_name/timestamp/.
    """
    # Infer run_name if not provided
    if run_name is None:
        run_name = os.path.basename(os.path.normpath(os.path.dirname(input_folder)))
        # another level
        if run_name == "test":
            run_name = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(input_folder))))
        # if still empty, use a default name
        if not run_name:
            run_name = "default_run"

        # take out npz
        run_name = run_name.replace(".npz", "")
        print(f"Run name inferred as: {run_name}")

    # Ensure the output folder exists
    run_output_folder = os.path.join(output_folder, "vis", run_name)
    os.makedirs(run_output_folder, exist_ok=True)

    # Process each .npz file in the input folder
    # if input ends with npz, then it is a single file
    if input_folder.endswith(".npz"):
        npz_file = input_folder.split("/")[-1]
        print(f"Input is a single .npz file: {npz_file}")
        input_folder = os.path.dirname(input_folder)
        files = [npz_file]
    else:
        files = os.listdir(input_folder)
    for file_name in files:
        if file_name.endswith(".npz"):
            file_path = os.path.join(input_folder, file_name)
            timestamp_pattern = r"(\d{8}_\d{6})"
            match = re.search(timestamp_pattern, file_name)
            match2 = re.search(r"(\d{8}_\d{4})", file_name)
            if not match and match2:
                match = match2  
            timestamp = match.group(0) if match else file_name
            
            # normalize flare timestamps: add 00 seconds if format is YYYYMMDD_HHMM
            timestamp_parts = timestamp.split('_')
            if len(timestamp_parts) == 2 and len(timestamp_parts[1]) == 4:
                # HHMM format, add 00 seconds for web viewer compatibility
                timestamp = timestamp + "00"
            
            print(f"Found .npz file: {file_name} with timestamp: {timestamp}")
            timestamp_folder = os.path.join(run_output_folder, timestamp.replace(".npz", ""))
            os.makedirs(timestamp_folder, exist_ok=True)
            print(f"Processing file: {file_path}")

            # lp path to the comparison file
            # file name is <timestamp>.npz

            # first look in the dem_folder base
            folder_to_look = dem_folder if dem_folder else "/scratch/vp2435/workspace/dem/demdemo/data/out"
            ts_raw = timestamp.replace(".npz", "")
            # check if timestamp is contained in a file name of dem_folder
            print(f"Looking for comparison file in: {folder_to_look}")
            try:
                files_in_folder = os.listdir(folder_to_look)
                name_of_file = [f for f in files_in_folder if f.find(ts_raw)!=-1]
            except FileNotFoundError:
                print(f"Warning: DEM folder {folder_to_look} does not exist.")
                files_in_folder = []
                name_of_file = []
            print(name_of_file)
            if name_of_file:
                print(f"Found comparison file in {folder_to_look}")
                comparison_file = os.path.join(folder_to_look, name_of_file[0])
            else:
                # if not found, check in the dem_folder/train, val, test splits
                print(f"Comparison file not found in {folder_to_look}, checking splits...")
                for split in ["train", "val", "test"]:
                    folder_to_look = os.path.join(dem_folder, split) if dem_folder else os.path.join("/scratch/vp2435/workspace/dem/demdemo/data/out", split)
                    comparison_file = os.path.join(folder_to_look, timestamp)
                    if not os.path.exists(comparison_file):
                        continue
                    else: 
                        break
            print(f"Comparison file: {comparison_file}")
            # make sure the comparison file exists
            if not os.path.exists(comparison_file):
                print(f"Warning: Comparison file does not exist in any split: {comparison_file}")
                comparison_file = None
            else:
                print(f"Comparison file found: {comparison_file}")
            
            # check if input contains classification outputs
            input_data = np.load(file_path, mmap_mode='r')
            has_classification = 'ClassificationMaxProb' in input_data or 'ClassificationEntropy' in input_data
            
            # Call dumpVisuals.py for the current .npz file
            print(f"Running dumpVisuals.py with target {timestamp_folder}")
            cmd = [
                "python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumpVisuals.py"),
                "--dem_path", file_path,
                "--target", timestamp_folder,
                "--dem",
                "--aia_resynth",
                "--dem_jpdfs",
                "--dem_pixels",
                "--regions_jpdfs",
                "--roi_dems",
            ]
            if comparison_file is not None:
                cmd.extend(["--dem_comparison_path", comparison_file])
            if has_classification:
                print("  Classification outputs detected, adding uncertainty visualizations...")
                # cmd.append("--uncertainty")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    '''
    example:
    python createTestVisuals.py --input_folder ../preds/l1_barrier_lr-stable-1e-4
    '''
    parser = argparse.ArgumentParser(description="Process a folder of .npz files and create visuals.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing .npz files.")
    parser.add_argument("--output_folder", type=str, default="/scratch/vp2435/workspace/dem/demdemo/results", help="Path to the output folder.")
    parser.add_argument("--dem_folder", type=str, default="/scratch/vp2435/workspace/dem/demdemo/data/out", help="Path to the folder containing DEM comparison data.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for organizing outputs.")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.run_name, args.dem_folder)