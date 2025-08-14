#!/bin/bash

# This script runs the entire Task 2 pipeline for the YOLOX-based tracker.
# The output of the final tracking step will be saved to task2results.txt.

echo "--- Starting Task 2: YOLOX Tracking Pipeline ---"
echo ""

# --- Step 1: Run YOLOX Detection ---
echo "--- [Step 1/3] Running YOLOX detection... ---"
cd 1.\ YOLOX/
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.80 -n "../outputs/1. det/mot17_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.95 -n "../outputs/1. det/mot17_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse
echo "--- YOLOX detection complete. ---"
echo ""

# --- Step 2: Run FastReID Feature Extraction ---
echo "--- [Step 2/3] Running FastReID feature extraction (suppressing output)... ---"
cd ../2.\ FastReID/
python ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/1. det/mot17_val_0.80.pickle' --output_path '../outputs/2. det_feat/mot17_val_0.80.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth' > /dev/null 2>&1
python ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/1. det/mot17_val_0.95.pickle' --output_path '../outputs/2. det_feat/mot17_val_0.95.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth' > /dev/null 2>&1
echo "--- Feature extraction complete. ---"
echo ""

# --- Step 3: Run the Tracker ---
echo "--- [Step 3/3] Running the tracker and saving results to task2results.txt... ---"
cd ../3.\ Tracker/


# Explicitly remove the old results file to ensure it is overwritten.
# The -f flag prevents errors if the file doesn't exist.
rm -f ../task2results.txt

# The output of this command will be saved to the results file in the project root.
python run.py --dataset "MOT17" --mode "val" > ../task2results.txt 2>&1
echo ""

echo "--- Task 2 pipeline finished. The final evaluation results have been saved to task2results.txt ---"

