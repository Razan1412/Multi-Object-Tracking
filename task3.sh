#!/bin/bash

# This script runs the entire Task 2 pipeline for the YOLOX-based tracker.
# The output of the final tracking step will be saved to task2results.txt.

echo "--- Starting Task 3: RF-DETR Tracking Pipeline ---"
echo ""

# --- Step 1: Run RF-DETR Detection ---
echo "--- [Step 1/3] Running RF-DETR detection... ---"
python 4.\ RF-DETR/detect_rfdetr.py
echo "--- RF-DETR detection complete. ---"
echo ""

# --- Step 2: Run FastReID Feature Extraction ---
echo "--- [Step 2/3] Running FastReID feature extraction (suppressing output)... ---"
cd 2.\ FastReID/
python rfdetr_ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/4. rfdet/mot17_val_0.80.pickle' --output_path '../outputs/5. rfdet_feat/mot17_val_0.80.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth' > /dev/null 2>&1
python rfdetr_ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/4. rfdet/mot17_val_0.95.pickle' --output_path '../outputs/5. rfdet_feat/mot17_val_0.95.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth' > /dev/null 2>&1
echo "--- Feature extraction complete. ---"
echo ""

# --- Step 3: Run the Tracker ---
echo "--- [Step 3/3] Running the tracker and saving results to task3results.txt... ---"
cd ../3.\ Tracker/


# Explicitly remove the old results file to ensure it is overwritten.
# The -f flag prevents errors if the file doesn't exist.
rm -f ../task3results.txt

# The output of this command will be saved to the results file in the project root.
python run.py --pickle_dir '../outputs/5. rfdet_feat/' --output_dir '../outputs/6. rfdet_track/' > ../task3results.txt 2>&1
echo ""

echo "--- Task 3 pipeline finished. The final evaluation results have been saved to task3results.txt ---"

