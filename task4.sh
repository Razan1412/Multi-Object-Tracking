#!/bin/bash

# This script runs the entire Task 4 pipeline for the RF-DETR-based tracker with OSNet Re-ID.
# The output of the final tracking step will be saved to task4results.txt.

echo "--- Starting Task 4: RF-DETR + OSNet Tracking Pipeline ---"
echo ""

# --- Step 1: Run RF-DETR Detection ---
echo "--- [Step 1/3] Running RF-DETR detection... ---"
python 4.\ RF-DETR/detect_rfdetr.py
echo "--- RF-DETR detection complete. ---"
echo ""

# --- Step 2: Run OSNet Feature Extraction ---
echo "--- [Step 2/3] Running OSNet feature extraction ... ---"
cd 5.\ Deep-Person-ReID/
python osnet_ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/4. rfdet/mot17_val_0.80.pickle' --output_path '../outputs/7. osnet_rfdet_feat/mot17_val_0.80.pickle'
python osnet_ext_feats.py --data_path '../dataset/MOT17/train/' --pickle_path '../outputs/4. rfdet/mot17_val_0.95.pickle' --output_path '../outputs/7. osnet_rfdet_feat/mot17_val_0.95.pickle' 
echo "--- Feature extraction complete. ---"
echo ""

# --- Step 3: Run the Tracker ---
echo "--- [Step 3/3] Running the tracker and saving results to task4results.txt... ---"
cd ../3.\ Tracker/

# Explicitly remove the old results file to ensure it is overwritten.
# The -f flag prevents errors if the file doesn't exist.
rm -f ../task4results.txt

# The output of this command will be saved to the results file
python run.py --pickle_dir '../outputs/7. osnet_rfdet_feat/' --output_dir '../outputs/8. osnet_rfdet_track/' > ../task4results.txt 2>&1
echo ""

echo "--- Task 4 pipeline finished. The final evaluation results have been saved to task4results.txt ---"
