import os
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import the high-level RFDETR class from the library
from rfdetr import RFDETRBase

def make_parser():
    """Creates the argument parser for the detection script."""
    parser = argparse.ArgumentParser("RF-DETR Detection Script for TrackTrack")

    # Parser arguments
    parser.add_argument("--weights", type=str,
                        default="4. RF-DETR/rf-detr-base.pth",
                        help="Path to the local RF-DETR model weights (if supported by the library).")
    parser.add_argument("--json_path", type=str,
                        default="4. RF-DETR/jsons/mot17_val.json",
                        help="Path to the MOT17 validation JSON file.")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                        help="Root directory of the 'Multi-Object-Tracking' project.")
    parser.add_argument("--output_dir", type=str, default="outputs/4. det",
                        help="Directory to save the output pickle files.")
    parser.add_argument("--model_size", type=str, default="medium",
                        help="Size of the RF-DETR model to load ('base', 'medium', etc.).")
    parser.add_argument("--conf_thresholds", type=float, nargs='+',
                        default=[0.80, 0.95],
                        help="A list of confidence thresholds to apply for filtering.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on ('cuda' or 'cpu').")

    return parser

def main(args):
    """Main detection and saving logic."""
    print("Initializing RF-DETR detection script with the following arguments:")
    print(args)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Model Loading
    print(f"Loading RF-DETR model (size: {args.model_size})...")
    model = RFDETRBase(size=args.model_size)
    print("Model loaded successfully.")

    # Data Preparation
    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    images_info = data['images']
    results = {f"thresh_{thresh}": {} for thresh in args.conf_thresholds}
    for thresh in args.conf_thresholds:
        for img_info in images_info:
            video_name = img_info['file_name'].split('/')[2]
            if video_name not in results[f"thresh_{thresh}"]:
                results[f"thresh_{thresh}"][video_name] = {}

    # Inference Loop
    print(f"Running inference on {len(images_info)} images...")
    for img_info in tqdm(images_info, desc="Processing images"):
        img_path = os.path.join(args.dataset_root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        predictions = model.predict(image)

        # Filter and Format detections
        for thresh in args.conf_thresholds:
            keep = predictions.confidence > thresh
            
            filtered_xyxy = predictions.xyxy[keep]
            
            # Clip boxes to image boundaries to prevent invalid crops
            if filtered_xyxy.shape[0] > 0:
                filtered_xyxy[:, 0] = np.maximum(0, filtered_xyxy[:, 0])
                filtered_xyxy[:, 1] = np.maximum(0, filtered_xyxy[:, 1])
                filtered_xyxy[:, 2] = np.minimum(img_w, filtered_xyxy[:, 2])
                filtered_xyxy[:, 3] = np.minimum(img_h, filtered_xyxy[:, 3])

            # Filter out boxes with zero or negative width/height after clipping
            valid_boxes_mask = (filtered_xyxy[:, 2] > filtered_xyxy[:, 0]) & (filtered_xyxy[:, 3] > filtered_xyxy[:, 1])
            
            # Always define detections_array. If there are valid boxes, create the array.
            # Otherwise, create an empty array. This prevents KeyErrors downstream.
            if np.any(valid_boxes_mask):
                final_boxes = filtered_xyxy[valid_boxes_mask]
                final_scores = predictions.confidence[keep][valid_boxes_mask]
                final_classes = predictions.class_id[keep][valid_boxes_mask]

                detections_array = np.hstack([
                    final_boxes,
                    final_scores[:, np.newaxis],
                    final_classes[:, np.newaxis]
                ])
            else:
                # If no valid boxes, create an empty array for this frame
                detections_array = np.empty((0, 6))

            # Store the result for every frame
            video_name = img_info['file_name'].split('/')[2]
            frame_id = img_info['frame_id']
            results[f"thresh_{thresh}"][video_name][frame_id] = detections_array

    # Save Outputs 
    print("Saving detection files...")
    for thresh in args.conf_thresholds:
        output_filename = f"mot17_val_{thresh:.2f}.pickle"
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results[f"thresh_{thresh}"], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> Saved detections to {output_path}")

    print("\nDetection complete. The output files are ready for the FastReID step.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
