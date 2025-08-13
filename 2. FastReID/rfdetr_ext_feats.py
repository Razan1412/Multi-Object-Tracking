import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer
from tqdm import tqdm


def make_parser():
    """Creates the argument parser for the feature extraction script."""
    parser = argparse.ArgumentParser("RF-DETR Feature Extraction")

    # --- Data Arguments ---
    parser.add_argument("--dataset", type=str, default="mot17", help="Dataset name, e.g., 'MOT17'.")
    parser.add_argument("--data_path", type=str, default="../dataset/MOT17/train/", help="Path to the training data images.")
    parser.add_argument("--pickle_path", type=str, default="../outputs/4. det/mot17_val_0.80.pickle", help="Path to the input detection pickle file from RF-DETR.")
    parser.add_argument("--output_path", type=str, default="../outputs/5. rfdet_feat/mot17_val_0.80.pickle", help="Path to save the output pickle file with features.")
    
    # --- Model Arguments ---
    parser.add_argument("--config_path", type=str, default="configs/MOT17_half/sbs_S50.yml", help="Path to the FastReID model config file.")
    parser.add_argument("--weight_path", type=str, default="weights/mot17_half_sbs_S50.pth", help="Path to the FastReID model weights.")

    # --- System Arguments ---
    parser.add_argument("--seed", type=float, default=10000, help="Random seed for reproducibility.")

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Initialize the feature extractor
    print("Initializing FastReID embedder...")
    embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)

    # Load the detections from the RF-DETR script
    print(f"Loading detections from: {args.pickle_path}")
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # --- Main Feature Extraction Loop ---
    print("Extracting features for each detection...")
    for vid_name in tqdm(detections.keys(), desc="Processing Videos"):
        for frame_id in detections[vid_name].keys():
            # Get the detection array for the current frame
            detection = detections[vid_name][frame_id]

            # *** MINIMAL FIX APPLIED HERE ***
            # Check if the detection array is empty (shape[0] == 0).
            # If it is, skip this frame entirely to avoid errors.
            if detection.shape[0] == 0:
                continue

            # Read the corresponding image file
            if 'MOT' in args.data_path:
                img_path = os.path.join(args.data_path, vid_name, 'img1', f'{frame_id:06d}.jpg')
            else:
                img_path = os.path.join(args.data_path, vid_name, 'img1', f'{frame_id:08d}.jpg')
            
            img = cv2.imread(img_path)

            # If the image is loaded successfully and there are detections, compute features
            if img is not None:
                # The embedder expects bounding boxes in [x1, y1, x2, y2] format.
                # Our detect_rfdetr.py script already provides this.
                embedding = embedder.compute_embedding(img, detection[:, :4])
                
                # Concatenate the original [N, 6] detection array with the [N, 2048] embedding array
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

    # Save the final dictionary with appended features
    print(f"Saving feature-augmented detections to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nFeature extraction complete.")
