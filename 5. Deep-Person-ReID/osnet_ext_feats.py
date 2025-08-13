import os
import cv2
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image 

# Import torchreid components
import torchreid
from torchreid.data.transforms import build_transforms

def make_parser():
    """Creates the argument parser for the OSNet feature extraction script."""
    parser = argparse.ArgumentParser("OSNet Feature Extraction")

    # Data Arguments
    parser.add_argument("--data_path", type=str, default="../dataset/MOT17/train/", help="Path to the training data images.")
    parser.add_argument("--pickle_path", type=str, default="../outputs/4. det/mot17_val_0.80.pickle", help="Path to the input detection pickle file from RF-DETR.")
    parser.add_argument("--output_path", type=str, default="../outputs/7. osnet_feat/mot17_val_0.80.pickle", help="Path to save the output pickle file with OSNet features.")
    
    # Model Arguments
    parser.add_argument("--model_name", type=str, default="osnet_ain_x0_25", help="Name of the Re-ID model to use from torchreid.")
    
    # System Arguments
    parser.add_argument("--seed", type=float, default=10000, help="Random seed for reproducibility.")

    return parser

def extract_features(model, input_patches, device):
    """
    Extracts features from a batch of image patches.
    
    Args:
        model: The loaded torchreid model.
        input_patches (torch.Tensor): A batch of pre-processed image patches.
        device (torch.device): The device to run inference on.

    Returns:
        numpy.ndarray: A numpy array of the extracted features.
    """
    with torch.no_grad():
        # Move patches to the correct device
        input_patches = input_patches.to(device)
        # Run inference
        features = model(input_patches)
        # L2-normalize the features, a common practice in Re-ID
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        # Move features to CPU and convert to numpy array
        return features.cpu().numpy()

if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Re-ID Model
    print(f"Initializing torchreid model: {args.model_name}...")
    model = torchreid.models.build_model(
        name=args.model_name,
        num_classes=1, # num_classes is not used for feature extraction but required
        loss='softmax',
        pretrained=True # This will automatically download weights if not cached
    )
    model.eval()
    model.to(device)
    
    # Define Image Transformations
    _, test_transforms = build_transforms(
        height=256,
        width=128,
        transforms='imagenet', # Use standard ImageNet normalization
        is_train=False
    )

    # Load Detections 
    print(f"Loading detections from: {args.pickle_path}")
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Main Feature Extraction Loop 
    print("Extracting features using OSNet...")
    for vid_name in tqdm(detections.keys(), desc="Processing Videos"):
        for frame_id in detections[vid_name].keys():
            detection = detections[vid_name][frame_id]

            if detection.shape[0] == 0:
                # If no detections, ensure the entry is an empty array with the correct future shape
                detections[vid_name][frame_id] = np.empty((0, 6 + 512)) # 6 from detection + 512 from OSNet
                continue

            # Read the corresponding image file
            img_path = os.path.join(args.data_path, vid_name, 'img1', f'{frame_id:06d}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert image from BGR (OpenCV) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop and Process Patches
            all_crops = []
            for box in detection[:, :4]: # Bounding boxes are in xyxy format
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                
                # Convert the numpy array crop to a PIL Image before transforming
                crop_pil = Image.fromarray(crop)
                
                # Apply transformations using the correct test_transforms variable
                crop_transformed = test_transforms(crop_pil)
                all_crops.append(crop_transformed)

            if not all_crops:
                detections[vid_name][frame_id] = np.empty((0, 6 + 512))
                continue

            # Stack all transformed crops into a single tensor
            batch_crops = torch.stack(all_crops)
            
            # Extract features for the batch
            embeddings = extract_features(model, batch_crops, device)
            
            # Concatenate the original detection data with the new embeddings
            detections[vid_name][frame_id] = np.concatenate([detection, embeddings], axis=1)

    # Save Final Results 
    print(f"Saved feature extractions to: {args.output_path}\n")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("OSNet feature extraction complete.")
