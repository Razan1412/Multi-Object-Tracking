import os
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
from torch.jit import TracerWarning

from rfdetr import RFDETRBase

def make_parser():
    parser = argparse.ArgumentParser("RF-DETR Detection Script for TrackTrack")
    parser.add_argument("--weights", type=str,
                        default="4. RF-DETR/rf-detr-base.pth")
    parser.add_argument("--json_path", type=str,
                        default="4. RF-DETR/jsons/mot17_val.json")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/4. rfdet")
    parser.add_argument("--model_size", type=str, default="medium")
    parser.add_argument("--conf_thresholds", type=float, nargs='+',
                        default=[0.80, 0.95])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,  # Added batch inference
                        help="Number of images per inference batch")
    return parser

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading RF-DETR Base...")
    model = RFDETRBase()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TracerWarning)
        model.optimize_for_inference(batch_size=args.batch_size)
    print("Model loaded successfully.")

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

    print(f"Running inference on {len(images_info)} images (batch size {args.batch_size})...")
    batch_images = []
    batch_meta = []

    for img_info in tqdm(images_info, desc="Processing images"):
        img_path = os.path.join(args.dataset_root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        batch_images.append(image)
        batch_meta.append((img_info, image.size))

        # If batch is full, process it
        if len(batch_images) == args.batch_size:
            process_batch(model, batch_images, batch_meta, results, args)
            batch_images.clear()
            batch_meta.clear()

    # Process any leftover images
    if batch_images:
        # Pad to match args.batch_size
        original_len = len(batch_images)
        while len(batch_images) < args.batch_size:
            batch_images.append(batch_images[-1])
            batch_meta.append(batch_meta[-1])

        process_batch(model, batch_images, batch_meta, results, args, trim_to=original_len)


    print("Saving detection files...")
    for thresh in args.conf_thresholds:
        output_filename = f"mot17_val_{thresh:.2f}.pickle"
        output_path = os.path.join(args.output_dir, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(results[f"thresh_{thresh}"], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> Saved detections to {output_path}")

    print("\nDetection complete. The output files are ready for the FastReID step.")

def process_batch(model, images, meta, results, args, trim_to=None):
    predictions_list = model.predict(images)  # Batch predict

    # If we padded, trim the predictions and metadata back
    if trim_to is not None:
        predictions_list = predictions_list[:trim_to]
        meta = meta[:trim_to]

    for pred, (img_info, (img_w, img_h)) in zip(predictions_list, meta):
        for thresh in args.conf_thresholds:
            keep = pred.confidence > thresh
            filtered_xyxy = pred.xyxy[keep]

            if filtered_xyxy.shape[0] > 0:
                filtered_xyxy[:, 0] = np.maximum(0, filtered_xyxy[:, 0])
                filtered_xyxy[:, 1] = np.maximum(0, filtered_xyxy[:, 1])
                filtered_xyxy[:, 2] = np.minimum(img_w, filtered_xyxy[:, 2])
                filtered_xyxy[:, 3] = np.minimum(img_h, filtered_xyxy[:, 3])

            valid_boxes_mask = (filtered_xyxy[:, 2] > filtered_xyxy[:, 0]) & \
                               (filtered_xyxy[:, 3] > filtered_xyxy[:, 1])

            if np.any(valid_boxes_mask):
                final_boxes = filtered_xyxy[valid_boxes_mask]
                final_scores = pred.confidence[keep][valid_boxes_mask]
                final_classes = pred.class_id[keep][valid_boxes_mask]
                detections_array = np.hstack([
                    final_boxes,
                    final_scores[:, np.newaxis],
                    final_classes[:, np.newaxis]
                ])
            else:
                detections_array = np.empty((0, 6))

            video_name = img_info['file_name'].split('/')[2]
            frame_id = img_info['frame_id']
            results[f"thresh_{thresh}"][video_name][frame_id] = detections_array


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
