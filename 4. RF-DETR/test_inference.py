from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv
import numpy as np

# COCO class list
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load RF-DETR Medium
print("Loading pretrain weights")
model = RFDETRBase(size="medium")

# Load a test image
img_path = "test.jpg"
image = Image.open(img_path).convert("RGB")
image_np = np.array(image)

# Run inference
results = model.predict(image)

# Create labels
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(results.class_id, results.confidence)
]

# Draw bounding boxes
box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=results)

# Draw labels
label_annotator = sv.LabelAnnotator()
annotated_image = label_annotator.annotate(scene=annotated_image, labels=labels, detections=results)

# Save result
Image.fromarray(annotated_image).save("annotated_test.jpg")
print("Annotated image saved to annotated_test.jpg")
