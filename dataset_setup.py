import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import torch

import supervision as sv
from supervision.metrics.mean_average_precision import MeanAveragePrecision

from ultralytics import YOLO, RTDETR
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from rfdetr import RFDETRNano

# ============================
# 1) VisDrone validation dataset
# ============================

class VisDroneValDataset:
    """
    Iterates over VisDrone val images + original annotation txt files.

    Yields:
      (image_path: str, image_bgr: np.ndarray, target: sv.Detections)
    """
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        img_paths = []
        for e in exts:
            img_paths.extend(glob.glob(os.path.join(images_dir, e)))
        self.image_paths = sorted(img_paths)
        print(f"Found {len(self.image_paths)} images in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.labels_dir, base + ".txt")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        boxes = []
        class_ids = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    x, y, w, h = map(float, parts[0:4])
                    cat_id = int(parts[5])

                    # ignore ignored regions
                    if cat_id <= 0:
                        continue

                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(cat_id - 1)  # 1..10 -> 0..9

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.array([], dtype=int)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=int)

        target = sv.Detections(
            xyxy=boxes,
            class_id=class_ids
        )
        return image_path, image_bgr, target


# ============================
# 2) CHANGE THESE PATHS
# ============================

VIS_VAL_IMAGES = "/content/VisDrone/val/VisDrone2019-DET-val/images"
VIS_VAL_LABELS = "/content/VisDrone/val/VisDrone2019-DET-val/annotations"

val_set = VisDroneValDataset(VIS_VAL_IMAGES, VIS_VAL_LABELS)
print("len(val_set):", len(val_set))

