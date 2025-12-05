from rfdetr import RFDETRNano

num_classes = 10
model = RFDETRNano(num_classes=num_classes)

import torch

# 1. Grab the inner torch model
print("outer type:", type(model))
print("wrapper type:", type(model.model))
print("core type:", type(model.model.model))

core = model.model.model  # this is the nn.Module

# 2. Freeze ALL layers first
for p in core.parameters():
    p.requires_grad = False

# 3. Inspect what heads exist
print([a for a in dir(core) if "embed" in a or "head" in a])

# 4. Unfreeze classification + bbox heads
# For DETR-style RF-DETR these are typically named class_embed and bbox_embed
for p in core.class_embed.parameters():
    p.requires_grad = True

for p in core.bbox_embed.parameters():
    p.requires_grad = True

from rfdetr import RFDETRNano
import torch

num_classes = 10
model = RFDETRNano(num_classes=num_classes)

# --- get inner nn.Module (you already printed these types) ---
core = model.model.model   # LWDETR

# 1. Freeze ALL layers
for p in core.parameters():
    p.requires_grad = False

# 2. Unfreeze ONLY detection heads
#    class_embed = classification head
#    bbox_embed  = box regression head
for p in core.class_embed.parameters():
    p.requires_grad = True

for p in core.bbox_embed.parameters():
    p.requires_grad = True

# (optional) sanity check: count trainable vs frozen
total = sum(p.numel() for p in core.parameters())
trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
print(f"Total params in core: {total:,}")
print(f"Trainable params (heads only): {trainable:,}")

import os
import json

ROOT = "/content/visdrone_coco_rfdetr"

# create test folder
os.makedirs(f"{ROOT}/test/images", exist_ok=True)

# create empty annotation file
empty_json = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "pedestrian"},
        {"id": 1, "name": "people"},
        {"id": 2, "name": "bicycle"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "van"},
        {"id": 5, "name": "truck"},
        {"id": 6, "name": "tricycle"},
        {"id": 7, "name": "awning-tricycle"},
        {"id": 8, "name": "bus"},
        {"id": 9, "name": "motor"}
    ]
}

with open(f"{ROOT}/test/_annotations.coco.json", "w") as f:
    json.dump(empty_json, f)

import json
import os

COCO_ROOT = "/content/visdrone_coco_rfdetr"

def patch_coco_json(path):
    print("Patching:", path)
    with open(path, "r") as f:
        data = json.load(f)

    if "info" not in data:
        data["info"] = {
            "description": "VisDrone for RF-DETR",
            "version": "1.0"
        }

    if "licenses" not in data:
        data["licenses"] = []  # empty list is fine

    with open(path, "w") as f:
        json.dump(data, f)
    print("  -> done")

for split in ["train", "valid", "test"]:
    json_path = os.path.join(COCO_ROOT, split, "_annotations.coco.json")
    patch_coco_json(json_path)

OUTPUT_DIR = "/content/rfdetr_last_layer"

model.train(
    dataset_dir="/content/visdrone_coco_rfdetr",  # your COCO root with train/valid
    epochs=5,
    batch_size=2,
    grad_accum_steps=8,
    lr=1e-4,
    output_dir=OUTPUT_DIR,
    early_stopping=True,
    tensorboard=False,
)

best_ckpt = f"{OUTPUT_DIR}/checkpoint_best_total.pth"

model_ft = RFDETRNano(
    num_classes=num_classes,
    pretrain_weights=best_ckpt,
)

def rfdetr_ft_predict(image_rgb):
    det = model_ft.predict(image_rgb, threshold=0.3)
    return det  # already supervision.Detections in your version

metrics_rfdetr_last_layer = eval_model_on_visdrone(
    model_name="RF-DETR-Nano-HeadFT",
    predict_fn=rfdetr_ft_predict,
    dataset=val_set,
    max_images=None,
)

metrics_rfdetr_last_layer

