# ----------------------this file is needed only for colab setup
#---------google colab link: https://colab.research.google.com/drive/1BC35qjr6julr71gmu9KCN4n9jmYKMJVt?usp=sharing

#!pip install -q ultralytics supervision "transformers[torch]" rfdetr opencv-python
#from google.colab import drive
#drive.mount('/content/drive')

# import os

# # Google Drive folder where your ZIPs are stored
# zip_base = "/content/drive/MyDrive/visdrone_dataset"

# train_zip = f"{zip_base}/VisDrone2019-DET-train.zip"
# val_zip   = f"{zip_base}/VisDrone2019-DET-val.zip"
# test_zip  = f"{zip_base}/VisDrone2019-DET-test-dev.zip"

# # Output folders
# os.makedirs("/content/VisDrone/train", exist_ok=True)
# os.makedirs("/content/VisDrone/val", exist_ok=True)
# os.makedirs("/content/VisDrone/test", exist_ok=True)

# # Unzip
# !unzip -q "{train_zip}" -d "/content/VisDrone/train"
# !unzip -q "{val_zip}"   -d "/content/VisDrone/val"
# !unzip -q "{test_zip}"  -d "/content/VisDrone/test"

# print("Done unzipping!")

#---Rf detr fine tuning setup!pip install -q rfdetr supervision opencv-python

# import os
# import glob
# import json
# import cv2
# import numpy as np
# from pathlib import Path

# import supervision as sv
# from rfdetr import RFDETRNano

# Raw VisDrone (original structure)
# RAW_ROOT = "content/VisDrone"

# TRAIN_IMAGES = "/content/VisDrone/train/VisDrone2019-DET-train/images"
# TRAIN_ANN    = "/content/VisDrone/train/VisDrone2019-DET-train/annotations"

# VAL_IMAGES   = "/content/VisDrone/val/VisDrone2019-DET-val/images"
# VAL_ANN      = "/content/VisDrone/val/VisDrone2019-DET-val/annotations"

# # Target COCO-like directory for RF-DETR
# COCO_ROOT = "/content/visdrone_coco_rfdetr"

# for split in ["train", "valid"]:
#     os.makedirs(os.path.join(COCO_ROOT, split, "images"), exist_ok=True)
