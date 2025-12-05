# ============================
# YOLO nano models
# ============================

yolo_v8n = YOLO("yolov8n.pt")
yolo_11n = YOLO("yolo11n.pt")
yolo_12n = YOLO("yolo12n.pt")

def make_yolo_predict_fn(yolo_model):
    def _predict(image_rgb):
        results = yolo_model(image_rgb, conf=0.25, verbose=False)
        det = sv.Detections.from_ultralytics(results[0])
        return det
    return _predict

# Evaluate all CNN nanos (full val set)
metrics_yolov8n = eval_model_on_visdrone(
    model_name="YOLOv8n",
    predict_fn=make_yolo_predict_fn(yolo_v8n),
    dataset=val_set,
    max_images=None,
)

metrics_yolo11n = eval_model_on_visdrone(
    model_name="YOLO11n",
    predict_fn=make_yolo_predict_fn(yolo_11n),
    dataset=val_set,
    max_images=None,
)

metrics_yolo12n = eval_model_on_visdrone(
    model_name="YOLO12n",
    predict_fn=make_yolo_predict_fn(yolo_12n),
    dataset=val_set,
    max_images=None,
)
