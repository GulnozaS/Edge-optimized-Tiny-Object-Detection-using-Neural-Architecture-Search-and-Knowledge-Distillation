# ============================
# RF-DETR Nano
# ============================

rfdetr_nano = RFDETRNano()  # COCO-pretrained

def rfdetr_predict(image_rgb):
    """
    RF-DETR's .predict() usually accepts RGB ndarray and returns a supervision.Detections
    or similar. If your version returns something else, adapt this function.
    """
    detections = rfdetr_nano.predict(image_rgb, threshold=0.3)

    # If already sv.Detections, just return
    if isinstance(detections, sv.Detections):
        return detections

    # Otherwise assume list of dicts
    # with keys: 'xyxy', 'confidence', 'class_id'
    if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
        xyxy = np.array([d["xyxy"] for d in detections], dtype=np.float32)
        conf = np.array([d.get("confidence", 1.0) for d in detections], dtype=np.float32)
        cid  = np.array([d.get("class_id", 0) for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid)

    # Fallback: empty
    return sv.Detections.empty()

metrics_rfdetr_nano = eval_model_on_visdrone(
    model_name="RF-DETR-Nano",
    predict_fn=rfdetr_predict,
    dataset=val_set,
    max_images=None,
)


# ============================
# RT-DETR Nano (or L)
# ============================

try:
    rtdetr_nano = RTDETR("rtdetr-n.pt")
    rtdetr_name = "RT-DETR-Nano"
except Exception:
    rtdetr_nano = RTDETR("rtdetr-l.pt")
    rtdetr_name = "RT-DETR-L"   # name will show in results

def rtdetr_predict(image_rgb):
    results = rtdetr_nano(image_rgb, conf=0.25, verbose=False)
    det = sv.Detections.from_ultralytics(results[0])
    return det

metrics_rtdetr_nano = eval_model_on_visdrone(
    model_name=rtdetr_name,
    predict_fn=rtdetr_predict,
    dataset=val_set,
    max_images=None,
)
