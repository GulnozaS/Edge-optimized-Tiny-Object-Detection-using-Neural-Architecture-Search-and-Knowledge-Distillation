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
