# ============================
# Deformable DETR (HF)
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deform_ckpt = "SenseTime/deformable-detr"
deform_proc = DeformableDetrImageProcessor.from_pretrained(deform_ckpt)
deform_model = DeformableDetrForObjectDetection.from_pretrained(deform_ckpt).to(device)
deform_model.eval()

def deformable_detr_predict(image_rgb, score_threshold: float = 0.3):
    h, w = image_rgb.shape[:2]

    inputs = deform_proc(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = deform_model(**inputs)

    logits = outputs.logits[0]       # [Q, C+1]
    boxes  = outputs.pred_boxes[0]   # [Q, 4] cx,cy,w,h normalized

    probs = logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)  # ignore background

    keep = scores > score_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes  = boxes[keep]

    if boxes.numel() == 0:
        return sv.Detections.empty()

    boxes  = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    cx = boxes[:, 0] * w
    cy = boxes[:, 1] * h
    bw = boxes[:, 2] * w
    bh = boxes[:, 3] * h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    xyxy = np.stack([x1, y1, x2, y2], axis=-1)

    det = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=labels.astype(int),
    )
    return det

metrics_deformable_nano = eval_model_on_visdrone(
    model_name="Deformable-DETR",
    predict_fn=deformable_detr_predict,
    dataset=val_set,
    max_images=None,
)
