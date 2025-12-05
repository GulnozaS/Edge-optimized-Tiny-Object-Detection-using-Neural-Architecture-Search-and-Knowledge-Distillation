def eval_model_on_visdrone(
    model_name: str,
    predict_fn,
    dataset,
    max_images: int | None = None,
):
    """
    model_name: just for printing
    predict_fn: function(image_rgb: np.ndarray) -> sv.Detections
    dataset: VisDroneValDataset
    """
    n = len(dataset)
    print(f"[{model_name}] Dataset size: {n}")

    metric = MeanAveragePrecision()
    total_time = 0.0
    count = 0

    for i in range(n):
        if max_images is not None and i >= max_images:
            break

        _, image_bgr, target = dataset[i]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        det = predict_fn(image_rgb)
        end = time.perf_counter()

        metric.update(predictions=det, targets=target)

        total_time += (end - start)
        count += 1

        if (i + 1) % 50 == 0:
            print(f"[{model_name}] Processed {i+1} images...")

    if count == 0:
        print(f"[{model_name}] No images processed.")
        return {
            "model": model_name,
            "avg_time_s": 0.0,
            "fps": 0.0,
            "map50_95": 0.0,
            "map50": 0.0,
            "map75": 0.0,
        }

    res = metric.compute()
    avg_time = total_time / count
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"\n===== {model_name} on VisDrone (ORIGINAL LABELS) =====")
    print(f"Images evaluated: {count}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"mAP50-95: {res.map50_95:.4f}")
    print(f"mAP50:    {res.map50:.4f}")
    print(f"mAP75:    {res.map75:.4f}")

    return {
        "model": model_name,
        "avg_time_s": avg_time,
        "fps": fps,
        "map50_95": res.map50_95,
        "map50": res.map50,
        "map75": res.map75,
    }
