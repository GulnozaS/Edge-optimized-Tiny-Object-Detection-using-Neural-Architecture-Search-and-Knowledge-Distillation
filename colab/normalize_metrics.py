import numpy as np

def normalize_metrics(d: dict):
    def f(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        return x
    return {
        "Model": d["model"],
        "mAP50-95": f(d.get("map50_95", 0.0)),
        "mAP50":    f(d.get("map50", 0.0)),
        "mAP75":    f(d.get("map75", 0.0)),
        "FPS":      f(d.get("fps", 0.0)),
        "avg_time_s": f(d.get("avg_time_s", 0.0)),
    }

# CNN nanos
rows_cnns_nano = [
    normalize_metrics(metrics_yolov8n),
    normalize_metrics(metrics_yolo11n),
    normalize_metrics(metrics_yolo12n),
]
df_cnns_nano = pd.DataFrame(rows_cnns_nano)
print("=== CNN Nano models ===")
display(df_cnns_nano)

# Transformer nanos
rows_transformers_nano = [
    normalize_metrics(metrics_rfdetr_nano),
    normalize_metrics(metrics_rtdetr_nano),
    normalize_metrics(metrics_deformable_nano),
]
df_transformers_nano = pd.DataFrame(rows_transformers_nano)
print("=== Transformer Nano models ===")
display(df_transformers_nano)

# Combined
df_all_nano = pd.concat([df_cnns_nano, df_transformers_nano], ignore_index=True)
print("=== All Nano models (CNN + Transformer) ===")
display(df_all_nano)

# Save for later
df_cnns_nano.to_csv("cnns_nano_metrics.csv", index=False)
df_transformers_nano.to_csv("transformers_nano_metrics.csv", index=False)
df_all_nano.to_csv("all_nano_metrics.csv", index=False)
