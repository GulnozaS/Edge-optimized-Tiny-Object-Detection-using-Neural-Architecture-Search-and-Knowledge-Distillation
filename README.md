````markdown
# Edge-optimized Tiny Object Detection using Neural Architecture Search and Knowledge Distillation

### Team Members
- **Coordinator:** Gulnoza Sabirjonova (220278@centralasian.uz)  
- **Co-member:** Feruza Khudoyberdiyeva (220328@centralasian.uz)

---

## 📘 Project Overview

This project explores **edge-optimized object detection** by combining **Neural Architecture Search (NAS)** and **Knowledge Distillation (KD)** to create a **lightweight yet accurate model** for detecting small objects in constrained environments such as drones, IoT devices, and embedded systems.

Our measurable goal:
> Achieve **>10% relative improvement in mAP@0.5 for small objects** compared to YOLOv5s, while maintaining **>25 FPS on NVIDIA Jetson Nano**.

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 1.13  
- CUDA Toolkit (for GPU training)  
- Git, pip, and virtualenv

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/GulnozaS/Edge-optimized-Tiny-Object-Detection-using-Neural-Architecture-Search-and-Knowledge-Distillation.git
cd Edge-optimized-Tiny-Object-Detection-using-Neural-Architecture-Search-and-Knowledge-Distillation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
````

### 3. Example `requirements.txt`

```txt
torch>=1.13.0
torchvision>=0.14.0
numpy
pandas
opencv-python
matplotlib
tqdm
pyyaml
```

---

## 🧠 Dataset Links

| Dataset                             | Description                                     | Link                                                                                 |
| ----------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| **COCO (Subset for Small Objects)** | Widely used object detection benchmark          | [https://cocodataset.org](https://cocodataset.org)                                   |
| **VisDrone 2019**                   | Drone-based object detection with small targets | [http://aiskyeye.com](http://aiskyeye.com)                                           |
| **TinyPerson**                      | Human detection for small-scale images          | [https://github.com/ucas-vg/TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark) |

---

## 🚀 Minimal Training Script Stub

```python
# train.py
import torch
from models import TinyNASDetector
from data import get_dataloader

def train():
    model = TinyNASDetector()
    train_loader, val_loader = get_dataloader(batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  # Placeholder
    
    for epoch in range(10):
        model.train()
        for imgs, targets in train_loader:
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

if __name__ == "__main__":
    train()
```

---

## 🧪 Minimal Evaluation Script Stub

```python
# eval.py
import torch
from models import TinyNASDetector
from data import get_dataloader
from metrics import calculate_map

def evaluate():
    model = TinyNASDetector()
    model.load_state_dict(torch.load("weights/best_model.pt"))
    model.eval()
    
    _, val_loader = get_dataloader(batch_size=8)
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            preds = model(imgs)
            all_preds.append(preds)
            all_targets.append(targets)
    
    mAP = calculate_map(all_preds, all_targets)
    print(f"Validation mAP@0.5: {mAP:.3f}")

if __name__ == "__main__":
    evaluate()
```

---

## 🗺️ Roadmap & Milestones

A detailed week-by-week roadmap with deliverables and owners is available in [`ROADMAP.md`](./ROADMAP.md).

---

## 💡 Citation

If you use or build upon this project, please cite it as:

```
Sabirjonova, G., & Khudoyberdiyeva, F. (2025). Edge-optimized Tiny Object Detection using Neural Architecture Search and Knowledge Distillation. Central Asian University, Computer Vision Course Project.
```

---

## 📄 License

This repository is released under the **MIT License**.

---

## 🤝 Acknowledgements

* YOLOv5 by Ultralytics
* TinyNAS (Google Research)
* PyTorch Lightning community tutorials
* NVIDIA Jetson Nano open benchmarks

```
```
