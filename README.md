```markdown
# E-TOD: Edge-Optimized Tiny Object Detection

## Project Description
This repository contains the implementation for our computer vision project on edge-optimized tiny object detection using Neural Architecture Search and Knowledge Distillation.

## Environment Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU (for training)
- NVIDIA Jetson Nano (for edge deployment)

### Installation
```bash
# Clone repository
git clone https://github.com/GulnozaS/Edge-optimized-Tiny-Object-Detection-using-Neural-Architecture-Search-and-Knowledge-Distillation.git
cd Edge-optimized-Tiny-Object-Detection-using-Neural-Architecture-Search-and-Knowledge-Distillation

# Create conda environment
conda create -n etod python=3.8
conda activate etod

# Install dependencies
pip install -r requirements.txt
```

### Required Packages (requirements.txt)
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.21.0
tqdm>=4.60.0
Pillow>=9.0.0
matplotlib>=3.5.0
tensorboard>=2.10.0
```

## Datasets

### VisDrone2021
- **Download Link**: [http://aiskyeye.com/download/object-detection-2/](http://aiskyeye.com/download/object-detection-2/)
- **License**: Academic Use
- **Setup**:
```bash
mkdir -p datasets/VisDrone
# Download and extract VisDrone dataset to datasets/VisDrone/
```

### Tiny Person
- **Download Link**: [https://github.com/ucas-vg/TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark)
- **License**: Academic Use

## Project Structure
```
├── configs/           # Configuration files
├── datasets/          # Dataset storage
├── models/            # Model architectures
├── scripts/           # Training and evaluation scripts
├── utils/             # Utility functions
├── requirements.txt   # Python dependencies
└── README.md
```

## Minimal Training Script Stub

```python
# scripts/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_teacher_model():
    """
    Stub for teacher model training
    """
    print("Teacher model training - To be implemented")
    # TODO: Implement YOLOX teacher training on VisDrone
    
def train_student_with_kd():
    """
    Stub for knowledge distillation training
    """
    print("Knowledge distillation training - To be implemented")
    # TODO: Implement NAS + KD training pipeline

if __name__ == "__main__":
    train_teacher_model()
```

## Minimal Evaluation Script Stub

```python
# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader

def evaluate_model(model, data_loader):
    """
    Stub for model evaluation
    """
    print("Model evaluation - To be implemented")
    # TODO: Implement mAP calculation and latency measurement
    
    # Placeholder metrics
    metrics = {
        'mAP@0.5:0.95': 0.0,
        'mAP_small': 0.0,
        'FPS': 0.0,
        'model_size': 0.0
    }
    return metrics

def benchmark_jetson(model):
    """
    Stub for Jetson Nano benchmarking
    """
    print("Jetson benchmarking - To be implemented")
    # TODO: Implement edge device performance measurement

if __name__ == "__main__":
    evaluate_model(None, None)
```

## NAS Search Script Stub

```python
# scripts/nas_search.py
import torch
import torch.nn as nn

def hardware_aware_nas():
    """
    Stub for hardware-aware neural architecture search
    """
    print("NAS search - To be implemented")
    # TODO: Implement evolutionary search with latency constraints
    
    search_space = {
        'operations': ['MBConv', 'DepthwiseConv'],
        'depths': [12, 16, 20],
        'widths': [32, 64, 128]
    }
    return search_space

if __name__ == "__main__":
    hardware_aware_nas()
```

## Usage Example
```python
# Basic usage example
from scripts.train import train_teacher_model
from scripts.evaluate import evaluate_model

# Train teacher model
train_teacher_model()

# Evaluate model (placeholder)
metrics = evaluate_model(None, None)
print(f"Model metrics: {metrics}")
```

## Team
- Gulnoza Sabirjonova (220278) - Coordinator
- Feruza Khudoyberdiyeva (220328)

## License
MIT License
```
