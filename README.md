# Edge-Optimized Tiny Object Detection

## 🚀 Quick Start

### Environment Setup
```bash
# Create and activate environment
conda create -n edge-detection python=3.8 -y
conda activate edge-detection

# Install dependencies
pip install torch torchvision opencv-python numpy pillow
```

### Dataset Links
- **VisDrone Dataset**: http://aiskyeye.com/download/object-detection-2/
- **Tiny Person Dataset**: https://github.com/ucas-vg/TinyBenchmark

## 📁 Project Structure
```
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 💻 Script Stubs

### Training Script
```python
# train.py
def main():
    """Training pipeline for edge-optimized object detection"""
    print("Training pipeline - to be implemented")
    # TODO: Implement NAS + Knowledge Distillation training

if __name__ == "__main__":
    main()
```

### Evaluation Script
```python
# evaluate.py
def main():
    """Evaluation pipeline for model performance"""
    print("Evaluation pipeline - to be implemented")
    # TODO: Implement mAP and latency evaluation

if __name__ == "__main__":
    main()
```

## 👥 Team
- Gulnoza Sabirjonova (220278)
- Feruza Khudoyberdiyeva (221328)
```
