# explore_visdrone.py
import os
import cv2
import random
from ultralytics import YOLO

def explore_visdrone():
    print("🔍 EXPLORING VISDRONE DATASET")
    print("=" * 40)
    
    # Check if dataset is organized
    if not os.path.exists("data/visdrone/images/train"):
        print("❌ Dataset not organized. Run organize_visdrone.py first!")
        return
    
    # Dataset statistics
    print("📊 DATASET OVERVIEW:")
    for split in ['train', 'val', 'test']:
        images_path = f"data/visdrone/images/{split}"
        if os.path.exists(images_path):
            images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
            print(f"   {split.upper()}: {len(images)} images")
    
    # Analyze a sample image
    print("\n🎯 ANALYZING SAMPLE IMAGE:")
    sample_image = get_sample_image()
    
    if sample_image:
        analyze_image(sample_image)
        
        # Test YOLO on real drone image
        test_yolo_on_drone(sample_image)
    else:
        print("❌ No images found in dataset!")

def get_sample_image():
    """Get a random sample image from the dataset"""
    train_images = "data/visdrone/images/train"
    if os.path.exists(train_images):
        images = [f for f in os.listdir(train_images) if f.endswith('.jpg')]
        if images:
            return os.path.join(train_images, random.choice(images))
    return None

def analyze_image(image_path):
    """Analyze a single drone image"""
    print(f"📸 Analyzing: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"   📐 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Check corresponding labels
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            objects = f.readlines()
        
        print(f"   📝 Objects in image: {len(objects)}")
        
        # Analyze object sizes
        small_objects = 0
        for obj in objects:
            parts = obj.strip().split()
            if len(parts) >= 5:
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                # Convert to pixels (assuming 2000x1500 image)
                w_pixels = w_norm * 2000
                h_pixels = h_norm * 1500
                
                # Count as "tiny" if smaller than 50x50 pixels
                if w_pixels < 50 and h_pixels < 50:
                    small_objects += 1
        
        print(f"   🔍 Tiny objects (<50px): {small_objects}/{len(objects)}")
        
        # Show object classes
        classes = set()
        for obj in objects:
            if obj.strip():
                classes.add(int(obj.split()[0]))
        print(f"   🏷️  Object classes: {classes}")

def test_yolo_on_drone(image_path):
    """Test pre-trained YOLO on drone image"""
    print("\n🤖 TESTING YOLO ON DRONE IMAGE:")
    
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    
    if results[0].boxes is not None:
        detections = len(results[0].boxes)
        print(f"   🔍 YOLO detected {detections} objects")
        
        # Save result
        results[0].save('data/visdrone/detection_demo.jpg')
        print("   💾 Saved detection demo: data/visdrone/detection_demo.jpg")
        
        # Show what was detected
        detected_classes = set()
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detected_classes.add(class_id)
        
        print(f"   🎯 Detected classes: {detected_classes}")
    else:
        print("   🔍 YOLO detected 0 objects")

if __name__ == "__main__":
    explore_visdrone()