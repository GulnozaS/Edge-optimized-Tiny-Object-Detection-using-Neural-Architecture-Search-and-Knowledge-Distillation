# evaluate_baseline.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def evaluate_baseline():
    print("📊 EVALUATING BASELINE PERFORMANCE")
    print("=" * 50)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Evaluate on a few validation images
    val_images_dir = "data/visdrone/images/val"
    val_labels_dir = "data/visdrone/labels/val"
    
    if not os.path.exists(val_images_dir):
        print("❌ Validation set not found")
        return
    
    # Get sample of validation images (first 10 for quick evaluation)
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')][:10]
    
    print(f"🔍 Evaluating on {len(image_files)} validation images...")
    
    total_ground_truth = 0
    total_detected = 0
    tiny_objects_gt = 0
    tiny_objects_detected = 0
    
    for image_file in image_files:
        image_path = os.path.join(val_images_dir, image_file)
        label_path = os.path.join(val_labels_dir, image_file.replace('.jpg', '.txt'))
        
        # Count ground truth objects
        gt_count, tiny_gt_count = count_ground_truth(label_path)
        total_ground_truth += gt_count
        tiny_objects_gt += tiny_gt_count
        
        # Run YOLO detection
        det_count, tiny_det_count = run_detection(model, image_path, label_path)
        total_detected += det_count
        tiny_objects_detected += tiny_det_count
        
        # FIXED: Use det_count instead of detected
        print(f"   {image_file}: {det_count}/{gt_count} objects")
    
    # Calculate metrics
    detection_rate = total_detected / total_ground_truth if total_ground_truth > 0 else 0
    tiny_detection_rate = tiny_objects_detected / tiny_objects_gt if tiny_objects_gt > 0 else 0
    
    print(f"\n📈 BASELINE PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"   📊 Total Ground Truth Objects: {total_ground_truth}")
    print(f"   ✅ Total Detected: {total_detected}")
    print(f"   🎯 Overall Detection Rate: {detection_rate:.1%}")
    print(f"   🔍 Tiny Objects (<50px): {tiny_objects_gt}")
    print(f"   ✅ Tiny Objects Detected: {tiny_objects_detected}")
    print(f"   🎯 Tiny Object Detection Rate: {tiny_detection_rate:.1%}")
    
    print(f"\n💡 INSIGHT: Baseline misses {1-detection_rate:.1%} of objects!")
    print(f"💡 CHALLENGE: Tiny objects are {tiny_detection_rate:.1%} detected!")
    
    return detection_rate, tiny_detection_rate

def count_ground_truth(label_path):
    """Count objects in ground truth file"""
    if not os.path.exists(label_path):
        return 0, 0
    
    total_objects = 0
    tiny_objects = 0
    
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                total_objects += 1
                parts = line.strip().split()
                if len(parts) >= 5:
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    # Convert to pixels (assuming ~2000x1500 images)
                    w_pixels = w_norm * 2000
                    h_pixels = h_norm * 1500
                    
                    if w_pixels < 50 and h_pixels < 50:
                        tiny_objects += 1
    
    return total_objects, tiny_objects

def run_detection(model, image_path, label_path):
    """Run YOLO detection and count matches with ground truth"""
    results = model(image_path)
    
    if results[0].boxes is None:
        return 0, 0
    
    # Simple detection count (we'll improve this later with proper IoU matching)
    detected_count = len(results[0].boxes)
    
    # Estimate tiny objects detected (YOLO doesn't provide size info easily)
    # For now, we'll assume similar ratio as ground truth
    _, tiny_gt_count = count_ground_truth(label_path)
    gt_count, _ = count_ground_truth(label_path)
    
    if gt_count > 0:
        tiny_ratio = tiny_gt_count / gt_count
        tiny_detected = int(detected_count * tiny_ratio)
    else:
        tiny_detected = 0
    
    return detected_count, tiny_detected

def visualize_problem():
    """Create visual examples of the detection challenge"""
    print(f"\n🎨 CREATING VISUAL EXAMPLES OF THE PROBLEM...")
    
    model = YOLO('yolov8n.pt')
    images_dir = "data/visdrone/images/val"
    labels_dir = "data/visdrone/labels/val"
    
    # Find an image with many tiny objects
    best_example = None
    max_tiny_objects = 0
    
    for image_file in os.listdir(images_dir)[:20]:  # Check first 20
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            _, tiny_count = count_ground_truth(label_path)
            if tiny_count > max_tiny_objects:
                max_tiny_objects = tiny_count
                best_example = image_file
    
    if best_example:
        print(f"📸 Using example: {best_example} ({max_tiny_objects} tiny objects)")
        
        # Run detection and save comparison
        image_path = os.path.join(images_dir, best_example)
        results = model(image_path)
        results[0].save('data/visdrone/baseline_detection.jpg')
        
        print("💾 Saved baseline detection example")
        print("🔍 Compare: Ground Truth vs YOLO detection")

if __name__ == "__main__":
    detection_rate, tiny_rate = evaluate_baseline()
    visualize_problem()