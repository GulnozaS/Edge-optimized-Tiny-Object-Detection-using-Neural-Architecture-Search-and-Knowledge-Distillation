# check_test_set.py
import os

def check_test_set():
    print("🔍 CHECKING TEST SET CONTENT")
    print("=" * 40)
    
    test_images_dir = "data/visdrone/images/test"
    test_labels_dir = "data/visdrone/labels/test"
    
    if os.path.exists(test_images_dir):
        test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
        test_labels = [f for f in os.listdir(test_labels_dir) if f.endswith('.txt')] if os.path.exists(test_labels_dir) else []
        
        print(f"📊 TEST SET FOUND:")
        print(f"   📸 Images: {len(test_images)}")
        print(f"   📝 Labels: {len(test_labels)}")
        
        if test_images:
            print(f"   📄 Sample images: {test_images[:3]}")  # Show first 3
            
        # Check one image to understand the data
        if test_images:
            sample_image = os.path.join(test_images_dir, test_images[0])
            import cv2
            img = cv2.imread(sample_image)
            if img is not None:
                print(f"   📐 Image size: {img.shape[1]}x{img.shape[0]}")
    else:
        print("❌ Test directory not found")

def update_dataset_config():
    print("\n📝 UPDATING DATASET CONFIGURATION...")
    
    yaml_content = """# visdrone.yaml
path: /Users/gulnoza/edge-tiny-detection/data/visdrone
train: images/train
val: images/val
test: images/test  # Now we have test data!

# Number of classes
nc: 9

# Class names
names: 
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    
    with open('data/visdrone/visdrone.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✅ Dataset config updated with test set!")
    print("💡 Now you have train/val/test splits for proper evaluation")

if __name__ == "__main__":
    check_test_set()
    update_dataset_config()