#!/usr/bin/env python3
"""
Debug script to test YOLO object detection on actual product images
"""

import os
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

def test_yolo_detection():
    """Test YOLO detection on product images"""
    print("🔍 Testing YOLO Object Detection")
    print("=" * 50)
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("yolov8l.pt")
    
    # Target classes
    TARGET_CLASSES = {24, 26, 28}  # backpack (24), handbag (26), suitcase (28)
    class_names = model.names
    
    print(f"Target classes: {TARGET_CLASSES}")
    print(f"Class 24: {class_names.get(24, 'Unknown')}")
    print(f"Class 26: {class_names.get(26, 'Unknown')}")
    print(f"Class 28: {class_names.get(28, 'Unknown')}")
    
    # Test on product images
    image_dir = "product-image"
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found")
        return
    
    # Get sample images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    if not image_files:
        print("No images found")
        return
    
    print(f"\nTesting on {len(image_files)} sample images:")
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n{i}. Testing: {image_file}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"   Image size: {image.size}")
            
            # Run detection
            results = model(image)
            
            # Check all detected objects
            all_objects = []
            target_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and boxes.shape[0] > 0:
                    for j in range(boxes.shape[0]):
                        cls = int(boxes.cls[j].item())
                        conf = boxes.conf[j].item()
                        x1, y1, x2, y2 = map(int, boxes.xyxy[j])
                        
                        obj_info = {
                            'class': cls,
                            'name': class_names.get(cls, 'Unknown'),
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        }
                        all_objects.append(obj_info)
                        
                        if cls in TARGET_CLASSES:
                            target_objects.append(obj_info)
            
            # Print results
            if all_objects:
                print(f"   All detected objects:")
                for obj in all_objects:
                    print(f"     - {obj['name']} (class {obj['class']}) - confidence: {obj['confidence']:.2f}")
            else:
                print("   No objects detected")
            
            if target_objects:
                print(f"Target objects found: {len(target_objects)}")
                for obj in target_objects:
                    print(f"     - {obj['name']} (class {obj['class']}) - confidence: {obj['confidence']:.2f}")
            else:
                print("No target objects (handbag/shoe) detected")
                
        except Exception as e:
            print(f"Error: {e}")

def test_with_different_confidence():
    """Test with different confidence thresholds"""
    print("\n🎯 Testing with different confidence thresholds")
    print("=" * 50)
    
    model = YOLO("yolov8l.pt")
    TARGET_CLASSES = {24, 26, 28}  # backpack (24), handbag (26), suitcase (28)
    
    # Test on first image
    image_dir = "product-image"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found")
        return
    
    image_path = os.path.join(image_dir, image_files[0])
    image = Image.open(image_path).convert('RGB')
    
    print(f"Testing on: {image_files[0]}")
    
    # Test different confidence thresholds
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for conf in conf_thresholds:
        print(f"\nConfidence threshold: {conf}")
        results = model(image, conf=conf)
        
        target_found = False
        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.shape[0] > 0:
                for i in range(boxes.shape[0]):
                    cls = int(boxes.cls[i].item())
                    if cls in TARGET_CLASSES:
                        conf_score = boxes.conf[i].item()
                        print(f"Found target object (class {cls}) with confidence {conf_score:.2f}")
                        target_found = True
                        break
                if target_found:
                    break
        
        if not target_found:
            print(f"No target objects found")

if __name__ == "__main__":
    test_yolo_detection()
    test_with_different_confidence()
    
    print("\n" + "=" * 50)
    print("🔧 Possible solutions if YOLO isn't detecting objects:")
    print("1. Check if your images actually contain handbags/shoes")
    print("2. Try different YOLO models (yolov8s.pt, yolov8m.pt)")
    print("3. Lower confidence threshold")
    print("4. Use custom trained YOLO model")
    print("5. Consider alternative object detection approaches")
    print("6. Use image preprocessing (resize, enhance contrast)")
    print("=" * 50)
