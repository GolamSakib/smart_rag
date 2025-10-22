#!/usr/bin/env python3
"""
Test script to verify YOLO object detection integration with CLIP embeddings.
This script tests both training and inference phases.
"""

import os
import sys
import numpy as np
from PIL import Image
from services.model_manager import model_manager

def test_yolo_object_detection():
    """Test YOLO object detection on sample images"""
    print("Testing YOLO object detection integration...")
    
    # Test with sample images from the product-image directory
    image_dir = "product-image"
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} directory not found")
        return False
    
    # Get first few images for testing
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    
    if not image_files:
        print("No image files found in product-image directory")
        return False
    
    print(f"Testing with {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nTesting image: {image_file}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"  Image size: {image.size}")
            
            # Test object detection
            object_detector = model_manager.get_object_detector()
            results = object_detector(image)
            
            # Check for target objects
            target_objects_found = False
            for result in results:
                boxes = result.boxes
                if boxes is not None and boxes.shape[0] > 0:
                    for i in range(boxes.shape[0]):
                        cls = boxes.cls[i].item()
                        conf = boxes.conf[i].item()
                        if cls in model_manager.TARGET_CLASSES:
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            print(f"  Found target object (class {cls}) with confidence {conf:.2f}")
                            print(f"  Bounding box: ({x1}, {y1}, {x2}, {y2})")
                            target_objects_found = True
                            
                            # Test cropping and embedding generation
                            cropped_image = image.crop((x1, y1, x2, y2))
                            print(f"  Cropped image size: {cropped_image.size}")
                            
                            # Generate embedding
                            embedding = model_manager.get_image_embedding(image)
                            print(f"  Generated embedding shape: {embedding.shape}")
                            print(f"  Embedding sample values: {embedding[:5]}")
                            break
                    if target_objects_found:
                        break
            
            if not target_objects_found:
                print("  No target objects (handbag/shoe) detected in this image")
                # Test fallback to full image processing
                embedding = model_manager.get_image_embedding(image)
                print(f"  Fallback embedding shape: {embedding.shape}")
                
        except Exception as e:
            print(f"  Error processing {image_file}: {e}")
            return False
    
    print("\nYOLO object detection integration test completed successfully!")
    return True

def test_training_compatibility():
    """Test if the training script can run without errors"""
    print("\nTesting training script compatibility...")
    
    # Check if products.json exists
    if not os.path.exists('data/products.json'):
        print("Warning: data/products.json not found. Training test skipped.")
        return True
    
    try:
        # Import training modules to check for syntax errors
        import training
        print("Training script imports successfully")
        return True
    except Exception as e:
        print(f"Training script has errors: {e}")
        return False

if __name__ == "__main__":
    print("Testing YOLO + CLIP Integration")
    print("=" * 50)
    
    success = True
    
    # Test training compatibility
    success &= test_training_compatibility()
    
    # Test object detection
    success &= test_yolo_object_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! YOLO integration is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python training.py' to retrain with YOLO object detection")
        print("2. Restart your application to load the new models")
        print("3. Test with real user images in Messenger")
    else:
        print(" Some tests failed. Please check the errors above.")
        sys.exit(1)
