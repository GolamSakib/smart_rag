#!/usr/bin/env python3
"""
Find correct YOLO class IDs for bags, shoes, and suitcases
"""

from ultralytics import YOLO

def find_correct_class_ids():
    """Find the correct class IDs for our target objects"""
    model = YOLO("yolov8n.pt")
    class_names = model.names
    
    print("🔍 YOLO Class Names for Bags, Shoes, and Suitcases:")
    print("=" * 60)
    
    # Search for relevant classes
    target_keywords = ['bag', 'shoe', 'suitcase', 'handbag', 'backpack', 'purse']
    
    relevant_classes = []
    for class_id, class_name in class_names.items():
        if any(keyword in class_name.lower() for keyword in target_keywords):
            relevant_classes.append((class_id, class_name))
    
    print("Found relevant classes:")
    for class_id, class_name in relevant_classes:
        print(f"  Class {class_id}: {class_name}")
    
    print("\n" + "=" * 60)
    print("📋 All YOLO Classes (for reference):")
    print("=" * 60)
    
    for class_id, class_name in class_names.items():
        print(f"{class_id:2d}: {class_name}")
    
    print("\n" + "=" * 60)
    print("✅ Recommended class IDs for your products:")
    print("=" * 60)
    
    # Based on the debug output, we know:
    # Class 26 = handbag
    # Class 28 = suitcase
    # We need to find shoe class
    
    shoe_classes = []
    for class_id, class_name in class_names.items():
        if 'shoe' in class_name.lower() or 'footwear' in class_name.lower():
            shoe_classes.append((class_id, class_name))
    
    print("For bags and suitcases:")
    print("  - Class 26: handbag")
    print("  - Class 28: suitcase")
    
    if shoe_classes:
        print("For shoes:")
        for class_id, class_name in shoe_classes:
            print(f"  - Class {class_id}: {class_name}")
    else:
        print("For shoes: No specific shoe class found in YOLO")
        print("  Consider using class 0 (person) if shoes are worn by people")

if __name__ == "__main__":
    find_correct_class_ids()
