#!/usr/bin/env python3
"""
Test script for intent detection system
"""

from services.intent_detector import intent_detector, IntentType

def test_intent_detection():
    """Test various user messages to verify intent detection"""
    
    test_cases = [
        # Greeting cases
        ("আসসালামু আলাইকুম", IntentType.GREETING),
        ("হাই", IntentType.GREETING),
        ("hello", IntentType.GREETING),
        
        # Product search cases
        ("জুতা দেখান", IntentType.PRODUCT_SEARCH),
        ("কোন জুতা আছে?", IntentType.PRODUCT_SEARCH),
        ("নতুন জুতা দেখান", IntentType.PRODUCT_SEARCH),
        ("স্যান্ডেল দেখান", IntentType.PRODUCT_SEARCH),
        ("কোন রঙের জুতা আছে?", IntentType.PRODUCT_SEARCH),
        
        # Price inquiry cases
        ("দাম কত?", IntentType.PRICE_INQUIRY),
        ("pp", IntentType.PRICE_INQUIRY),
        ("মূল্য জানান", IntentType.PRICE_INQUIRY),
        ("কত টাকা?", IntentType.PRICE_INQUIRY),
        
        # Order inquiry cases
        ("অর্ডার করতে চাই", IntentType.ORDER_INQUIRY),
        ("কিভাবে কিনব?", IntentType.ORDER_INQUIRY),
        ("পেমেন্ট কিভাবে?", IntentType.ORDER_INQUIRY),
        ("কনফার্ম করুন", IntentType.ORDER_INQUIRY),
        
        # Delivery inquiry cases
        ("ডেলিভারি কতদিনে?", IntentType.DELIVERY_INQUIRY),
        ("কখন পাব?", IntentType.DELIVERY_INQUIRY),
        ("ডেলিভারি চার্জ কত?", IntentType.DELIVERY_INQUIRY),
        
        # Return policy cases
        ("রিটার্ন পলিসি কি?", IntentType.RETURN_POLICY),
        ("ফেরত নিতে পারব?", IntentType.RETURN_POLICY),
        ("পছন্দ না হলে কি করব?", IntentType.RETURN_POLICY),
        
        # Size chart cases
        ("সাইজ চার্ট দেখান", IntentType.SIZE_CHART),
        ("কোন সাইজ নিব?", IntentType.SIZE_CHART),
        ("বাটা সাইজ জানান", IntentType.SIZE_CHART),
        
        # Image request cases
        ("ছবি দেখান", IntentType.IMAGE_REQUEST),
        ("চোবি দেখান", IntentType.IMAGE_REQUEST),
        ("কেমন দেখায়?", IntentType.IMAGE_REQUEST),
        
        # Track order cases
        ("অর্ডার ট্র্যাক করুন", IntentType.TRACK_ORDER),
        ("অর্ডার কোথায়?", IntentType.TRACK_ORDER),
        ("আপডেট দিন", IntentType.TRACK_ORDER),
        
        # Bargaining cases
        ("দাম কমানো যায়?", IntentType.BARGAINING),
        ("একটু কমানো", IntentType.BARGAINING),
        ("অফার আছে?", IntentType.BARGAINING),
        
        # General chat cases
        ("ধন্যবাদ", IntentType.GENERAL_CHAT),
        ("ঠিক আছে", IntentType.GENERAL_CHAT),
        ("", IntentType.GENERAL_CHAT),
    ]
    
    print("Testing Intent Detection System")
    print("=" * 50)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (message, expected_intent) in enumerate(test_cases, 1):
        detected_intent, confidence = intent_detector.detect_intent(message)
        
        is_correct = detected_intent == expected_intent
        if is_correct:
            correct_predictions += 1
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{i:2d}. {status} | '{message}' -> {detected_intent.value} (conf: {confidence:.2f}) [Expected: {expected_intent.value}]")
    
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 50)
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    return accuracy > 80  # Consider test passed if accuracy > 80%

def test_should_search_products():
    """Test the should_search_products function"""
    
    print("\nTesting should_search_products function")
    print("=" * 50)
    
    test_cases = [
        (IntentType.PRODUCT_SEARCH, 0.8, True),   # Should search
        (IntentType.PRICE_INQUIRY, 0.6, True),    # Should search
        (IntentType.ORDER_INQUIRY, 0.5, True),    # Should search
        (IntentType.IMAGE_REQUEST, 0.4, True),    # Should search
        (IntentType.DELIVERY_INQUIRY, 0.8, True), # Should search (high confidence)
        (IntentType.DELIVERY_INQUIRY, 0.5, False), # Should not search (low confidence)
        (IntentType.GREETING, 0.9, False),        # Should not search
        (IntentType.GENERAL_CHAT, 0.1, False),    # Should not search
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (intent, confidence, expected) in enumerate(test_cases, 1):
        result = intent_detector.should_search_products(intent, confidence)
        is_correct = result == expected
        
        if is_correct:
            correct_predictions += 1
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{i}. {status} | {intent.value} (conf: {confidence:.1f}) -> {result} [Expected: {expected}]")
    
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 50)
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    return accuracy == 100  # Should be 100% for this logic test

if __name__ == "__main__":
    print("Intent Detection System Test")
    print("=" * 60)
    
    # Test intent detection
    intent_test_passed = test_intent_detection()
    
    # Test should_search_products logic
    search_test_passed = test_should_search_products()
    
    print("\n" + "=" * 60)
    if intent_test_passed and search_test_passed:
        print("🎉 ALL TESTS PASSED! Intent detection system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the intent detection logic.")
    
    print("=" * 60)
