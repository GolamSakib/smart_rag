#!/usr/bin/env python3
"""
Test script to demonstrate intent-based routing functionality
"""

from services.intent_detector import intent_detector, IntentType
from services.chat_tools import chat_tools

def test_intent_detection():
    """Test intent detection with various user queries"""
    
    test_cases = [
        # Product search queries
        ("জুতা দেখান", False, IntentType.PRODUCT_SEARCH),
        ("আমি জুতা খুঁজছি", False, IntentType.PRODUCT_SEARCH),
        ("সব জুতা দেখান", False, IntentType.PRODUCT_SEARCH),
        
        # Price inquiries
        ("দাম কত", False, IntentType.PRICE_INQUIRY),
        ("pp", False, IntentType.PRICE_INQUIRY),
        ("মূল্য জানান", False, IntentType.PRICE_INQUIRY),
        
        # Order inquiries
        ("কিভাবে অর্ডার করব", False, IntentType.ORDER_INQUIRY),
        ("কিনতে চাই", False, IntentType.ORDER_INQUIRY),
        
        # Delivery inquiries
        ("ডেলিভারি কতদিনে হবে", False, IntentType.DELIVERY_INQUIRY),
        ("কখন পাব", False, IntentType.DELIVERY_INQUIRY),
        
        # Return policy
        ("রিটার্ন পলিসি কি", False, IntentType.RETURN_POLICY),
        
        # Size chart
        ("সাইজ চার্ট দেখান", False, IntentType.SIZE_CHART),
        
        # Greetings
        ("আসসালামু আলাইকুম", False, IntentType.GREETING),
        ("হ্যালো", False, IntentType.GREETING),
        
        # Image requests
        ("ছবি দেখান", False, IntentType.IMAGE_REQUEST),
        
        # Bargaining
        ("দাম একটু কমানো যায়", False, IntentType.BARGAINING),
        
        # General chat
        ("কেমন আছেন", False, IntentType.GENERAL_CHAT),
    ]
    
    print("=== Testing Intent Detection ===")
    print()
    
    for query, has_images, expected_intent in test_cases:
        detected_intent, confidence = intent_detector.detect_intent(query, has_images)
        
        status = "✅" if detected_intent == expected_intent else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_intent.value}")
        print(f"   Detected: {detected_intent.value} (confidence: {confidence:.2f})")
        print()

def test_k_value_selection():
    """Test k value selection based on intent and query"""
    
    test_cases = [
        ("সব জুতা দেখান", IntentType.PRODUCT_SEARCH, 5),  # "all" keyword
        ("একটি জুতা দেখান", IntentType.PRODUCT_SEARCH, 1),  # specific request
        ("জুতা দেখান", IntentType.PRODUCT_SEARCH, 3),  # default for product search
        ("দাম কত", IntentType.PRICE_INQUIRY, 1),  # price inquiry
        ("সব পণ্য দেখান", IntentType.GENERAL_CHAT, 5),  # "all" keyword in general chat
    ]
    
    print("=== Testing K Value Selection ===")
    print()
    
    for query, intent, expected_k in test_cases:
        k_value = chat_tools.get_k_value_for_intent(intent, query)
        
        status = "✅" if k_value == expected_k else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Intent: {intent.value}")
        print(f"   Expected K: {expected_k}")
        print(f"   Actual K: {k_value}")
        print()

def test_intent_routing():
    """Test intent-based routing"""
    
    print("=== Testing Intent-Based Routing ===")
    print()
    
    # Test cases for different intents
    test_cases = [
        ("জুতা দেখান", IntentType.PRODUCT_SEARCH, True),  # Should search products
        ("দাম কত", IntentType.PRICE_INQUIRY, True),  # Should search products
        ("ডেলিভারি কতদিনে", IntentType.DELIVERY_INQUIRY, False),  # Should not search products
        ("আসসালামু আলাইকুম", IntentType.GREETING, False),  # Should not search products
        ("রিটার্ন পলিসি কি", IntentType.RETURN_POLICY, False),  # Should not search products
    ]
    
    for query, intent, should_search in test_cases:
        should_search_result = intent_detector.should_search_products(intent, 0.8)
        
        status = "✅" if should_search_result == should_search else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Intent: {intent.value}")
        print(f"   Should search products: {should_search}")
        print(f"   Actually searches: {should_search_result}")
        print()

if __name__ == "__main__":
    print("🧪 Testing Intent-Based Routing System")
    print("=" * 50)
    print()
    
    test_intent_detection()
    test_k_value_selection()
    test_intent_routing()
    
    print("✅ All tests completed!")
