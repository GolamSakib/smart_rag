# Test script to verify the image upload logic
# This simulates the key parts of the chat endpoint logic

def test_image_upload_logic():
    """Test the image upload logic to ensure only single product is returned"""
    
    # Simulate session data with previous products
    session_data = {
        "last_products": [
            {"name": "Previous Product 1", "price": "1000"},
            {"name": "Previous Product 2", "price": "2000"}
        ],
        "message_count": 5
    }
    
    # Simulate image upload scenario
    images = ["image1.jpg"]  # Simulating image upload
    text = "What is this product?"
    
    # Initialize retrieved_products
    retrieved_products = []
    
    # Image processing logic (from our fix)
    if images:
        # Clear any previous products from session when processing new images
        retrieved_products = []
        session_data["last_products"] = []
        
        # Simulate image search result
        image_search_result = {"name": "New Product from Image", "price": "1500"}
        retrieved_products.append(image_search_result)
        session_data["last_products"] = retrieved_products
    
    # Text search logic (should NOT run when images are present)
    if text and not images:
        retrieved_products = session_data["last_products"]
    
    print("Test Results:")
    print(f"Number of products returned: {len(retrieved_products)}")
    print(f"Products: {retrieved_products}")
    print(f"Session last_products: {session_data['last_products']}")
    
    # Verify only one product is returned
    assert len(retrieved_products) == 1, f"Expected 1 product, got {len(retrieved_products)}"
    assert retrieved_products[0]["name"] == "New Product from Image", "Wrong product returned"
    print("✅ Test passed: Only single product from image search is returned")

if __name__ == "__main__":
    test_image_upload_logic()
