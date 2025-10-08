from typing import List, Dict, Optional, Tuple
from services.intent_detector import IntentType, intent_detector
from services.model_manager import model_manager
import numpy as np
from PIL import Image

class ChatTools:
    """Tool functions for handling different types of user intents"""
    
    def __init__(self):
        self.model_manager = model_manager
    
    def search_products_by_text(self, query: str, k: int = 1) -> List[Dict]:
        """
        Search for products using text query
        
        Args:
            query: Text query to search for
            k: Number of results to return
            
        Returns:
            List of product dictionaries
        """
        try:
            text_vector_store = self.model_manager.get_text_vector_store()
            if text_vector_store is None:
                return []
            
            docs = text_vector_store.similarity_search(query, k=k)
            products = []
            for doc in docs:
                products.append(doc.metadata)
            
            return products
        except Exception as e:
            print(f"Error in text product search: {e}")
            return []
    
    def search_products_by_image(self, images: List[Image.Image]) -> List[Dict]:
        """
        Search for products using image similarity
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of product dictionaries
        """
        try:
            image_index = self.model_manager.get_image_index()
            image_metadata = self.model_manager.get_image_metadata()
            
            if image_index is None or not image_metadata:
                return []
            
            products = []
            for image in images:
                image_embedding = self.model_manager.get_image_embedding(image)
                D, I = image_index.search(np.array([image_embedding]).astype('float32'), k=1)
                if I[0][0] < len(image_metadata):
                    products.append(image_metadata[I[0][0]])
            
            return products
        except Exception as e:
            print(f"Error in image product search: {e}")
            return []
    
    def remove_duplicate_products(self, products: List[Dict]) -> List[Dict]:
        """
        Remove duplicate products based on name and code
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of unique products
        """
        seen_products = set()
        unique_products = []
        
        for product in products:
            identifier = (product.get('name', '').strip(), product.get('code', '').strip())
            if identifier not in seen_products:
                seen_products.add(identifier)
                unique_products.append(product)
        
        return unique_products
    
    def build_product_context(self, products: List[Dict]) -> str:
        """
        Build context string from product list
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Formatted context string
        """
        if not products:
            return ""
        
        context = "\nAvailable products:\n"
        for product in products:
            context += f"- Name: {product['name']}, Price: {product['price']}, Description: {product['description']} Link: {product['link']}\n"
        
        return context
    
    def handle_product_search_intent(self, query: str, images: Optional[List[Image.Image]] = None) -> Tuple[List[Dict], str]:
        """
        Handle product search intent
        
        Args:
            query: User's text query
            images: Optional list of images
            
        Returns:
            Tuple of (products, context_string)
        """
        products = []
        
        # Search by images first (if provided)
        if images:
            products.extend(self.search_products_by_image(images))
        
        # Search by text (if query provided and no images or additional search needed)
        if query and query.strip():
            text_products = self.search_products_by_text(query.strip())
            products.extend(text_products)
        
        # Remove duplicates
        unique_products = self.remove_duplicate_products(products)
        
        # Build context
        context = self.build_product_context(unique_products)
        
        return unique_products, context
    
    def handle_general_chat_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle general chat intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def handle_price_inquiry_intent(self, query: str, existing_products: List[Dict] = None) -> Tuple[List[Dict], str]:
        """
        Handle price inquiry intent
        
        Args:
            query: User's text query
            existing_products: Previously retrieved products (if any)
            
        Returns:
            Tuple of (products, context_string)
        """
        # If we already have products from image search, use those
        if existing_products:
            context = self.build_product_context(existing_products)
            return existing_products, context
        
        # Otherwise, search for products
        return self.handle_product_search_intent(query)
    
    def handle_order_inquiry_intent(self, query: str, existing_products: List[Dict] = None) -> Tuple[List[Dict], str]:
        """
        Handle order inquiry intent
        
        Args:
            query: User's text query
            existing_products: Previously retrieved products (if any)
            
        Returns:
            Tuple of (products, context_string)
        """
        # If we already have products from image search, use those
        if existing_products:
            context = self.build_product_context(existing_products)
            return existing_products, context
        
        # Otherwise, search for products
        return self.handle_product_search_intent(query)
    
    def handle_delivery_inquiry_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle delivery inquiry intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def handle_return_policy_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle return policy inquiry intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def handle_size_chart_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle size chart inquiry intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def handle_image_request_intent(self, query: str, existing_products: List[Dict] = None) -> Tuple[List[Dict], str]:
        """
        Handle image request intent
        
        Args:
            query: User's text query
            existing_products: Previously retrieved products (if any)
            
        Returns:
            Tuple of (products, context_string)
        """
        # If we already have products from image search, use those
        if existing_products:
            context = self.build_product_context(existing_products)
            return existing_products, context
        
        # Otherwise, search for products
        return self.handle_product_search_intent(query)
    
    def handle_track_order_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle track order intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def handle_bargaining_intent(self, query: str, existing_products: List[Dict] = None) -> Tuple[List[Dict], str]:
        """
        Handle bargaining intent
        
        Args:
            query: User's text query
            existing_products: Previously retrieved products (if any)
            
        Returns:
            Tuple of (products, context_string)
        """
        # If we already have products from image search, use those
        if existing_products:
            context = self.build_product_context(existing_products)
            return existing_products, context
        
        # Otherwise, search for products
        return self.handle_product_search_intent(query)
    
    def handle_greeting_intent(self, query: str) -> Tuple[List[Dict], str]:
        """
        Handle greeting intent (no product search needed)
        
        Args:
            query: User's text query
            
        Returns:
            Tuple of (empty_products, empty_context)
        """
        return [], ""
    
    def process_intent(self, intent: IntentType, query: str, images: Optional[List[Image.Image]] = None, 
                      existing_products: List[Dict] = None) -> Tuple[List[Dict], str]:
        """
        Process user intent and return appropriate products and context
        
        Args:
            intent: Detected intent type
            query: User's text query
            images: Optional list of images
            existing_products: Previously retrieved products (if any)
            
        Returns:
            Tuple of (products, context_string)
        """
        # Map intents to handler functions
        intent_handlers = {
            IntentType.PRODUCT_SEARCH: self.handle_product_search_intent,
            IntentType.GENERAL_CHAT: self.handle_general_chat_intent,
            IntentType.PRICE_INQUIRY: self.handle_price_inquiry_intent,
            IntentType.ORDER_INQUIRY: self.handle_order_inquiry_intent,
            IntentType.DELIVERY_INQUIRY: self.handle_delivery_inquiry_intent,
            IntentType.RETURN_POLICY: self.handle_return_policy_intent,
            IntentType.SIZE_CHART: self.handle_size_chart_intent,
            IntentType.IMAGE_REQUEST: self.handle_image_request_intent,
            IntentType.TRACK_ORDER: self.handle_track_order_intent,
            IntentType.BARGAINING: self.handle_bargaining_intent,
            IntentType.GREETING: self.handle_greeting_intent
        }
        
        handler = intent_handlers.get(intent, self.handle_general_chat_intent)
        
        # Call appropriate handler
        if intent in [IntentType.PRICE_INQUIRY, IntentType.ORDER_INQUIRY, IntentType.IMAGE_REQUEST, 
                     IntentType.BARGAINING]:
            return handler(query, existing_products)
        elif intent == IntentType.PRODUCT_SEARCH:
            return handler(query, images)
        else:
            return handler(query)

# Global chat tools instance
chat_tools = ChatTools()
