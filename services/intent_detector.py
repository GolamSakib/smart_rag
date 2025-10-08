import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

class IntentType(Enum):
    PRODUCT_SEARCH = "product_search"
    GENERAL_CHAT = "general_chat"
    ORDER_INQUIRY = "order_inquiry"
    DELIVERY_INQUIRY = "delivery_inquiry"
    PRICE_INQUIRY = "price_inquiry"
    RETURN_POLICY = "return_policy"
    SIZE_CHART = "size_chart"
    GREETING = "greeting"
    IMAGE_REQUEST = "image_request"
    TRACK_ORDER = "track_order"
    BARGAINING = "bargaining"

class IntentDetector:
    """Intent detection system to classify user messages and determine appropriate actions"""
    
    def __init__(self):
        # Define patterns for different intents
        self.intent_patterns = {
            IntentType.PRODUCT_SEARCH: [
                # Product-related keywords
                r'\b(জুতা|shoe|shoes|স্যান্ডেল|sandal|slipper|বুট|boot|হিল|heel|ফ্ল্যাট|flat|স্পোর্টস|sports|রানিং|running|ক্যাজুয়াল|casual|ফরমাল|formal)\b',
                r'\b(কি আছে|what.*available|কোন.*পণ্য|which.*product|দেখান|show|show.*me)\b',
                r'\b(ব্র্যান্ড|brand|কোম্পানি|company|কোথা থেকে|where.*from)\b',
                r'\b(কোয়ালিটি|quality|গুণমান|material|কাপড়|fabric)\b',
                r'\b(রঙ|color|colour|কোন.*রঙ|which.*color)\b',
                r'\b(সাইজ|size|কোন.*সাইজ|which.*size)\b',
                r'\b(মডেল|model|স্টাইল|style|ডিজাইন|design)\b',
                r'\b(নতুন|new|latest|সর্বশেষ|recent)\b',
                r'\b(অফার|offer|ডিসকাউন্ট|discount|সেল|sale)\b',
                r'\b(কিভাবে|how.*look|কেমন|look.*like|appearance)\b'
            ],
            
            IntentType.PRICE_INQUIRY: [
                r'\b(দাম|price|মূল্য|cost|কত|how.*much|টাকা|taka)\b',
                r'\b(pp\b|price.*please|দাম.*জানান|মূল্য.*বলুন)\b',
                r'\b(কত.*টাকা|how.*much.*money|খরচ|expense)\b'
            ],
            
            IntentType.ORDER_INQUIRY: [
                r'\b(অর্ডার|order|কিনা|buy|purchase|নিতে.*চাই|want.*to.*buy)\b',
                r'\b(কিভাবে.*কিনব|how.*to.*buy|অর্ডার.*কিভাবে|order.*process)\b',
                r'\b(পেমেন্ট|payment|টাকা.*দিতে|pay|money)\b',
                r'\b(কনফার্ম|confirm|অর্ডার.*কনফার্ম|confirm.*order)\b'
            ],
            
            IntentType.DELIVERY_INQUIRY: [
                r'\b(ডেলিভারি|delivery|কখন.*পাব|when.*get|কতদিন|how.*long)\b',
                r'\b(কোথায়.*ডেলিভারি|where.*delivery|এলাকা|area|location)\b',
                r'\b(ডেলিভারি.*চার্জ|delivery.*charge|কুরিয়ার|courier)\b',
                r'\b(কতদিনে.*পৌঁছাবে|how.*long.*delivery|time.*delivery)\b'
            ],
            
            IntentType.RETURN_POLICY: [
                r'\b(রিটার্ন|return|ফেরত|exchange|বদল|change)\b',
                r'\b(পলিসি|policy|নিয়ম|rule|শর্ত|condition)\b',
                r'\b(পছন্দ.*না|don.*like|খারাপ|bad|ভুল|wrong)\b',
                r'\b(রিফান্ড|refund|টাকা.*ফেরত|money.*back)\b'
            ],
            
            IntentType.SIZE_CHART: [
                r'\b(সাইজ.*চার্ট|size.*chart|সাইজ.*টেবিল|size.*table)\b',
                r'\b(কোন.*সাইজ|which.*size|সাইজ.*কিভাবে|how.*size)\b',
                r'\b(ফিট|fit|সাইজ.*ফিট|size.*fit)\b',
                r'\b(বাটা|bata|এপেক্স|apex|সাইজ.*জানান|tell.*size)\b'
            ],
            
            IntentType.IMAGE_REQUEST: [
                r'\b(ছবি|image|picture|photo|চিত্র|দেখান|show.*image)\b',
                r'\b(চোবি.*দেখান|show.*picture|image.*dekhte.*chai)\b',
                r'\b(কেমন.*দেখায়|how.*look|appearance|look.*like)\b'
            ],
            
            IntentType.TRACK_ORDER: [
                r'\b(ট্র্যাক|track|অর্ডার.*কোথায়|where.*order|status)\b',
                r'\b(অর্ডার.*ট্র্যাক|order.*track|কোথায়.*আছে|where.*is)\b',
                r'\b(আপডেট|update|কোন.*খবর|what.*news)\b'
            ],
            
            IntentType.BARGAINING: [
                r'\b(দাম.*কমানো|reduce.*price|কম.*দাম|less.*price)\b',
                r'\b(একটু.*কমানো|little.*less|কিছু.*কমানো|some.*discount)\b',
                r'\b(দাম.*বেশি|price.*high|কমানো.*যায়|can.*reduce)\b',
                r'\b(অফার|offer|ডিসকাউন্ট|discount|সেল|sale)\b'
            ],
            
            IntentType.GREETING: [
                r'\b(আসসালামু.*আলাইকুম|assalamu.*alaiikum|সালাম|salam)\b',
                r'\b(হাই|hi|হ্যালো|hello|হেলো|hey)\b',
                r'\b(কেমন.*আছেন|how.*are.*you|কি.*খবর|what.*news)\b'
            ]
        }
        
        # Define priority order for intents (higher priority first)
        self.intent_priority = [
            IntentType.GREETING,
            IntentType.PRODUCT_SEARCH,
            IntentType.PRICE_INQUIRY,
            IntentType.ORDER_INQUIRY,
            IntentType.DELIVERY_INQUIRY,
            IntentType.RETURN_POLICY,
            IntentType.SIZE_CHART,
            IntentType.IMAGE_REQUEST,
            IntentType.TRACK_ORDER,
            IntentType.BARGAINING,
            IntentType.GENERAL_CHAT
        ]
    
    def detect_intent(self, message: str, has_images: bool = False) -> Tuple[IntentType, float]:
        """
        Detect the primary intent of a user message
        
        Args:
            message: User's message text
            has_images: Whether the message contains images
            
        Returns:
            Tuple of (IntentType, confidence_score)
        """
        if not message or not message.strip():
            return IntentType.GENERAL_CHAT, 0.0
        
        message_lower = message.lower().strip()
        intent_scores = {}
        
        # If images are present, prioritize product search
        if has_images:
            intent_scores[IntentType.PRODUCT_SEARCH] = 0.9
        
        # Check each intent pattern
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    matches += 1
                    # Give higher score for more specific patterns
                    if len(pattern) > 20:  # Longer patterns are more specific
                        score += 0.3
                    else:
                        score += 0.2
            
            # Normalize score based on number of matches
            if matches > 0:
                score = min(score, 1.0)  # Cap at 1.0
                intent_scores[intent_type] = score
        
        # If no specific intent detected, classify as general chat
        if not intent_scores:
            return IntentType.GENERAL_CHAT, 0.1
        
        # Return the intent with highest score, considering priority
        best_intent = max(intent_scores.items(), 
                         key=lambda x: (x[1], -self.intent_priority.index(x[0])))
        
        return best_intent[0], best_intent[1]
    
    def should_search_products(self, intent: IntentType, confidence: float) -> bool:
        """
        Determine if product search should be performed based on intent
        
        Args:
            intent: Detected intent type
            confidence: Confidence score of the intent
            
        Returns:
            Boolean indicating whether to search products
        """
        # Always search for product-related intents with reasonable confidence
        product_related_intents = [
            IntentType.PRODUCT_SEARCH,
            IntentType.PRICE_INQUIRY,
            IntentType.ORDER_INQUIRY,
            IntentType.IMAGE_REQUEST
        ]
        
        if intent in product_related_intents and confidence > 0.3:
            return True
        
        # For other intents, only search if confidence is very high
        if intent in [IntentType.DELIVERY_INQUIRY, IntentType.RETURN_POLICY, IntentType.SIZE_CHART] and confidence > 0.7:
            return True
        
        return False
    
    def get_context_requirements(self, intent: IntentType) -> Dict[str, bool]:
        """
        Get context requirements for different intents
        
        Returns:
            Dictionary with context requirements
        """
        requirements = {
            'needs_products': False,
            'needs_chat_history': True,
            'needs_specific_context': False
        }
        
        if intent in [IntentType.PRODUCT_SEARCH, IntentType.PRICE_INQUIRY, IntentType.ORDER_INQUIRY, IntentType.IMAGE_REQUEST]:
            requirements['needs_products'] = True
            requirements['needs_specific_context'] = True
        elif intent in [IntentType.DELIVERY_INQUIRY, IntentType.RETURN_POLICY, IntentType.SIZE_CHART]:
            requirements['needs_specific_context'] = True
        elif intent == IntentType.GREETING:
            requirements['needs_chat_history'] = False
        
        return requirements

# Global intent detector instance
intent_detector = IntentDetector()
