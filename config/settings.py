import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Environment variables
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_verify_token")
    PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
    PAGE_ID = os.getenv("PAGE_ID")
    
    FB_GRAPH_URL = f"https://graph.facebook.com/v21.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    

    
    # Database Configuration
    DB_CONFIG = {
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'smart_rag'),
    }
    
    # Model Configuration
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    TEXT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    
    
    # LLM Configuration
    # Grok / x.ai (OpenAI compatible)
    LLM_MODEL = "grok-4-fast-non-reasoning"
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = "https://api.x.ai/v1"
    LLM_MAX_TOKENS = 520
    LLM_TEMPERATURE = 0.5
    
    
    # Ollama (uncomment to use)
    # LLM_MODEL = "mistral:latest"
    # LLM_BASE_URL = "http://localhost:11434/v1"
    # LLM_API_KEY = "not-needed"
    
    # Google Gemini (uncomment to use)
    # LLM_MODEL = "gemini-2.5-flash-lite"
    # LLM_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # File Paths
    VECTOR_STORES_PATH = "vector_stores"
    IMAGE_FAISS_PATH = f"{VECTOR_STORES_PATH}/image_faiss"
    TEXT_FAISS_PATH = f"{VECTOR_STORES_PATH}/text_faiss"
    PRODUCT_IMAGES_PATH = "product-image"
    PRODUCTS_JSON_PATH = "data/products.json"
    
    # CORS Configuration
    CORS_ORIGINS = ["*"]
    
    # Server Configuration
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 8000))

settings = Settings() 