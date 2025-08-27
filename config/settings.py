import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Environment variables
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_verify_token")
    PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
    FB_GRAPH_URL = f"https://graph.facebook.com/v21.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    
    # Database Configuration
    DB_CONFIG = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'database': 'smart_rag',
    }
    
    # Model Configuration
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    TEXT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    
    # LLM Configuration
    # OpenAI
    LLM_MODEL = "openai/gpt-oss-20b:free"
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_API_KEY = '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'
    LLM_MAX_TOKENS = 300
    LLM_TEMPERATURE = 0.7
    
    # Ollama
    # LLM_MODEL = "mistral:latest"
    # LLM_BASE_URL = "http://localhost:11434/v1"
    # LLM_API_KEY = "not-needed"
    # LLM_MAX_TOKENS = 300
    # LLM_TEMPERATURE = 0.7
    
    # File Paths
    VECTOR_STORES_PATH = "vector_stores"
    IMAGE_FAISS_PATH = f"{VECTOR_STORES_PATH}/image_faiss"
    TEXT_FAISS_PATH = f"{VECTOR_STORES_PATH}/text_faiss"
    PRODUCT_IMAGES_PATH = "product-image"
    PRODUCTS_JSON_PATH = "data/products.json"
    
    # CORS Configuration
    CORS_ORIGINS = ["*"]
    
    # Server Configuration
    HOST = "127.0.0.1"
    PORT = 8000

settings = Settings() 