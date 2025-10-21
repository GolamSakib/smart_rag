import json
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from config.settings import settings
from typing import Optional
import tempfile
import os
from pydub import AudioSegment
import io


class ModelManager:
    """Singleton class to manage model loading with lazy initialization"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._models = {
            'clip_model': None,
            'clip_processor': None,
            'embeddings': None,
            'image_index': None,
            'text_vector_store': None,
            'image_metadata': None,
            'llm': None,
            'fallback_llm': None,
            'gemini_transcriber': None
        }
    
    def get_clip_model(self):
        """Lazy load CLIP model for image processing"""
        if self._models['clip_model'] is None:
            print("Loading CLIP model...")
            self._models['clip_model'] = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
        return self._models['clip_model']
    
    def get_clip_processor(self):
        """Lazy load CLIP processor"""
        if self._models['clip_processor'] is None:
            print("Loading CLIP processor...")
            self._models['clip_processor'] = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        return self._models['clip_processor']
    
    def get_embeddings(self):
        """Lazy load text embeddings model"""
        if self._models['embeddings'] is None:
            print("Loading text embeddings model...")
            self._models['embeddings'] = HuggingFaceEmbeddings(model_name=settings.TEXT_EMBEDDING_MODEL)
        return self._models['embeddings']
    
    def get_image_index(self):
        """Lazy load image FAISS index"""
        if self._models['image_index'] is None:
            print("Loading image FAISS index...")
            try:
                self._models['image_index'] = faiss.read_index(f'{settings.IMAGE_FAISS_PATH}/image.index')
            except Exception as e:
                print(f"Warning: Could not load image index: {e}")
                self._models['image_index'] = None
        return self._models['image_index']
    
    def get_text_vector_store(self):
        """Lazy load text vector store"""
        if self._models['text_vector_store'] is None:
            print("Loading text vector store...")
            try:
                embeddings = self.get_embeddings()
                self._models['text_vector_store'] = LangchainFAISS.load_local(
                    settings.TEXT_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load text vector store: {e}")
                self._models['text_vector_store'] = None
        return self._models['text_vector_store']
    
    def get_image_metadata(self):
        """Lazy load image metadata"""
        if self._models['image_metadata'] is None:
            print("Loading image metadata...")
            try:
                with open(f'{settings.IMAGE_FAISS_PATH}/image_metadata.json', 'r') as f:
                    self._models['image_metadata'] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load image metadata: {e}")
                self._models['image_metadata'] = []
        return self._models['image_metadata']
    
    def get_llm(self):
        """Lazy load LLM"""
        if self._models['llm'] is None:
            print(f"Loading Grok LLM: {settings.LLM_MODEL}...")
            self._models['llm'] = ChatOpenAI(
                model=settings.LLM_MODEL,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_BASE_URL,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE
            )
        return self._models['llm']
    
    def get_fallback_llm(self):
        """Lazy load Fallback LLM"""
        if self._models['fallback_llm'] is None:
            print("Loading Fallback LLM...")
            print(f"Loading Fallback LLM: {settings.FALLBACK_LLM_MODEL}...")
            self._models['fallback_llm'] = ChatOpenAI(
                model=settings.FALLBACK_LLM_MODEL,
                openai_api_key=settings.FALLBACK_LLM_API_KEY,
                openai_api_base=settings.FALLBACK_LLM_BASE_URL,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE
            )
        return self._models['fallback_llm']

    def get_gemini_transcriber(self):
        """Lazy load Gemini model for transcription"""
        if self._models['gemini_transcriber'] is None:
            print("Loading Gemini for transcription...")
            if not settings.GOOGLE_API_KEY:
                raise Exception("GOOGLE_API_KEY is not set.")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self._models['gemini_transcriber'] = genai.GenerativeModel(settings.TRANSCRIPTION_MODEL_NAME)
        return self._models['gemini_transcriber']

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """Transcribes audio to text using audio bytes."""
        model = self.get_gemini_transcriber()
        print(f"Transcribing audio with original mime_type: {mime_type}")

        # The Gemini API supports 'audio/mp4' but Facebook sends 'video/mp4'.
        # We change the mime_type to 'audio/mp4' if we receive 'video/mp4'.
        if mime_type == 'video/mp4':
            mime_type = 'audio/mp4'
            print(f"Changed mime_type to: {mime_type}")

        # The audio data is passed directly to the model, avoiding file uploads.
        response = model.generate_content(
            [
                "Transcribe this audio to text.",
                {
                    "mime_type": mime_type,
                    "data": audio_bytes
                }
            ]
        )
        
        return response.text
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding using CLIP model"""
        model = self.get_clip_model()
        processor = self.get_clip_processor()
        
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding.cpu().numpy().flatten()
    
    def reload_vector_stores(self):
        """Reloads the vector stores and metadata from disk."""
        print("Reloading vector stores...")
        self._models['image_index'] = None
        self._models['text_vector_store'] = None
        self._models['image_metadata'] = None
        self.get_image_index()
        self.get_text_vector_store()
        self.get_image_metadata()
        print("OK: Vector stores reloaded.")
    
    def preload_essential_models(self):
        """Preload only essential models for basic functionality"""
        print("Preloading essential models...")
        # Only load what's absolutely necessary for startup
        pass
    
    def preload_all_models(self):
        """Preload all models (use only when needed)"""
        print("Preloading all models...")
        self.get_clip_model()
        self.get_clip_processor()
        self.get_embeddings()
        self.get_image_index()
        self.get_text_vector_store()
        self.get_image_metadata()
        self.get_llm()
        self.get_fallback_llm()
    
    def clear_models(self):
        """Clear all loaded models to free memory"""
        print("Clearing models from memory...")
        for key in self._models:
            self._models[key] = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Global model manager instance
model_manager = ModelManager() 