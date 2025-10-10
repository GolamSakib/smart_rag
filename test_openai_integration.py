#!/usr/bin/env python3
"""
Test script to verify OpenAI GPT model integration
"""

import os
import sys
from config.settings import settings
from services.model_manager import model_manager

def test_openai_integration():
    """Test OpenAI GPT model integration"""
    print("Testing OpenAI GPT model integration...")
    
    try:
        # Test 1: Check settings configuration
        print(f"[OK] Model: {settings.LLM_MODEL}")
        print(f"[OK] API Key configured: {'Yes' if settings.LLM_API_KEY else 'No'}")
        print(f"[OK] Max tokens: {settings.LLM_MAX_TOKENS}")
        print(f"[OK] Temperature: {settings.LLM_TEMPERATURE}")
        
        # Test 2: Initialize LLM
        print("\nInitializing LLM...")
        llm = model_manager.get_llm()
        print(f"[OK] LLM initialized successfully: {type(llm).__name__}")
        
        # Test 3: Simple text generation
        print("\nTesting text generation...")
        test_prompt = "Hello, can you respond with 'Integration successful'?"
        
        response = llm.invoke(test_prompt)
        # Safely print the response, replacing unsupported characters
        safe_response = response.content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"[OK] Response received: {safe_response}")
        
        # Test 4: Test with system prompt (as used in the chat system)
        print("\nTesting system response...")
        system_prompt = "You are a friendly sales assistant for momsandkidsworld. Give a brief welcome message."
        
        system_response = llm.invoke(system_prompt)
        
        # Safely print the system response, replacing unsupported characters
        safe_system_response = system_response.content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"[OK] System response: {safe_system_response}")
        
        print("\n[SUCCESS] All tests passed! OpenAI GPT model integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during testing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    sys.exit(0 if success else 1)
