#!/usr/bin/env python3
"""
Smart RAG Startup Script
Provides different startup modes for optimized loading
"""

import argparse
import sys
import uvicorn
from config.settings import settings


def start_fast():
    """Start with minimal loading for fastest startup"""
    print("🚀 Starting in FAST mode (minimal loading)...")
    print("   Models will load on-demand when first used")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )


def start_preloaded():
    """Start with all models preloaded"""
    print("🔄 Starting in PRELOADED mode (all models loaded at startup)...")
    
    # Preload models before starting server
    from services.model_manager import model_manager
    model_manager.preload_all_models()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )


def start_dev():
    """Start in development mode with auto-reload"""
    print("🛠️  Starting in DEV mode (auto-reload enabled)...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="debug"
    )


def check_health():
    """Check system health and dependencies"""
    print("🏥 Checking system health...")
    
    # Check database connection
    from services.database_service import db_service
    if db_service.test_connection():
        print("✓ Database connection successful")
    else:
        print("✗ Database connection failed")
        return False
    
    # Check if vector stores exist
    import os
    if os.path.exists(settings.IMAGE_FAISS_PATH):
        print("✓ Image vector store found")
    else:
        print("⚠️  Image vector store not found - run training first")
    
    if os.path.exists(settings.TEXT_FAISS_PATH):
        print("✓ Text vector store found")
    else:
        print("⚠️  Text vector store not found - run training first")
    
    # Check products data
    if os.path.exists(settings.PRODUCTS_JSON_PATH):
        print("✓ Products data found")
    else:
        print("⚠️  Products data not found - generate JSON first")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Smart RAG Startup Script")
    parser.add_argument(
        "mode",
        choices=["fast", "preloaded", "dev", "health"],
        help="Startup mode: fast (minimal loading), preloaded (all models), dev (development), health (check system)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Port to run on (default: {settings.PORT})"
    )
    parser.add_argument(
        "--host",
        default=settings.HOST,
        help=f"Host to bind to (default: {settings.HOST})"
    )
    
    args = parser.parse_args()
    
    # Update settings if provided
    if args.port != settings.PORT:
        settings.PORT = args.port
    if args.host != settings.HOST:
        settings.HOST = args.host
    
    if args.mode == "fast":
        start_fast()
    elif args.mode == "preloaded":
        start_preloaded()
    elif args.mode == "dev":
        start_dev()
    elif args.mode == "health":
        if check_health():
            print("🎉 System is healthy!")
            sys.exit(0)
        else:
            print("❌ System health check failed!")
            sys.exit(1)


if __name__ == "__main__":
    main() 