from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

from config.settings import settings
from routes import chat_routes, product_routes, image_routes, login_routes
from services.database_service import db_service

BASE_DIR = Path(__file__).resolve().parent

# Create FastAPI app with minimal startup overhead
app = FastAPI(
    title="Smart RAG API",
    description="Intelligent Retrieval-Augmented Generation API for product search",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(login_routes.router, tags=["Login"])
app.include_router(chat_routes.router, tags=["Chat"])
app.include_router(product_routes.router, tags=["Products"])
app.include_router(image_routes.router, tags=["Images"])

# Static files
app.mount("/product-image", StaticFiles(directory=settings.PRODUCT_IMAGES_PATH), name="product-image")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if "session" not in request.cookies:
        return RedirectResponse(url="/login")
    return FileResponse(str(BASE_DIR / "index.html"))


@app.on_event("startup")
async def startup_event():
    """Fast startup with minimal model loading"""
    print("Starting Smart RAG API...")
    
    # Test database connection
    if db_service.test_connection():
        print("OK: Database connection successful")
    else:
        print("FAIL: Database connection failed")
    
    print("OK: API startup complete (models will load on-demand)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Smart RAG API...")
    from services.model_manager import model_manager
    model_manager.clear_models()
    print("OK: Cleanup complete")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_healthy = db_service.test_connection()
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected"
    }


@app.post("/preload-models")
async def preload_models():
    """Endpoint to preload all models when needed"""
    from services.model_manager import model_manager
    model_manager.preload_all_models()
    return {"message": "All models preloaded successfully"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )