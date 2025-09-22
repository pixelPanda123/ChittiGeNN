"""
ChittiGeNN - Offline Multimodal RAG System
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from app.routers import documents as documents_router
from app.core.config import settings
from app.database import Base, engine

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s - %(message)s')

# Create FastAPI app
app = FastAPI(
    title="ChittiGeNN API",
    description="Offline Multimodal RAG System for Document Intelligence",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB init
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# Include API routes
app.include_router(documents_router.router, prefix="/api/v1")

# Mount static files for uploaded documents
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "ChittiGeNN - Offline Multimodal RAG System",
        "version": "0.1.0",
        "status": "running",
        "docs": "/api/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
