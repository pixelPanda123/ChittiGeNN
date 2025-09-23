"""
ChittiGeNN - Offline Multimodal RAG System
Main FastAPI application entry point
"""

from pathlib import Path
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# -------------------------------------------------------------------
# Simple settings (no pydantic required)
# -------------------------------------------------------------------
class Settings:
    ALLOWED_ORIGINS = ["*"]  # allow all origins in development

settings = Settings()

# -------------------------------------------------------------------
# Lazy import routers to avoid circular imports
# -------------------------------------------------------------------
def get_routers():
    from app.routers import documents, search, health
    return documents.router, search.router, health.router

# -------------------------------------------------------------------
# Database imports
# -------------------------------------------------------------------
from app.database.base import Base
from app.database.connection import engine

# -------------------------------------------------------------------
# Create the FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="ChittiGeNN API",
    description="Offline Multimodal RAG System for Document Intelligence",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# -------------------------------------------------------------------
# Middleware: CORS
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Database initialization
# -------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------
documents_router, search_router, health_router = get_routers()
app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")

# -------------------------------------------------------------------
# Serve uploaded files
# -------------------------------------------------------------------
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# -------------------------------------------------------------------
# Serve React frontend
# -------------------------------------------------------------------
frontend_build = Path(__file__).resolve().parents[1] / "frontend" / "build"

if frontend_build.exists():
    app.mount("/static", StaticFiles(directory=frontend_build / "static"), name="static")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str, request: Request):
        """
        Serve React app for any route not handled by API.
        """
        index_file = frontend_build / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"error": "Frontend build not found."}

# -------------------------------------------------------------------
# Entry point for development
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
