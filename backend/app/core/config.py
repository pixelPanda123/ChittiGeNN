"""
Configuration settings for ChittiGeNN
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ChittiGeNN"
    VERSION: str = "0.1.0"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/chittigenn.db"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "audio/wav",
        "audio/mp3",
        "audio/m4a"
    ]
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    
    # ML Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Search Settings
    MAX_SEARCH_RESULTS: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    # OCR Settings
    TESSERACT_CMD: str = "/usr/local/bin/tesseract"
    
    # Audio Processing Settings
    WHISPER_MODEL: str = "base"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure all directories exist
for directory in [
    settings.DATA_DIR,
    settings.UPLOADS_DIR,
    settings.PROCESSED_DIR,
    settings.EMBEDDINGS_DIR,
    settings.VECTOR_DB_DIR,
    settings.UPLOADS_DIR / "pdfs",
    settings.UPLOADS_DIR / "images",
    settings.UPLOADS_DIR / "audio"
]:
    directory.mkdir(parents=True, exist_ok=True)
