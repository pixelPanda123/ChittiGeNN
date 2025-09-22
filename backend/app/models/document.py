"""
Document models for ChittiGeNN
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum

Base = declarative_base()

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"

class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

class Document(Base):
    """Document database model"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Processing status
    status = Column(String(50), default=DocumentStatus.UPLOADED)
    processing_error = Column(Text, nullable=True)
    
    # Extracted content
    extracted_text = Column(Text, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Document chunk for vector search"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)  # text, image, audio
    
    # Vector embedding
    embedding_id = Column(String(255), nullable=True)  # Reference to vector DB
    embedding = Column(JSON, nullable=True)  # Store embedding as JSON array
    
    # Metadata
    page_number = Column(Integer, nullable=True)
    timestamp = Column(Float, nullable=True)  # For audio/video
    confidence = Column(Float, nullable=True)  # For OCR/transcription
    char_count = Column(Integer, nullable=False, default=0)
    word_count = Column(Integer, nullable=False, default=0)
    metadata = Column(JSON, nullable=True)  # Additional metadata as JSON
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    document = relationship("Document", back_populates="chunks")

# Pydantic models for API
class DocumentCreate(BaseModel):
    """Document creation model"""
    filename: str
    file_type: DocumentType
    file_size: int
    mime_type: str

class DocumentResponse(BaseModel):
    """Document response model"""
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    processing_error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    """Document list response model"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int

class DocumentChunkResponse(BaseModel):
    """Document chunk response model"""
    id: int
    document_id: int
    chunk_index: int
    content: str
    content_type: str
    page_number: Optional[int] = None
    timestamp: Optional[float] = None
    confidence: Optional[float] = None
    
    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    """Search result model"""
    chunk: DocumentChunkResponse
    document: DocumentResponse
    similarity_score: float
    highlight: Optional[str] = None

class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
