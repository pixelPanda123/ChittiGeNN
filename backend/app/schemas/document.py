from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from app.models.document import ProcessingStatus, ContentType


class ContentChunkBase(BaseModel):
    content_text: str
    chunk_type: str = "TEXT"
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    timestamp: Optional[float] = None
    confidence_score: Optional[float] = None
    language: Optional[str] = None


class ContentChunkCreate(ContentChunkBase):
    pass


class ContentChunkRead(ContentChunkBase):
    id: int

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    filename: str
    content_type: ContentType
    original_size: int


class DocumentCreate(DocumentBase):
    file_path: str
    checksum: Optional[str] = None


class DocumentRead(DocumentBase):
    id: int
    file_path: str
    processed_size: Optional[int] = None
    creation_date: Optional[datetime] = None
    ingestion_date: datetime
    last_modified: datetime
    processing_status: ProcessingStatus
    checksum: Optional[str] = None
    num_pages: Optional[int] = None

    class Config:
        from_attributes = True


class DocumentWithChunks(DocumentRead):
    chunks: List[ContentChunkRead] = []

