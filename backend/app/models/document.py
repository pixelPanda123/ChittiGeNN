from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"

class ContentType(enum.Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    content_chunks = relationship("ContentChunk", back_populates="document")


class ContentChunk(Base):
    __tablename__ = "content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    content_type = Column(Enum(ContentType), default=ContentType.TEXT)

    document = relationship("Document", back_populates="content_chunks")
