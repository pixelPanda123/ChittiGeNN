from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db


router = APIRouter()

# Lazy import models to avoid circular import
def get_models():
    from app.models.document import Document, ContentChunk
    return Document, ContentChunk

@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    Document, _ = get_models()
    documents = db.query(Document).all()
    return [{"id": doc.id, "title": doc.title, "status": doc.status.value} for doc in documents]

@router.post("/documents")
def create_document(title: str, db: Session = Depends(get_db)):
    Document, _ = get_models()
    doc = Document(title=title)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"id": doc.id, "title": doc.title}
