import logging
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.models.document import Document, ProcessingStatus, ContentType
from app.schemas.document import DocumentRead
from app.services.document_processor import validate_pdf, process_pdf

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentRead)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    uploads_dir = settings.UPLOADS_DIR / "pdfs"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    save_path = uploads_dir / file.filename
    try:
        content = await file.read()
        with open(save_path, "wb") as out:
            out.write(content)
    finally:
        await file.close()

    ok, reason = validate_pdf(save_path, settings.MAX_FILE_SIZE)
    if not ok:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=reason)

    doc = Document(
        filename=file.filename,
        file_path=str(save_path),
        content_type=ContentType.PDF,
        original_size=len(content),
        ingestion_date=datetime.utcnow(),
        processing_status=ProcessingStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_pdf, db, save_path)

    return doc


@router.post("/batch", response_model=List[DocumentRead])
async def upload_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    results: List[Document] = []
    for file in files:
        if file.content_type != "application/pdf":
            continue
        uploads_dir = settings.UPLOADS_DIR / "pdfs"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / file.filename
        content = await file.read()
        with open(save_path, "wb") as out:
            out.write(content)
        await file.close()
        ok, reason = validate_pdf(save_path, settings.MAX_FILE_SIZE)
        if not ok:
            save_path.unlink(missing_ok=True)
            continue
        doc = Document(
            filename=file.filename,
            file_path=str(save_path),
            content_type=ContentType.PDF,
            original_size=len(content),
            ingestion_date=datetime.utcnow(),
            processing_status=ProcessingStatus.PENDING,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        results.append(doc)
        background_tasks.add_task(process_pdf, db, save_path)
    return results


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/status/{document_id}")
async def get_status(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": doc.id, "status": doc.processing_status}

