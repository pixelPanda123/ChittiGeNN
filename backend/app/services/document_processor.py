"""
Main document processing service that orchestrates text extraction, chunking, embedding, and vector storage
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from datetime import datetime

from app.core.config import settings
from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
import fitz  # PyMuPDF
from app.models.document import Document, ContentChunk, ProcessingStatus, ContentType

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Minimal processor: extract text via PyMuPDF and chunk per page"""

    async def process_document(self, document: Document) -> Dict[str, Any]:
        start_time = datetime.now()
        file_path = Path(document.file_path)
        try:
            pages: List[tuple[int, str]] = []
            full_text_parts: List[str] = []
            with fitz.open(file_path) as pdf:
                for i in range(pdf.page_count):
                    page = pdf.load_page(i)
                    text = page.get_text("text") or ""
                    pages.append((i + 1, text))
                    full_text_parts.append(text)

            # simple chunking per page
            chunks_created = 0
            for page_num, text in pages:
                text = (text or "").strip()
                if not text:
                    continue
                chunk = ContentChunk(
                    document_id=document.id,
                    content_text=text,
                    chunk_type="TEXT",
                    page_number=page_num,
                )
                # session management handled by caller
                chunks_created += 1

            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": True,
                "document_id": document.id,
                "chunks_created": chunks_created,
                "processing_time_seconds": round(processing_time, 2),
                "metadata": {"num_pages": len(pages)},
            }
        except Exception as e:
            return {
                "success": False,
                "document_id": document.id,
                "error": str(e),
            }
    
    async def reprocess_document(self, document: Document) -> Dict[str, Any]:
        """
        Reprocess a document (useful for failed documents or when models change)
        
        Args:
            document: Document database model
            
        Returns:
            Processing result
        """
        try:
            # Delete existing chunks from vector store
            await vector_store.delete_document(str(document.id))
            
            # Process document again
            return await self.process_document(document)
            
        except Exception as e:
            logger.error(f"Failed to reprocess document {document.filename}: {e}")
            return {
                "success": False,
                "document_id": document.id,
                "error": str(e)
            }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and system health"""
        try:
            vector_stats = await vector_store.get_collection_stats()
            model_info = await embedding_service.get_model_info()
            
            return {
                "vector_store_stats": vector_stats,
                "embedding_model_info": model_info,
                "processing_config": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "max_file_size": settings.MAX_FILE_SIZE,
                    "supported_file_types": settings.ALLOWED_FILE_TYPES
                },
                "system_initialized": self.initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {"error": str(e)}
    
    async def batch_process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel
        
        Args:
            documents: List of document database models
            
        Returns:
            List of processing results
        """
        try:
            # Process documents in parallel (with reasonable concurrency limit)
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent processes
            
            async def process_with_semaphore(doc):
                async with semaphore:
                    return await self.process_document(doc)
            
            tasks = [process_with_semaphore(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "document_id": documents[i].id,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"Batch processed {len(documents)} documents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to batch process documents: {e}")
            return [{"success": False, "error": str(e)} for _ in documents]
    
    async def cleanup_document(self, document: Document) -> bool:
        """
        Clean up document data from vector store and database
        
        Args:
            document: Document database model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from vector store
            success = await vector_store.delete_document(str(document.id))
            
            # Note: Database cleanup is handled by SQLAlchemy cascade delete
            
            logger.info(f"Cleaned up document {document.filename}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cleanup document {document.filename}: {e}")
            return False
    
    async def validate_document(self, file_path: Path, mime_type: str) -> Dict[str, Any]:
        """
        Validate a document before processing
        
        Args:
            file_path: Path to the document file
            mime_type: MIME type of the document
            
        Returns:
            Validation result
        """
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {}
            }
            
            # Check if file exists
            if not file_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = file_path.stat().st_size
            validation_result["file_info"]["size_bytes"] = file_size
            
            if file_size > settings.MAX_FILE_SIZE:
                validation_result["valid"] = False
                validation_result["errors"].append(f"File size ({file_size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)")
            
            if file_size == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("File is empty")
            
            # Check file type
            if mime_type not in settings.ALLOWED_FILE_TYPES:
                validation_result["valid"] = False
                validation_result["errors"].append(f"File type {mime_type} is not supported")
            
            # Additional file-specific validations
            if mime_type == "application/pdf":
                try:
                    # Quick PDF validation
                    with open(file_path, 'rb') as f:
                        first_bytes = f.read(8)
                        if not first_bytes.startswith(b'%PDF'):
                            validation_result["valid"] = False
                            validation_result["errors"].append("File does not appear to be a valid PDF")
                except Exception as e:
                    validation_result["warnings"].append(f"Could not validate PDF structure: {e}")
            
            elif mime_type.startswith("image/"):
                try:
                    # Quick image validation
                    from PIL import Image
                    with Image.open(file_path) as img:
                        validation_result["file_info"]["image_size"] = img.size
                        validation_result["file_info"]["image_mode"] = img.mode
                except Exception as e:
                    validation_result["warnings"].append(f"Could not validate image: {e}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Document validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": []
            }

# Global instance
document_processor = DocumentProcessor()

# API compatibility functions
def validate_pdf(file_path: Path, max_size: int) -> tuple[bool, str]:
    """
    Validate PDF file for processing
    
    Args:
        file_path: Path to the PDF file
        max_size: Maximum file size in bytes
        
    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size:
            return False, f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
        
        if file_size == 0:
            return False, "File is empty"
        
        # Quick PDF validation
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(8)
                if not first_bytes.startswith(b'%PDF'):
                    return False, "File does not appear to be a valid PDF"
        except Exception as e:
            return False, f"Could not validate PDF structure: {e}"
        
        return True, "Valid PDF"
        
    except Exception as e:
        return False, f"Validation error: {e}"

async def process_pdf(file_path: Path, document_id: int):
    db = SessionLocal()
    try:
        document = db.query(Document).get(document_id)
        if not document:
            logger.error(f"Document not found for id: {document_id}")
            return

        document.processing_status = ProcessingStatus.PROCESSING
        db.add(document)
        db.commit()

        result = await document_processor.process_document(document)

        if result.get("success"):
            with fitz.open(file_path) as pdf:
                db.query(ContentChunk).filter(ContentChunk.document_id == document.id).delete()
                total_len = 0
                for i in range(pdf.page_count):
                    text = (pdf.load_page(i).get_text("text") or "").strip()
                    if not text:
                        continue
                    total_len += len(text)
                    db.add(ContentChunk(
                        document_id=document.id,
                        content_text=text,
                        chunk_type="TEXT",
                        page_number=i+1,
                    ))
            document.processing_status = ProcessingStatus.COMPLETED
            document.processed_size = total_len
            document.num_pages = result.get("metadata", {}).get("num_pages", 0)
        else:
            document.processing_status = ProcessingStatus.FAILED
        db.add(document)
        db.commit()
    except Exception as e:
        logger.error(f"Background PDF processing failed for {file_path}: {e}")
        try:
            document = db.query(Document).get(document_id)
            if document:
                document.processing_status = ProcessingStatus.FAILED
                db.add(document)
                db.commit()
        except Exception as commit_error:
            logger.error(f"Failed to update document status: {commit_error}")
    finally:
        db.close()
