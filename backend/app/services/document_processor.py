"""
Main document processing service that orchestrates text extraction, chunking, embedding, and vector storage
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from datetime import datetime

from app.core.config import settings
from app.services.text_processor import text_processor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.models.document import Document, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main service for processing documents through the entire pipeline"""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize all required services"""
        try:
            await vector_store.initialize()
            await embedding_service.load_model()
            self.initialized = True
            logger.info("Document processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            raise
    
    async def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Process a document through the entire pipeline:
        1. Extract text
        2. Create chunks
        3. Generate embeddings
        4. Store in vector database
        
        Args:
            document: Document database model
            
        Returns:
            Processing result with statistics
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            document.processing_error = None
            
            file_path = Path(document.file_path)
            
            logger.info(f"Starting processing of document: {document.filename}")
            
            # Step 1: Extract text from document
            text_data = await text_processor.process_document(file_path, document.mime_type)
            
            # Store extracted text in document
            document.extracted_text = text_data["full_text"]
            document.metadata = str(text_data["metadata"])  # Store as JSON string
            
            # Step 2: Process chunks with embeddings
            processed_chunks = await embedding_service.process_document_chunks(text_data["chunks"])
            
            # Step 3: Store in vector database
            chunk_ids = await vector_store.add_document_chunks(str(document.id), processed_chunks)
            
            # Step 4: Store chunks in database (optional - for metadata tracking)
            # This could be done in parallel with vector storage
            
            # Update document status
            document.status = DocumentStatus.PROCESSED
            document.processed_at = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "document_id": document.id,
                "chunks_created": len(processed_chunks),
                "chunk_ids": chunk_ids,
                "processing_time_seconds": round(processing_time, 2),
                "extracted_text_length": len(document.extracted_text),
                "metadata": text_data["metadata"]
            }
            
            logger.info(f"Successfully processed document {document.filename}: {len(processed_chunks)} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            # Update document status to failed
            document.status = DocumentStatus.FAILED
            document.processing_error = str(e)
            document.processed_at = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Failed to process document {document.filename}: {e}")
            
            return {
                "success": False,
                "document_id": document.id,
                "error": str(e),
                "processing_time_seconds": round(processing_time, 2)
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
