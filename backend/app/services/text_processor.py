"""
Text processing service for document chunking and preprocessing
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber
from PIL import Image
import pytesseract
import io

from app.core.config import settings

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text extraction and chunking for various document types"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
    async def extract_text_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple methods for better coverage
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text, metadata, and page information
        """
        try:
            text_data = {
                "full_text": "",
                "pages": [],
                "metadata": {},
                "extraction_method": "pdfplumber"
            }
            
            # Try pdfplumber first (better for complex layouts)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text_data["metadata"] = {
                        "num_pages": len(pdf.pages),
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "subject": pdf.metadata.get("Subject", ""),
                        "creator": pdf.metadata.get("Creator", "")
                    }
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text() or ""
                        # Clean up the text
                        page_text = self._clean_text(page_text)
                        
                        if page_text.strip():
                            text_data["pages"].append({
                                "page_number": page_num,
                                "text": page_text,
                                "char_count": len(page_text)
                            })
                            text_data["full_text"] += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                            
            except Exception as e:
                logger.warning(f"pdfplumber failed for {file_path}: {e}")
                # Fallback to PyPDF2
                text_data["extraction_method"] = "pypdf2"
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_data["metadata"] = {
                        "num_pages": len(pdf_reader.pages),
                        "title": pdf_reader.metadata.get("/Title", "") if pdf_reader.metadata else "",
                        "author": pdf_reader.metadata.get("/Author", "") if pdf_reader.metadata else ""
                    }
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        page_text = self._clean_text(page_text)
                        
                        if page_text.strip():
                            text_data["pages"].append({
                                "page_number": page_num,
                                "text": page_text,
                                "char_count": len(page_text)
                            })
                            text_data["full_text"] += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            # Clean the full text
            text_data["full_text"] = self._clean_text(text_data["full_text"])
            text_data["total_chars"] = len(text_data["full_text"])
            
            logger.info(f"Successfully extracted text from PDF: {file_path.name}")
            return text_data
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            raise
    
    async def extract_text_from_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from image using OCR
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Open image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(img, config='--psm 6')
                text = self._clean_text(text)
                
                text_data = {
                    "full_text": text,
                    "pages": [{
                        "page_number": 1,
                        "text": text,
                        "char_count": len(text)
                    }],
                    "metadata": {
                        "num_pages": 1,
                        "image_mode": img.mode,
                        "image_size": img.size,
                        "extraction_method": "ocr"
                    },
                    "total_chars": len(text)
                }
                
                logger.info(f"Successfully extracted text from image: {file_path.name}")
                return text_data
                
        except Exception as e:
            logger.error(f"Failed to extract text from image {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of text chunks with metadata
        """
        if not text.strip():
            return []
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size
            if current_size + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(
                    current_chunk.strip(), 
                    chunk_index, 
                    metadata
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + para
                current_size = len(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size = len(current_chunk)
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), 
                chunk_index, 
                metadata
            ))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and section markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines, page breaks, or section headers
        paragraphs = re.split(r'\n\s*\n|\n\s*---\s*', text)
        
        # Filter out empty paragraphs and very short ones
        filtered_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Only keep substantial paragraphs
                filtered_paragraphs.append(para)
        
        return filtered_paragraphs
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last portion of text for overlap"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.chunk_overlap:
                overlap_text = sentence + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def _create_chunk(self, text: str, chunk_index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a chunk with metadata"""
        chunk = {
            "content": text,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "word_count": len(text.split()),
            "metadata": metadata or {}
        }
        
        # Add page number if available in metadata
        if metadata and "page_number" in metadata:
            chunk["page_number"] = metadata["page_number"]
        
        return chunk
    
    async def process_document(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """
        Main method to process any document type
        
        Args:
            file_path: Path to the document
            file_type: MIME type of the document
            
        Returns:
            Processed document data with chunks
        """
        try:
            # Extract text based on file type
            if file_type == "application/pdf":
                text_data = await self.extract_text_from_pdf(file_path)
            elif file_type.startswith("image/"):
                text_data = await self.extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create chunks from the extracted text
            chunks = []
            for page_data in text_data["pages"]:
                page_chunks = self.chunk_text(
                    page_data["text"], 
                    {"page_number": page_data["page_number"]}
                )
                chunks.extend(page_chunks)
            
            # Add chunk information to text_data
            text_data["chunks"] = chunks
            text_data["total_chunks"] = len(chunks)
            
            logger.info(f"Successfully processed document: {file_path.name} -> {len(chunks)} chunks")
            return text_data
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise

# Global instance
text_processor = TextProcessor()
