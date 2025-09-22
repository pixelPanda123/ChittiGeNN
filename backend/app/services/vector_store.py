"""
ChromaDB vector store service for document embeddings and similarity search
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
import json

from app.core.config import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB vector store for document embeddings and similarity search"""
    
    def __init__(self):
        self.collection_name = "documents"
        self.client = None
        self.collection = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        if self._initialized:
            return
            
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll handle embeddings ourselves
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"description": "Document chunks with embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_document_chunks(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add document chunks to the vector store
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of processed chunks with embeddings
            
        Returns:
            List of chunk IDs that were added
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return []
        
        try:
            # Prepare data for ChromaDB
            chunk_ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in chunks:
                # Generate unique chunk ID
                chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"
                chunk_ids.append(chunk_id)
                
                # Extract embedding
                embedding = chunk.get("embedding", [])
                if not embedding:
                    logger.warning(f"No embedding found for chunk {chunk_id}")
                    continue
                    
                embeddings.append(embedding)
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "char_count": chunk["char_count"],
                    "word_count": chunk["word_count"],
                    "embedding_model": chunk.get("embedding_model", ""),
                }
                
                # Add page number if available
                if "page_number" in chunk:
                    metadata["page_number"] = chunk["page_number"]
                
                # Add any additional metadata from the chunk
                if "metadata" in chunk:
                    metadata.update(chunk["metadata"])
                
                metadatas.append(metadata)
                documents.append(chunk["content"])
            
            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}")
            raise
    
    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.generate_single_embedding(query)
            if len(query_embedding) == 0:
                return []
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results["distances"][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_threshold and similarity_score < similarity_threshold:
                        continue
                    
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "similarity_score": similarity_score,
                        "metadata": results["metadatas"][0][i]
                    }
                    search_results.append(result)
            
            logger.info(f"Found {len(search_results)} similar chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document chunks
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    chunk = {
                        "chunk_id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    }
                    chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.info(f"No chunks found for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return False
    
    async def update_document_chunks(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Update document chunks (delete old, add new)
        
        Args:
            document_id: Document identifier
            chunks: New chunks to add
            
        Returns:
            List of new chunk IDs
        """
        try:
            # Delete existing chunks
            await self.delete_document(document_id)
            
            # Add new chunks
            return await self.add_document_chunks(document_id, chunks)
            
        except Exception as e:
            logger.error(f"Failed to update document chunks: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if not self._initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=100)
            
            stats = {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "sample_size": len(sample_results["ids"]) if sample_results["ids"] else 0
            }
            
            # Analyze document distribution
            if sample_results["metadatas"]:
                document_ids = set()
                for metadata in sample_results["metadatas"]:
                    document_ids.add(metadata.get("document_id", ""))
                stats["estimated_documents"] = len(document_ids)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def reset_collection(self) -> bool:
        """Reset the entire collection (use with caution)"""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "Document chunks with embeddings"}
            )
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    async def backup_collection(self, backup_path: Path) -> bool:
        """Backup collection data to a file"""
        try:
            if not backup_path.parent.exists():
                backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all data from collection
            all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            backup_data = {
                "collection_name": self.collection_name,
                "data": all_data,
                "stats": await self.get_collection_stats()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Collection backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            return False

# Global instance
vector_store = VectorStore()
