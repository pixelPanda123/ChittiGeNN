"""
Embedding service for generating and managing document embeddings
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import pickle
import json

from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles embedding generation and management using sentence-transformers"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_cache = {}
        self._model_loaded = False
        
    async def load_model(self) -> None:
        """Load the sentence transformer model"""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if not self._model_loaded:
            await self.load_model()
        
        if not texts:
            return np.array([])
        
        try:
            # Remove empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return np.array([])
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=batch_size
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 10,
        threshold: float = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = await self.compute_similarity(query_embedding, candidate)
                if threshold is None or similarity >= threshold:
                    similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    async def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], file_path: Path) -> None:
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Embedding vectors
            metadata: List of metadata for each embedding
            file_path: Path to save the data
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "embeddings": embeddings.tolist(),
                "metadata": metadata,
                "model_name": self.model_name,
                "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    async def load_embeddings(self, file_path: Path) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load embeddings and metadata from disk
        
        Args:
            file_path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        try:
            if not file_path.exists():
                logger.warning(f"Embeddings file not found: {file_path}")
                return np.array([]), []
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = np.array(data["embeddings"])
            metadata = data["metadata"]
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return np.array([]), []
    
    async def process_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process document chunks and add embeddings
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings added
        """
        try:
            if not chunks:
                return []
            
            # Extract text content from chunks
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunk = chunk.copy()
                processed_chunk["embedding"] = embeddings[i].tolist()
                processed_chunk["embedding_model"] = self.model_name
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self._model_loaded:
            await self.load_model()
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "loaded": self._model_loaded
        }
    
    async def clear_cache(self) -> None:
        """Clear the embeddings cache"""
        self.embeddings_cache.clear()
        logger.info("Embeddings cache cleared")

# Global instance
embedding_service = EmbeddingService()
