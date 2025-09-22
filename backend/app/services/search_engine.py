"""
Search engine service that combines vector search with metadata filtering and ranking
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

from app.core.config import settings
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SearchEngine:
    """Main search engine that orchestrates vector search and result ranking"""
    
    def __init__(self):
        self.max_results = settings.MAX_SEARCH_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
    async def initialize(self) -> None:
        """Initialize all required services"""
        try:
            await vector_store.initialize()
            await embedding_service.load_model()
            logger.info("Search engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main search method that performs semantic search across documents
        
        Args:
            query: Search query text
            filters: Optional metadata filters (document_id, date_range, etc.)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results with metadata and ranking information
        """
        try:
            start_time = datetime.now()
            
            # Use provided parameters or defaults
            top_k = top_k or self.max_results
            similarity_threshold = similarity_threshold or self.similarity_threshold
            
            # Perform vector search
            search_results = await vector_store.search_similar(
                query=query,
                top_k=top_k * 2,  # Get more results for better ranking
                filter_metadata=filters,
                similarity_threshold=similarity_threshold
            )
            
            if not search_results:
                return {
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "search_time_ms": 0,
                    "filters_applied": filters or {}
                }
            
            # Apply additional ranking and filtering
            ranked_results = await self._rank_results(search_results, query)
            
            # Limit results to requested top_k
            final_results = ranked_results[:top_k]
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Format results
            formatted_results = await self._format_search_results(final_results)
            
            logger.info(f"Search completed: {len(formatted_results)} results in {search_time:.2f}ms")
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_time_ms": round(search_time, 2),
                "filters_applied": filters or {},
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _rank_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply additional ranking logic to search results
        
        Args:
            results: Raw search results from vector store
            query: Original search query
            
        Returns:
            Ranked results
        """
        try:
            # Sort by similarity score (already done by ChromaDB, but ensure it's correct)
            ranked_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
            
            # Apply additional ranking factors
            for i, result in enumerate(ranked_results):
                # Add ranking position
                result["rank"] = i + 1
                
                # Calculate content quality score
                content_score = self._calculate_content_quality(result["content"])
                result["content_quality_score"] = content_score
                
                # Calculate combined score
                combined_score = (
                    result["similarity_score"] * 0.7 +  # Semantic similarity (70%)
                    content_score * 0.3  # Content quality (30%)
                )
                result["combined_score"] = combined_score
            
            # Re-sort by combined score
            ranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Failed to rank results: {e}")
            return results
    
    def _calculate_content_quality(self, content: str) -> float:
        """
        Calculate content quality score based on various factors
        
        Args:
            content: Text content to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if not content or not content.strip():
                return 0.0
            
            score = 0.0
            
            # Length factor (optimal length gets higher score)
            length = len(content.strip())
            if 100 <= length <= 1000:
                score += 0.3
            elif 50 <= length < 100 or 1000 < length <= 2000:
                score += 0.2
            else:
                score += 0.1
            
            # Sentence structure (complete sentences get higher score)
            sentences = content.split('.')
            complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
            if complete_sentences > 0:
                score += min(0.3, complete_sentences * 0.1)
            
            # Word diversity (avoid repetitive content)
            words = content.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                diversity = unique_words / len(words)
                score += diversity * 0.2
            
            # Information density (avoid too much whitespace)
            non_space_chars = len([c for c in content if not c.isspace()])
            if len(content) > 0:
                density = non_space_chars / len(content)
                score += density * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Failed to calculate content quality: {e}")
            return 0.5  # Default score
    
    async def _format_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results for API response
        
        Args:
            results: Raw search results
            
        Returns:
            Formatted results
        """
        try:
            formatted_results = []
            
            for result in results:
                formatted_result = {
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "similarity_score": round(result["similarity_score"], 4),
                    "combined_score": round(result["combined_score"], 4),
                    "rank": result["rank"],
                    "metadata": {
                        "document_id": result["metadata"].get("document_id", ""),
                        "chunk_index": result["metadata"].get("chunk_index", 0),
                        "page_number": result["metadata"].get("page_number", None),
                        "char_count": result["metadata"].get("char_count", 0),
                        "word_count": result["metadata"].get("word_count", 0)
                    }
                }
                
                # Add content snippet for display
                content = result["content"]
                if len(content) > 200:
                    formatted_result["snippet"] = content[:200] + "..."
                else:
                    formatted_result["snippet"] = content
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
            return results
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """
        Get a summary of a document including all its chunks
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document summary with chunk information
        """
        try:
            chunks = await vector_store.get_document_chunks(document_id)
            
            if not chunks:
                return {
                    "document_id": document_id,
                    "total_chunks": 0,
                    "total_characters": 0,
                    "total_words": 0,
                    "chunks": []
                }
            
            total_chars = sum(chunk["metadata"].get("char_count", 0) for chunk in chunks)
            total_words = sum(chunk["metadata"].get("word_count", 0) for chunk in chunks)
            
            return {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "total_characters": total_chars,
                "total_words": total_words,
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"]
                    }
                    for chunk in chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return {"error": str(e)}
    
    async def search_with_context(
        self,
        query: str,
        context_document_ids: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search with additional context from specific documents
        
        Args:
            query: Search query
            context_document_ids: List of document IDs to focus search on
            **kwargs: Additional search parameters
            
        Returns:
            Search results with context information
        """
        try:
            # If context documents are specified, add them as filters
            filters = kwargs.get("filters", {})
            if context_document_ids:
                filters["document_id"] = {"$in": context_document_ids}
                kwargs["filters"] = filters
            
            # Perform search
            results = await self.search_documents(query, **kwargs)
            
            # Add context information
            results["context_documents"] = context_document_ids or []
            results["context_applied"] = bool(context_document_ids)
            
            return results
            
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            raise
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """
        Get search suggestions based on partial query
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of search suggestions
        """
        try:
            # For now, return simple suggestions based on common patterns
            # In a production system, this would use query logs or auto-complete data
            
            suggestions = [
                "What is",
                "How to",
                "Explain",
                "Find information about",
                "Summary of"
            ]
            
            # Filter suggestions based on partial query
            if partial_query:
                suggestions = [
                    s for s in suggestions 
                    if s.lower().startswith(partial_query.lower())
                ]
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search engine analytics and statistics"""
        try:
            stats = await vector_store.get_collection_stats()
            model_info = await embedding_service.get_model_info()
            
            return {
                "vector_store_stats": stats,
                "embedding_model_info": model_info,
                "search_config": {
                    "max_results": self.max_results,
                    "similarity_threshold": self.similarity_threshold,
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {"error": str(e)}

# Global instance
search_engine = SearchEngine()
