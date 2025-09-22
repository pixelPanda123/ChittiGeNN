"""
Test script for Day 1 ML Pipeline components
This script verifies that all ML pipeline services are working correctly
"""

import asyncio
import logging
from pathlib import Path
import tempfile
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_text_processor():
    """Test text processing functionality"""
    logger.info("🧪 Testing Text Processor...")
    
    try:
        from app.services.text_processor import text_processor
        
        # Test text chunking
        sample_text = """
        This is a sample document with multiple paragraphs. 
        It contains various sentences and information that should be chunked properly.
        
        The chunking algorithm should split this text into meaningful segments
        while maintaining context and readability.
        
        Each chunk should have appropriate size and overlap for optimal
        embedding generation and retrieval.
        """
        
        chunks = text_processor.chunk_text(sample_text)
        
        assert len(chunks) > 0, "No chunks generated"
        assert all("content" in chunk for chunk in chunks), "Missing content in chunks"
        assert all("chunk_index" in chunk for chunk in chunks), "Missing chunk_index"
        
        logger.info(f"✅ Text Processor: Generated {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"❌ Text Processor failed: {e}")
        return False

async def test_embedding_service():
    """Test embedding service functionality"""
    logger.info("🧪 Testing Embedding Service...")
    
    try:
        from app.services.embedding_service import embedding_service
        
        # Test model loading
        await embedding_service.load_model()
        model_info = await embedding_service.get_model_info()
        
        assert model_info["loaded"], "Model not loaded"
        assert model_info["embedding_dim"] > 0, "Invalid embedding dimension"
        
        # Test embedding generation
        sample_texts = [
            "This is a test document about machine learning.",
            "Artificial intelligence and natural language processing are fascinating topics.",
            "Vector embeddings help computers understand text semantics."
        ]
        
        embeddings = await embedding_service.generate_embeddings(sample_texts)
        
        assert len(embeddings) == len(sample_texts), "Wrong number of embeddings"
        assert embeddings.shape[1] == model_info["embedding_dim"], "Wrong embedding dimension"
        
        # Test similarity calculation
        similarity = await embedding_service.compute_similarity(embeddings[0], embeddings[1])
        assert 0 <= similarity <= 1, "Invalid similarity score"
        
        logger.info(f"✅ Embedding Service: Generated {len(embeddings)} embeddings with dim {embeddings.shape[1]}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding Service failed: {e}")
        return False

async def test_vector_store():
    """Test vector store functionality"""
    logger.info("🧪 Testing Vector Store...")
    
    try:
        from app.services.vector_store import vector_store
        
        # Initialize vector store
        await vector_store.initialize()
        
        # Test adding document chunks
        test_document_id = "test_doc_001"
        test_chunks = [
            {
                "content": "This is a test chunk about machine learning algorithms.",
                "chunk_index": 0,
                "char_count": 55,
                "word_count": 9,
                "embedding": [0.1] * 384,  # Mock embedding
                "embedding_model": "test-model"
            },
            {
                "content": "Natural language processing is a subset of AI.",
                "chunk_index": 1,
                "char_count": 45,
                "word_count": 7,
                "embedding": [0.2] * 384,  # Mock embedding
                "embedding_model": "test-model"
            }
        ]
        
        chunk_ids = await vector_store.add_document_chunks(test_document_id, test_chunks)
        assert len(chunk_ids) == len(test_chunks), "Wrong number of chunks added"
        
        # Test search
        search_results = await vector_store.search_similar(
            query="machine learning",
            top_k=5
        )
        
        assert len(search_results) >= 0, "Search returned invalid results"
        
        # Test document retrieval
        doc_chunks = await vector_store.get_document_chunks(test_document_id)
        assert len(doc_chunks) == len(test_chunks), "Wrong number of chunks retrieved"
        
        # Test deletion
        success = await vector_store.delete_document(test_document_id)
        assert success, "Failed to delete document"
        
        logger.info("✅ Vector Store: All operations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector Store failed: {e}")
        return False

async def test_search_engine():
    """Test search engine functionality"""
    logger.info("🧪 Testing Search Engine...")
    
    try:
        from app.services.search_engine import search_engine
        from app.services.vector_store import vector_store
        from app.services.embedding_service import embedding_service
        
        # Initialize services
        await search_engine.initialize()
        
        # Add some test data
        test_document_id = "search_test_doc"
        test_chunks = [
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "chunk_index": 0,
                "char_count": 60,
                "word_count": 9,
                "embedding": await embedding_service.generate_single_embedding("Machine learning is a subset of artificial intelligence."),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            {
                "content": "Deep learning uses neural networks with multiple layers.",
                "chunk_index": 1,
                "char_count": 55,
                "word_count": 8,
                "embedding": await embedding_service.generate_single_embedding("Deep learning uses neural networks with multiple layers."),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        ]
        
        await vector_store.add_document_chunks(test_document_id, test_chunks)
        
        # Test search
        search_results = await search_engine.search_documents(
            query="What is machine learning?",
            top_k=5
        )
        
        assert "results" in search_results, "Missing results in search response"
        assert "total_results" in search_results, "Missing total_results"
        assert "search_time_ms" in search_results, "Missing search_time_ms"
        
        # Test search suggestions
        suggestions = await search_engine.get_search_suggestions("What")
        assert isinstance(suggestions, list), "Suggestions should be a list"
        
        # Test analytics
        analytics = await search_engine.get_search_analytics()
        assert "vector_store_stats" in analytics, "Missing vector store stats"
        assert "embedding_model_info" in analytics, "Missing model info"
        
        # Cleanup
        await vector_store.delete_document(test_document_id)
        
        logger.info("✅ Search Engine: All operations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search Engine failed: {e}")
        return False

async def test_document_processor():
    """Test document processor functionality"""
    logger.info("🧪 Testing Document Processor...")
    
    try:
        from app.services.document_processor import document_processor
        
        # Test initialization
        await document_processor.initialize()
        
        # Test processing stats
        stats = await document_processor.get_processing_stats()
        assert "processing_config" in stats, "Missing processing config"
        assert "system_initialized" in stats, "Missing system status"
        
        # Test document validation
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is a test document for validation.")
            tmp_file_path = Path(tmp_file.name)
        
        validation_result = await document_processor.validate_document(tmp_file_path, "text/plain")
        assert "valid" in validation_result, "Missing validation result"
        
        # Cleanup
        tmp_file_path.unlink()
        
        logger.info("✅ Document Processor: All operations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Document Processor failed: {e}")
        return False

async def run_all_tests():
    """Run all ML pipeline tests"""
    logger.info("🚀 Starting Day 1 ML Pipeline Tests...")
    
    tests = [
        ("Text Processor", test_text_processor),
        ("Embedding Service", test_embedding_service),
        ("Vector Store", test_vector_store),
        ("Search Engine", test_search_engine),
        ("Document Processor", test_document_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Day 1 ML Pipeline tests passed! Ready for integration.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    asyncio.run(run_all_tests())
