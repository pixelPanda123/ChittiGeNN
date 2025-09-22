# ML Pipeline Integration Guide

## Overview
This guide shows how to integrate with the ML pipeline services implemented by Developer B. All services are production-ready and include comprehensive error handling, logging, and async support.

## 🚀 Quick Start

### 1. Initialize Services
```python
from app.services.document_processor import document_processor
from app.services.search_engine import search_engine

# Initialize all ML services
await document_processor.initialize()
await search_engine.initialize()
```

### 2. Process a Document
```python
# Process a document through the entire pipeline
result = await document_processor.process_document(document_model)

if result["success"]:
    print(f"Processed {result['chunks_created']} chunks in {result['processing_time_seconds']}s")
else:
    print(f"Processing failed: {result['error']}")
```

### 3. Search Documents
```python
# Search for relevant content
search_results = await search_engine.search_documents(
    query="machine learning algorithms",
    top_k=10,
    similarity_threshold=0.7
)

print(f"Found {search_results['total_results']} results")
for result in search_results['results']:
    print(f"Score: {result['similarity_score']} - {result['snippet']}")
```

## 📋 Available Services

### 1. Document Processor (`document_processor.py`)
**Purpose**: Main orchestration service for document processing pipeline

**Key Methods**:
- `process_document(document)` - Process document through entire pipeline
- `reprocess_document(document)` - Reprocess failed documents
- `validate_document(file_path, mime_type)` - Validate documents before processing
- `get_processing_stats()` - Get system statistics
- `cleanup_document(document)` - Clean up document data

**Example Usage**:
```python
from app.services.document_processor import document_processor

# Validate document before processing
validation = await document_processor.validate_document(file_path, "application/pdf")
if not validation["valid"]:
    raise ValueError(f"Document validation failed: {validation['errors']}")

# Process document
result = await document_processor.process_document(document)
```

### 2. Search Engine (`search_engine.py`)
**Purpose**: Semantic search with ranking and filtering

**Key Methods**:
- `search_documents(query, filters, top_k, similarity_threshold)` - Main search method
- `search_with_context(query, context_document_ids)` - Search with document context
- `get_search_suggestions(partial_query)` - Get search suggestions
- `get_search_analytics()` - Get search statistics
- `get_document_summary(document_id)` - Get document summary

**Example Usage**:
```python
from app.services.search_engine import search_engine

# Basic search
results = await search_engine.search_documents(
    query="artificial intelligence",
    top_k=5,
    similarity_threshold=0.8
)

# Search with filters
filtered_results = await search_engine.search_documents(
    query="machine learning",
    filters={"document_id": "specific_doc_id"},
    top_k=10
)
```

### 3. Text Processor (`text_processor.py`)
**Purpose**: Text extraction and intelligent chunking

**Key Methods**:
- `extract_text_from_pdf(file_path)` - Extract text from PDF
- `extract_text_from_image(file_path)` - Extract text from images (OCR)
- `chunk_text(text, metadata)` - Split text into chunks
- `process_document(file_path, file_type)` - Process any document type

**Example Usage**:
```python
from app.services.text_processor import text_processor

# Extract text from PDF
text_data = await text_processor.extract_text_from_pdf(pdf_path)
print(f"Extracted {text_data['total_chars']} characters from {len(text_data['pages'])} pages")

# Create chunks
chunks = text_processor.chunk_text(text_data["full_text"])
print(f"Created {len(chunks)} chunks")
```

### 4. Embedding Service (`embedding_service.py`)
**Purpose**: Generate and manage document embeddings

**Key Methods**:
- `generate_embeddings(texts, batch_size)` - Generate embeddings for text list
- `generate_single_embedding(text)` - Generate embedding for single text
- `compute_similarity(embedding1, embedding2)` - Compute cosine similarity
- `load_model()` - Load sentence-transformers model

**Example Usage**:
```python
from app.services.embedding_service import embedding_service

# Generate embeddings
texts = ["Document 1 content", "Document 2 content"]
embeddings = await embedding_service.generate_embeddings(texts)

# Compute similarity
similarity = await embedding_service.compute_similarity(embeddings[0], embeddings[1])
```

### 5. Vector Store (`vector_store.py`)
**Purpose**: ChromaDB integration for vector storage and retrieval

**Key Methods**:
- `add_document_chunks(document_id, chunks)` - Add chunks to vector store
- `search_similar(query, top_k, filter_metadata)` - Search similar chunks
- `get_document_chunks(document_id)` - Get all chunks for document
- `delete_document(document_id)` - Delete document chunks
- `get_collection_stats()` - Get vector store statistics

**Example Usage**:
```python
from app.services.vector_store import vector_store

# Add document chunks
chunk_ids = await vector_store.add_document_chunks("doc_123", processed_chunks)

# Search similar content
results = await vector_store.search_similar(
    query="machine learning",
    top_k=5,
    filter_metadata={"document_id": "doc_123"}
)
```

## 🔧 Configuration

All services use the centralized configuration in `app/core/config.py`:

```python
from app.core.config import settings

# Key settings
CHUNK_SIZE = settings.CHUNK_SIZE  # 1000 characters
CHUNK_OVERLAP = settings.CHUNK_OVERLAP  # 200 characters
EMBEDDING_MODEL = settings.EMBEDDING_MODEL  # "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEARCH_RESULTS = settings.MAX_SEARCH_RESULTS  # 10
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD  # 0.7
```

## 📊 Error Handling

All services include comprehensive error handling:

```python
try:
    result = await document_processor.process_document(document)
    if result["success"]:
        # Handle success
        pass
    else:
        # Handle processing failure
        error_msg = result["error"]
except Exception as e:
    # Handle service failure
    logger.error(f"Service error: {e}")
```

## 🚀 Performance Tips

1. **Batch Processing**: Use batch methods for multiple operations
2. **Async Operations**: All services are async - use `await` properly
3. **Memory Management**: Services handle memory optimization automatically
4. **Caching**: Embeddings are cached to avoid recomputation

## 🧪 Testing

Run the test suite to verify all services:

```bash
cd backend
python test_ml_pipeline.py
```

## 📝 Integration Examples

### For Developer A (API Endpoints)
```python
# In your API endpoints
from app.services.document_processor import document_processor
from app.services.search_engine import search_engine

@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    # Process document
    result = await document_processor.process_document(document)
    return {"status": "processed", "chunks": result["chunks_created"]}

@app.get("/search")
async def search_documents(query: str):
    # Search documents
    results = await search_engine.search_documents(query)
    return results
```

### For Developer C (LLM Integration)
```python
# Get relevant context for LLM
search_results = await search_engine.search_documents(query)
context_chunks = [result["content"] for result in search_results["results"]]

# Combine context for LLM prompt
context = "\n\n".join(context_chunks)
llm_prompt = f"Based on the following context:\n{context}\n\nAnswer: {query}"
```

## 🔍 Troubleshooting

### Common Issues:

1. **Model Loading Errors**: Ensure sentence-transformers is installed
2. **ChromaDB Issues**: Check if vector_db directory is writable
3. **Memory Issues**: Reduce batch sizes for large documents
4. **File Processing Errors**: Validate file types and sizes first

### Logging:
All services use structured logging. Check logs for detailed error information:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📞 Support

For issues with the ML pipeline:
1. Check the logs for detailed error messages
2. Run the test suite to identify problems
3. Verify all dependencies are installed
4. Check configuration settings

---

**Status**: ✅ Production Ready  
**Last Updated**: Day 1 Complete  
**Maintained By**: Developer B (ML Pipeline Team)
