# ChittiGeNN - Team Task Assignments

## Current Project Status ✅

### What's Already Set Up:
- ✅ **FastAPI backend structure** with proper organization
- ✅ **React frontend** with TypeScript and modern dependencies
- ✅ **Configuration system** with all necessary settings
- ✅ **Database models** (document.py exists)
- ✅ **CORS middleware** and static file serving
- ✅ **Dependencies** for ML, document processing, and audio
- ✅ **Project structure** matches the development plan

### What Needs to be Built:
- ❌ **API endpoints** (documents, search, health) - files exist but are empty
- ❌ **ML pipeline** components (embeddings, vector search)
- ❌ **Frontend components** (upload, search, results)
- ❌ **Database setup** and migrations
- ❌ **LLM integration**

---

## Team Assignments & Specific Tasks

### 🔧 **Backend Team**

#### **Developer A: Core API & File Processing**
**Current Status**: FastAPI structure exists, need to implement endpoints

**Day 1 Tasks:**
```bash
# Create these files:
backend/app/api/endpoints/documents.py
backend/app/api/endpoints/search.py  
backend/app/api/endpoints/health.py
backend/app/api/models.py
backend/app/database.py
```

**Specific Implementation:**
1. **Database Setup** (2 hours)
   - Create SQLAlchemy models for documents and chunks
   - Set up database connection and session management
   - Create migration scripts

2. **File Upload API** (3 hours)
   - Implement POST `/api/v1/documents/upload`
   - Add file validation (size, type, corruption)
   - Store document metadata in database
   - Return upload confirmation with document ID

3. **Document Management API** (2 hours)
   - Implement GET `/api/v1/documents` (list all)
   - Implement GET `/api/v1/documents/{id}` (get specific)
   - Implement DELETE `/api/v1/documents/{id}`
   - Add document status tracking

4. **Health Check API** (1 hour)
   - Implement GET `/api/v1/health`
   - Check database connectivity
   - Check ML model availability
   - Return system status

**Cursor Prompt for Developer A:**
```
"Implement the FastAPI endpoints for document management. I need:
1. Complete documents.py with upload, list, get, delete endpoints
2. Complete health.py with system health checks
3. Create database.py with SQLAlchemy setup
4. Create api/models.py with Pydantic request/response models
5. Add proper error handling and validation
6. Include async file upload with progress tracking
7. Add database models for Document and DocumentChunk

Use the existing config.py settings and maintain the current project structure."
```

#### **Developer B: ML Pipeline & Vector Search (You)**
**Current Status**: Dependencies installed, need to implement ML components

**Day 1 Tasks:**
```bash
# Create these files:
backend/app/services/embedding_service.py
backend/app/services/vector_store.py
backend/app/services/text_processor.py
backend/app/models/chunk.py
```

**Specific Implementation:**
1. **Text Processing** (2 hours)
   - Implement text chunking with configurable size/overlap
   - Add PDF text extraction using existing PyPDF2/pdfplumber
   - Create document preprocessing pipeline
   - Handle different file types (PDF, images, audio)

2. **Embedding Service** (3 hours)
   - Set up sentence-transformers model loading
   - Implement batch embedding generation
   - Add embedding caching and persistence
   - Create embedding similarity calculations

3. **Vector Store** (2 hours)
   - Implement ChromaDB integration (already in requirements)
   - Create add/remove/update operations for embeddings
   - Implement similarity search with metadata filtering
   - Add index persistence and backup

4. **Search Engine** (1 hour)
   - Create search API endpoint
   - Implement query embedding generation
   - Add result ranking and scoring
   - Return formatted search results

**Cursor Prompt for Developer B:**
```
"Implement the ML pipeline for document embeddings and vector search. I need:
1. Text processing service with chunking and PDF extraction
2. Embedding service using sentence-transformers/all-MiniLM-L6-v2
3. ChromaDB vector store integration for similarity search
4. Search engine with configurable similarity thresholds
5. Batch processing capabilities for large documents
6. Proper error handling for ML operations

Use the existing config.py settings and integrate with the document models."
```

#### **Developer C: LLM Integration & Response Generation**
**Current Status**: Need to set up local LLM integration

**Day 1 Tasks:**
```bash
# Create these files:
backend/app/services/llm_service.py
backend/app/services/response_generator.py
backend/app/templates/prompts.py
```

**Specific Implementation:**
1. **LLM Setup** (3 hours)
   - Install and configure Ollama locally
   - Download TinyLlama or similar lightweight model
   - Create LLM service with async response generation
   - Add model loading and caching

2. **Prompt Engineering** (2 hours)
   - Create prompt templates for document Q&A
   - Add context window management
   - Implement citation insertion system
   - Create response quality filters

3. **Response Generation** (2 hours)
   - Implement context assembly from search results
   - Create streaming response API
   - Add response caching for common queries
   - Handle response validation and safety

4. **API Integration** (1 hour)
   - Create chat/query endpoints
   - Add response history tracking
   - Implement feedback collection

**Cursor Prompt for Developer C:**
```
"Set up local LLM integration for response generation. I need:
1. Ollama integration with TinyLlama model
2. Prompt templates for document-based Q&A with citations
3. Context assembly from search results with proper truncation
4. Streaming response generation API
5. Response caching and quality assessment
6. Integration with the search results

Focus on local deployment and efficient resource usage."
```

---

### 🎨 **Frontend Team**

#### **Developer D: Core UI Components & File Upload**
**Current Status**: React app exists with good dependencies, need to implement components

**Day 1 Tasks:**
```bash
# Create these files:
frontend/src/components/FileUpload.tsx
frontend/src/components/DocumentList.tsx
frontend/src/components/Layout.tsx
frontend/src/services/api.ts
frontend/src/types/document.ts
```

**Specific Implementation:**
1. **Layout & Navigation** (1 hour)
   - Create main layout with header and sidebar
   - Add navigation between upload and search
   - Implement responsive design
   - Add loading states and error boundaries

2. **File Upload Component** (3 hours)
   - Implement drag-and-drop functionality
   - Add file validation on frontend
   - Create upload progress indicators
   - Add file preview and thumbnail generation

3. **Document Management** (2 hours)
   - Create document list with status indicators
   - Add document actions (view, delete, reprocess)
   - Implement document status updates
   - Add search and filtering

4. **API Integration** (2 hours)
   - Create API service functions
   - Add error handling and retry logic
   - Implement request/response types
   - Add loading states for all operations

**Cursor Prompt for Developer D:**
```
"Create the core UI components for document management. I need:
1. File upload component with drag-and-drop and progress tracking
2. Document list component with status indicators and actions
3. Main layout with navigation and responsive design
4. API service integration with proper error handling
5. TypeScript interfaces for all data types
6. Loading states and user feedback

Use the existing dependencies: react-dropzone, tailwindcss, react-hot-toast.
Integrate with the backend API endpoints."
```

#### **Developer E: Search Interface & Results Display**
**Current Status**: Need to implement search components

**Day 1 Tasks:**
```bash
# Create these files:
frontend/src/components/SearchBox.tsx
frontend/src/components/SearchResults.tsx
frontend/src/components/DocumentViewer.tsx
frontend/src/hooks/useSearch.ts
```

**Specific Implementation:**
1. **Search Interface** (2 hours)
   - Create search input with real-time functionality
   - Add search suggestions and autocomplete
   - Implement debounced search requests
   - Add search history and recent queries

2. **Results Display** (3 hours)
   - Create search results list with relevance scores
   - Add document snippet highlighting
   - Implement result filtering and sorting
   - Add pagination or infinite scroll

3. **Document Viewer** (2 hours)
   - Create document preview modal
   - Add PDF viewer integration
   - Implement citation highlighting
   - Add document navigation

4. **Search State Management** (1 hour)
   - Create search hooks and context
   - Add search result caching
   - Implement search analytics
   - Add keyboard shortcuts

**Cursor Prompt for Developer E:**
```
"Build the search interface and results display. I need:
1. Search input with real-time search and debouncing
2. Search results display with relevance scores and snippets
3. Document viewer with PDF preview and citation highlighting
4. Search state management with React hooks
5. Filtering, sorting, and pagination for results
6. Mobile-responsive design and keyboard navigation

Use react-markdown for content display and integrate with the search API."
```

#### **Developer F: Integration & Polish**
**Current Status**: Need to integrate all components and add polish

**Day 1 Tasks:**
```bash
# Create these files:
frontend/src/App.tsx
frontend/src/pages/Dashboard.tsx
frontend/src/pages/Search.tsx
frontend/src/context/AppContext.tsx
```

**Specific Implementation:**
1. **App Integration** (2 hours)
   - Integrate all components into main app
   - Add routing between pages
   - Create app-wide state management
   - Add error boundaries and fallbacks

2. **User Experience** (3 hours)
   - Add loading states and transitions
   - Implement user feedback and notifications
   - Add keyboard shortcuts and accessibility
   - Create help tooltips and onboarding

3. **Performance Optimization** (2 hours)
   - Add component memoization
   - Implement virtual scrolling for large lists
   - Add image lazy loading
   - Optimize bundle size

4. **Testing & Polish** (1 hour)
   - Add component testing setup
   - Test responsive design
   - Fix any UI inconsistencies
   - Prepare for demo

**Cursor Prompt for Developer F:**
```
"Integrate all frontend components and add final polish. I need:
1. Complete app integration with routing and state management
2. User experience enhancements with loading states and feedback
3. Performance optimization with memoization and lazy loading
4. Responsive design testing and accessibility improvements
5. Component testing setup and demo preparation

Focus on creating a polished, production-ready interface."
```

---

## 🚀 **Day 1 Priority Order**

### **Morning (9 AM - 12 PM)**
1. **Developer A**: Database setup and basic endpoints
2. **Developer B**: Text processing and embedding service setup
3. **Developer C**: Ollama installation and basic LLM setup
4. **Developer D**: Layout and file upload component
5. **Developer E**: Search interface foundation
6. **Developer F**: App structure and routing

### **Afternoon (1 PM - 5 PM)**
1. **Developer A**: Complete document management API
2. **Developer B**: Vector store integration and search engine
3. **Developer C**: Response generation and API integration
4. **Developer D**: Document management UI and API integration
5. **Developer E**: Results display and document viewer
6. **Developer F**: Integration and user experience polish

---

## 🔄 **Integration Checkpoints**

### **End of Day 1**
- [ ] Backend: All API endpoints implemented and tested
- [ ] Frontend: All components created and integrated
- [ ] Integration: File upload → processing → search workflow working
- [ ] Demo: Basic end-to-end functionality demonstrated

### **Daily Standup Questions**
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers or dependencies?
4. API contract changes needed?
5. Integration issues or concerns?

---

## 📋 **Success Criteria**

### **Developer A (API)**
- [ ] All endpoints return proper responses
- [ ] File upload handles large files
- [ ] Database operations are optimized
- [ ] Error handling covers edge cases

### **Developer B (ML Pipeline)**
- [ ] Text extraction works for various PDFs
- [ ] Embeddings are generated correctly
- [ ] Vector search returns relevant results
- [ ] Performance is acceptable for large documents

### **Developer C (LLM)**
- [ ] Local LLM runs without issues
- [ ] Responses are coherent and cited
- [ ] Streaming works smoothly
- [ ] Context management handles long documents

### **Developer D (Upload UI)**
- [ ] Upload works for all supported file types
- [ ] Progress tracking is accurate
- [ ] UI is responsive and accessible
- [ ] Error states are user-friendly

### **Developer E (Search UI)**
- [ ] Search is fast and responsive
- [ ] Results are well-formatted and relevant
- [ ] Document viewer works for PDFs
- [ ] Mobile experience is optimized

### **Developer F (Integration)**
- [ ] All components work together seamlessly
- [ ] Performance is optimized
- [ ] User experience is polished
- [ ] Demo is ready for presentation

---

**Ready to start? Each developer should begin with their Day 1 morning tasks and check in at the afternoon standup!**
