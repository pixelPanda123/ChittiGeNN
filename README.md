# ChittiGeNN - Offline Multimodal RAG System

An **offline-first document intelligence platform** that allows users to upload various file types (PDFs, images, audio recordings) and query them using natural language - all while keeping data completely private and local.

## 🎯 Core Problem Solved

Organizations handling sensitive information (government, healthcare, legal, research) need to:
- Search across different types of documents intelligently
- Get AI-powered answers with verifiable citations
- Maintain complete data privacy (no cloud dependency)
- Process multiple file formats in one unified system

## 🚀 Key Features

- **100% Offline**: No internet required after initial setup
- **Multimodal**: Handles text documents, images, and audio in one system
- **Citation-Backed**: Every answer includes references to original sources
- **Cross-Modal Search**: Find text mentions of topics that appear in images or audio
- **Privacy-First**: All processing happens locally on your machine

## 🏗️ Architecture

```
ChittiGeNN/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints and models
│   │   ├── core/           # Core configuration and settings
│   │   ├── models/         # Database and ML models
│   │   ├── services/       # Business logic services
│   │   └── utils/          # Utility functions
│   ├── data/               # Local data storage
│   │   ├── uploads/        # Original uploaded files
│   │   ├── processed/      # Processed content
│   │   ├── embeddings/     # Vector embeddings
│   │   └── vector_db/      # Vector database storage
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API services
│   │   └── utils/          # Frontend utilities
│   └── public/             # Static assets
├── docs/                   # Documentation
├── scripts/                # Deployment and utility scripts
└── config/                 # Configuration files
```

## 🛠️ Tech Stack

- **Backend**: Python + FastAPI + Local ML models
- **Frontend**: React + TypeScript + Modern UI
- **AI Models**: Sentence transformers + Local LLM
- **Storage**: Local vector database + SQLite
- **Processing**: OCR (Tesseract), Speech-to-Text (Whisper)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pixelPanda123/ChittiGeNN.git
   cd ChittiGeNN
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m uvicorn app.main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 📋 Development Roadmap

### Phase 1: Basic Prototype
- [ ] PDF upload and text extraction
- [ ] Basic search functionality
- [ ] Simple Q&A responses
- [ ] Local vector database setup

### Phase 2: Multimodal System
- [ ] Image OCR processing
- [ ] Audio transcription
- [ ] Cross-modal search
- [ ] Professional UI/UX

### Phase 3: Enterprise Features
- [ ] Advanced security features
- [ ] Batch processing
- [ ] User management
- [ ] Analytics dashboard



---

**Made with ❤️ for privacy-conscious organizations**
