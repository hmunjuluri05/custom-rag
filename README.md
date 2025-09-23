# Custom RAG System

A modern, full-stack Retrieval-Augmented Generation (RAG) system with intelligent document processing, semantic search, and AI-powered chat interface.

## 🌟 Key Features

- **📄 Multi-format document support** - PDF, DOCX, XLSX, TXT
- **🔍 Semantic search** - Vector-based similarity search with 16 embedding models
- **🤖 AI-powered responses** - OpenAI GPT & Google Gemini integration
- **💬 Real-time chat** - WebSocket-based instant messaging
- **🎛️ Admin panel** - Document management, model configuration, system monitoring
- **🛡️ Permanent deletion** - Complete cleanup of documents and files
- **📊 Visual interface** - Modern Bootstrap UI with drag-and-drop uploads

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI1[Chat Interface<br/>WebSocket]
        UI2[Admin Panel<br/>Upload & Management]
    end

    subgraph "API Layer"
        API1[FastAPI Server<br/>REST + WebSocket]
        API2[Route Handlers<br/>Documents, Query, System]
    end

    subgraph "Core Services"
        RAG[RAG System<br/>Orchestrator]
        CHAT[Chat Service<br/>Query Processing]
        FILE[File Service<br/>Upload Handler]
    end

    subgraph "Processing Layer"
        DOC[Document Processor<br/>Text Extraction]
        CHUNK[Text Chunking<br/>9 Strategies]
        EMB[Embedding Models<br/>16 Models Available]
    end

    subgraph "Storage Layer"
        VDB[(ChromaDB<br/>Vector Database)]
        FS[(File System<br/>uploads/)]
    end

    subgraph "External Services"
        OPENAI[OpenAI API<br/>GPT Models]
        GOOGLE[Google API<br/>Gemini Models]
    end

    UI1 <--> API1
    UI2 <--> API1
    API1 --> API2
    API2 --> RAG
    API2 --> CHAT
    API2 --> FILE

    RAG --> DOC
    RAG --> CHUNK
    RAG --> EMB
    RAG --> VDB

    FILE --> FS
    CHAT --> RAG

    EMB -.-> OPENAI
    EMB -.-> GOOGLE
    CHAT -.-> OPENAI
    CHAT -.-> GOOGLE

    style RAG fill:#e1f5fe
    style VDB fill:#f3e5f5
    style OPENAI fill:#fff3e0
    style GOOGLE fill:#e8f5e8
```

## 🔄 Workflow Sequence Diagrams

### Document Upload & Processing Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Admin Panel
    participant API as FastAPI
    participant File as File Service
    participant RAG as RAG System
    participant Doc as Document Processor
    participant Emb as Embedding Model
    participant VDB as ChromaDB
    participant FS as File System

    User->>UI: Upload documents
    UI->>API: POST /api/upload-documents/
    API->>File: Process upload
    File->>FS: Save files
    File->>RAG: Add documents

    loop For each document
        RAG->>Doc: Extract text
        Doc-->>RAG: Return text content
        RAG->>RAG: Apply chunking strategy
        RAG->>Emb: Generate embeddings
        Emb-->>RAG: Return vectors
        RAG->>VDB: Store vectors + metadata
    end

    RAG-->>API: Success response
    API-->>UI: Upload complete
    UI-->>User: Show success + document list
```

### Query & Response Workflow

```mermaid
sequenceDiagram
    participant User
    participant Chat as Chat Interface
    participant WS as WebSocket
    participant API as FastAPI
    participant ChatSvc as Chat Service
    participant RAG as RAG System
    participant Emb as Embedding Model
    participant VDB as ChromaDB
    participant LLM as LLM Provider

    User->>Chat: Type question
    Chat->>WS: Send query
    WS->>API: WebSocket message
    API->>ChatSvc: Process query
    ChatSvc->>RAG: Query with LLM

    RAG->>Emb: Convert query to vector
    Emb-->>RAG: Query embedding
    RAG->>VDB: Similarity search
    VDB-->>RAG: Relevant chunks

    RAG->>LLM: Generate response with context
    LLM-->>RAG: AI response
    RAG-->>ChatSvc: Response + sources
    ChatSvc-->>API: Formatted result
    API->>WS: Send response
    WS->>Chat: Display message
    Chat-->>User: Show answer + sources
```

### Document Deletion Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Admin Panel
    participant API as FastAPI
    participant RAG as RAG System
    participant VDB as ChromaDB
    participant FS as File System

    User->>UI: Click delete document
    UI->>API: DELETE /api/documents/{id}
    API->>RAG: Delete document

    RAG->>RAG: Find document metadata
    RAG->>VDB: Delete from vector store
    VDB-->>RAG: Deletion confirmed

    alt File path exists
        RAG->>FS: Delete physical file
        FS-->>RAG: File deleted
    end

    RAG-->>API: Success response
    API-->>UI: Document deleted
    UI->>UI: Refresh document list
    UI->>UI: Update storage statistics
    UI-->>User: Show updated interface
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- uv package manager (recommended)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd custom-rag
   uv sync
   ```

2. **Configure environment** (optional):
   ```bash
   # Create .env file for API keys
   echo "OPENAI_API_KEY=your_key_here" > .env
   echo "GOOGLE_API_KEY=your_key_here" >> .env
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

4. **Access the interface**:
   - 🏠 **Main Chat**: http://localhost:8000
   - ⚙️ **Admin Panel**: http://localhost:8000/upload
   - 📚 **API Docs**: http://localhost:8000/docs

## 💻 Usage

### 1. Upload Documents
- Navigate to **Admin Panel** (http://localhost:8000/upload)
- Drag & drop files or click to browse
- Configure chunking strategy and embedding model
- Wait for processing to complete

### 2. Chat with Documents
- Go to **Main Chat** (http://localhost:8000)
- Type questions about your uploaded documents
- Get AI-powered responses with source attribution

### 3. Manage System
- Use **Admin Panel** for:
  - Document library management
  - Embedding model configuration
  - LLM provider settings
  - System statistics monitoring

## 🎛️ Configuration Options

### Embedding Models (16 available)

| **Provider** | **Models** | **Cost** | **Privacy** | **Setup** |
|-------------|------------|----------|-------------|-----------|
| **Local (SentenceTransformers)** | 11 models | 🆓 Free | 🔒 Complete | ✅ None |
| **OpenAI** | 3 models | 💰 $0.02-0.13/1M tokens | ⚠️ External | 🔑 API Key |
| **Google** | 2 models | 🆓/💰 Free tier + Paid | ⚠️ External | 🔑 API Key |

**Recommended:**
- **Beginners**: `all-mpnet-base-v2` (default local model)
- **Speed**: `all-MiniLM-L6-v2` (fast local model)
- **Quality**: `text-embedding-3-large` (OpenAI, paid)

### LLM Providers

| **Provider** | **Models** | **Cost** | **Best For** |
|-------------|------------|----------|--------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-3.5 | $0.15-15/1M tokens | Advanced reasoning |
| **Google** | Gemini 1.5 Pro/Flash | Free tier available | Cost-effective |

### Chunking Strategies (9 available)

| **Strategy** | **Best For** | **Description** |
|-------------|--------------|-----------------|
| **Recursive Character** ⭐ | Most documents | Intelligent multi-separator splitting |
| **Token-based** | LLM integration | Precise token count control |
| **Semantic-based** | Research docs | Advanced semantic splitting |
| **Sentence-based** | Q&A systems | Preserves sentence boundaries |

## 📁 Project Structure

```
src/
├── api/              # FastAPI routes and handlers
│   ├── chat.py       # Chat and query endpoints
│   ├── documents.py  # Document management APIs
│   ├── system.py     # System configuration APIs
│   └── upload.py     # File upload handling
├── ui/               # User interface components
│   ├── templates/    # Jinja2 HTML templates
│   ├── static/       # CSS, JS, assets
│   ├── chat_ui.py    # WebSocket chat interface
│   └── upload_ui.py  # Upload interface logic
├── embedding/        # Vector embeddings and storage
│   ├── models.py     # Embedding model factory
│   ├── vector_store.py # ChromaDB wrapper
│   └── chunking.py   # Text chunking strategies
├── llm/              # Language model integrations
│   └── models.py     # LLM provider factory
├── upload/           # Document processing
│   ├── document_processor.py # Text extraction
│   └── file_service.py       # Upload handling
├── config.py         # Configuration management
└── rag_system.py     # Main orchestrator
```

## 🔗 API Endpoints

### Core APIs
- `GET /` - Main chat interface
- `GET /upload` - Admin panel
- `POST /api/upload-documents/` - Upload documents
- `POST /api/query/` - Query documents
- `GET /api/documents/` - List documents
- `DELETE /api/documents/{id}` - Delete document
- `GET /api/system/stats` - System statistics

### WebSocket
- `WS /ws/chat` - Real-time chat communication

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **ChromaDB** - Vector database
- **Sentence-Transformers** - Embedding models
- **LangChain** - Text processing
- **OpenAI/Google APIs** - LLM integration

### Frontend
- **Bootstrap 5** - UI framework
- **Vanilla JavaScript** - Client-side logic
- **WebSocket** - Real-time communication
- **Jinja2** - Template engine

### Storage & Processing
- **SQLite** (ChromaDB) - Vector storage
- **File System** - Document storage
- **AsyncIO** - Concurrent processing

## 🔒 Security Features

- ✅ File type validation
- ✅ Path traversal protection
- ✅ Input sanitization
- ✅ Secure WebSocket connections
- ✅ API key protection
- ✅ Permanent file deletion

## 📈 Performance Tips

1. **Model Selection**: Choose embedding models based on speed vs. accuracy needs
2. **Chunking Strategy**: Adjust chunk size for document types (500-2000 words)
3. **Batch Processing**: Upload multiple documents simultaneously
4. **Cache Management**: Vector database automatically optimizes storage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 **Documentation**: Check the inline API docs at `/docs`
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💡 **Feature Requests**: Use GitHub Discussions
- 📧 **Contact**: Create an issue for support

---

**Built with ❤️ using FastAPI, ChromaDB, and modern AI technologies**