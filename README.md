# Custom RAG System

A modular Retrieval-Augmented Generation (RAG) system with document upload capabilities and a web-based chat interface.

## Architecture

The system is organized into separate modules within the `src/` folder:

```
src/
├── upload/           # File upload and document processing
│   ├── document_processor.py    # Text extraction from various formats
│   └── file_service.py         # File upload handling
├── embedding/        # Embedding models and vector storage
│   ├── models.py               # Embedding model abstractions
│   └── vector_store.py         # ChromaDB vector database wrapper
├── ui/              # User interface components
│   ├── upload_ui.py            # File upload interface
│   ├── chat_ui.py              # Chat interface with WebSocket
│   ├── templates/              # HTML templates
│   └── static/                 # CSS, JS, and assets
└── rag_system.py    # Main RAG system orchestrator
```

## Features

### Document Upload Module
- **Multi-format support**: PDF, DOCX, XLSX, XLS, TXT
- **Batch processing**: Upload multiple files simultaneously
- **Text extraction**: Intelligent content extraction with metadata
- **Validation**: File type and size validation

### Embedding Models Module
- **Multiple model support**: SentenceTransformers integration
- **Configurable models**: Choose from various embedding models
- **Vector storage**: Persistent ChromaDB integration
- **Similarity search**: Efficient semantic search

### User Interface
- **Upload Interface**: Drag-and-drop file upload with progress tracking
- **Chat Interface**: Real-time chat with WebSocket support
- **Document Management**: View, search, and delete uploaded documents
- **Responsive design**: Works on desktop and mobile

### RAG System
- **Smart chunking**: Configurable text splitting with overlap
- **Vector retrieval**: Semantic similarity search
- **Source attribution**: Track which documents provide answers
- **Real-time processing**: Immediate availability after upload

## Installation

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Access the web interface**:
   - Open your browser to `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`

## Usage

### Web Interface

1. **Upload Documents**: Navigate to `/upload` to upload your documents
2. **Chat**: Use `/chat` to ask questions about your documents
3. **Manage**: Visit `/documents` to view and manage uploaded files

### API Endpoints

- `POST /api/upload-documents/` - Upload multiple documents
- `POST /api/query/` - Query the RAG system
- `GET /api/documents/` - List all documents
- `DELETE /api/documents/{doc_id}` - Delete a document
- `GET /api/stats` - Get system statistics

### Programmatic Usage

```python
from src.rag_system import RAGSystem

# Initialize RAG system (uses all-mpnet-base-v2 by default)
rag = RAGSystem()

# Or specify a different model
rag = RAGSystem(embedding_model="all-MiniLM-L6-v2")

# Add document
doc_id = await rag.add_document(text_content, file_path, filename)

# Query documents
results = await rag.query("What is the main topic?", top_k=5)
```

## Configuration

### Embedding Models

The system supports multiple high-quality embedding models through **SentenceTransformers** (Hugging Face). You can change the global embedding model through the Admin Panel Settings.

#### Model Providers & Sources

The embedding models come from different research organizations and are distributed through **Sentence-Transformers** (Hugging Face). Here are the actual providers for each model:

**Microsoft Research Models:**
- `all-MiniLM-L6-v2` - Microsoft Research
- `all-MiniLM-L12-v2` - Microsoft Research
- `all-mpnet-base-v2` - Microsoft Research
- `all-roberta-large-v1` - Microsoft Research (based on Facebook's RoBERTa)

**Specialized Training Models:**
- `multi-qa-mpnet-base-dot-v1` - Sentence-Transformers team (fine-tuned on Q&A datasets)
- `multi-qa-MiniLM-L6-cos-v1` - Sentence-Transformers team (fine-tuned on Q&A datasets)
- `msmarco-distilbert-base-v4` - Microsoft Research (MS MARCO dataset)

**Multilingual Models:**
- `paraphrase-multilingual-MiniLM-L12-v2` - Sentence-Transformers team
- `paraphrase-multilingual-mpnet-base-v2` - Sentence-Transformers team

**Alternative Models:**
- `all-distilroberta-v1` - Sentence-Transformers team (based on Hugging Face's DistilRoBERTa)

**Distribution Platform:** All models are distributed through **Hugging Face Hub** and accessed via the **Sentence-Transformers** library.

#### Recommended Models

| Model | Provider | Dimensions | Size | Description | Use Case |
|-------|----------|------------|------|-------------|----------|
| **all-mpnet-base-v2** (default) | Microsoft Research | 768 | ~420MB | High quality model with better accuracy | General purpose, best quality |
| **all-MiniLM-L6-v2** | Microsoft Research | 384 | ~80MB | Fast and efficient model | Speed-critical applications |
| **sentence-transformers/all-MiniLM-L12-v2** | Microsoft Research | 384 | ~130MB | Balanced model between speed and accuracy | Balanced performance |

#### Specialized Models

| Model | Provider | Dimensions | Size | Description | Use Case |
|-------|----------|------------|------|-------------|----------|
| **sentence-transformers/multi-qa-mpnet-base-dot-v1** | Sentence-Transformers Team | 768 | ~420MB | Optimized for question-answering tasks | Q&A systems, chatbots |
| **sentence-transformers/multi-qa-MiniLM-L6-cos-v1** | Sentence-Transformers Team | 384 | ~80MB | Fast model optimized for question-answering | Fast Q&A applications |
| **sentence-transformers/msmarco-distilbert-base-v4** | Microsoft Research | 768 | ~250MB | Optimized for passage retrieval and search | Document search, retrieval |

#### Multilingual Models

| Model | Provider | Dimensions | Size | Description | Languages |
|-------|----------|------------|------|-------------|-----------|
| **paraphrase-multilingual-MiniLM-L12-v2** | Sentence-Transformers Team | 384 | ~420MB | Multilingual model supporting 50+ languages | Multi-language documents |
| **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** | Sentence-Transformers Team | 768 | ~970MB | High-quality multilingual model | Premium multi-language support |

#### High Performance Models

| Model | Provider | Dimensions | Size | Description | Use Case |
|-------|----------|------------|------|-------------|----------|
| **sentence-transformers/all-roberta-large-v1** | Microsoft Research | 1024 | ~1.3GB | Large high-performance model (slower but very accurate) | Maximum accuracy needs |

#### Alternative Models

| Model | Provider | Dimensions | Size | Description | Use Case |
|-------|----------|------------|------|-------------|----------|
| **all-distilroberta-v1** | Sentence-Transformers Team | 768 | ~290MB | Distilled RoBERTa model with good performance | Alternative to MPNet |

#### Model Selection Guidelines

- **For most users**: Use `all-mpnet-base-v2` (default) for best quality
- **For speed**: Use `all-MiniLM-L6-v2` for fastest processing
- **For multilingual content**: Use `paraphrase-multilingual-MiniLM-L12-v2`
- **For Q&A systems**: Use `multi-qa-mpnet-base-dot-v1`
- **For maximum accuracy**: Use `all-roberta-large-v1` (if you have sufficient resources)

#### Changing Embedding Models

⚠️ **Important**: Changing the embedding model requires handling existing documents:

1. **Settings Access**: Go to Admin Panel → Settings → Global Embedding Model
2. **Model Selection**: Choose from the dropdown list
3. **Document Handling**:
   - **Change model only**: Existing documents become incompatible and need re-upload
   - **Change model and clear documents**: All existing documents are removed

#### Technical Notes

- All models use **cosine similarity** for document matching
- Higher dimensions generally mean better accuracy but slower processing
- Model size affects download time on first use
- All models are cached locally after first download

### Chunking Settings

```python
rag_system = RAGSystem(
    chunk_size=1000,      # Words per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## Development

### Adding New File Types

1. Extend `DocumentProcessor` in `src/upload/document_processor.py`
2. Add extraction method for new format
3. Update `supported_formats` dictionary

### Custom Embedding Models

1. Implement `EmbeddingModel` interface in `src/embedding/models.py`
2. Add to `EmbeddingModelFactory`

### UI Customization

- Templates: `src/ui/templates/`
- Styles: `src/ui/static/css/style.css`
- Scripts: `src/ui/static/js/`

## Supported File Formats

| Format | Description | Features |
|--------|-------------|----------|
| **PDF** | Portable Document Format | Multi-page text extraction |
| **DOCX** | Microsoft Word | Text + tables extraction |
| **XLSX/XLS** | Microsoft Excel | All sheets, preserves structure |
| **TXT** | Plain text | UTF-8 and Latin-1 encoding support |

## Technical Details & Python Frameworks

### Core Framework Stack

**Web Framework & API:**
- **FastAPI** - Modern, fast web framework for building APIs with automatic documentation
- **Uvicorn** - ASGI server for running FastAPI applications
- **Pydantic** - Data validation and settings management using Python type annotations

**Embedding & AI Frameworks:**
- **Sentence-Transformers** - Framework for state-of-the-art sentence embeddings (Hugging Face)
- **LangChain** - Framework for text splitting and chunking strategies
  - `langchain-text-splitters` - Advanced text splitting algorithms
  - `langchain-community` - Community-contributed components
- **tiktoken** - OpenAI's tokenizer for precise token counting

**Vector Database & Storage:**
- **ChromaDB** - Open-source embedding database for vector storage and similarity search
- **NumPy** - Numerical computing for embedding operations
- **Pandas** - Data manipulation and analysis

**Document Processing:**
- **PyPDF2** - PDF document text extraction
- **python-docx** - Microsoft Word document processing
- **openpyxl** - Excel file reading and processing
- **python-magic** - File type detection

**Chat & Real-time Communication:**
- **WebSocket** (FastAPI built-in) - Real-time bidirectional communication for chat
- **AsyncIO** - Asynchronous programming for handling multiple chat connections

**UI & Templates:**
- **Jinja2** - Template engine for HTML rendering
- **Bootstrap 5** - Frontend CSS framework
- **Vanilla JavaScript** - Client-side interactivity and WebSocket handling

**Security & Authentication:**
- **python-jose** - JSON Web Token (JWT) handling
- **passlib** - Password hashing and verification
- **python-multipart** - File upload handling

**Additional Utilities:**
- **aiofiles** - Asynchronous file operations
- **python-magic** - MIME type detection

### Embedding Architecture

**Primary Framework:** **Sentence-Transformers** (Hugging Face)
- **Purpose**: Converts text documents into numerical vector representations
- **Models**: Supports 11 different embedding models from Microsoft Research and Sentence-Transformers team
- **Backend**: Uses PyTorch for neural network computations
- **Storage**: Vectors stored in ChromaDB for fast similarity search

**Text Processing Pipeline:**
1. **LangChain Text Splitters** - Intelligent document chunking
2. **Sentence-Transformers** - Convert text chunks to embeddings
3. **ChromaDB** - Store and index vector embeddings
4. **Cosine Similarity** - Calculate semantic similarity for search

### Chat Architecture

**Framework:** **FastAPI WebSockets** with **AsyncIO**
- **Real-time Communication**: WebSocket connections for instant messaging
- **Concurrency**: AsyncIO handles multiple simultaneous chat sessions
- **RAG Integration**: Chat queries processed through vector similarity search
- **Response Generation**: Combines retrieved document chunks with user queries

**Chat Flow:**
1. **WebSocket Connection** - Establishes real-time communication channel
2. **Query Processing** - User messages sent through WebSocket
3. **Vector Search** - Sentence-Transformers finds relevant document chunks
4. **Response Assembly** - Combines search results into coherent answers
5. **Real-time Delivery** - Responses streamed back via WebSocket

**No External LLM Required**: The system uses retrieval-augmented generation without external APIs like OpenAI or Claude.

## Performance Tips

1. **Choose appropriate embedding model** for your use case
2. **Adjust chunk size** based on document types
3. **Use document filtering** for targeted queries
4. **Monitor vector database size** for large deployments

## Security

- File type validation on upload
- Path traversal protection
- Input sanitization in UI
- Secure WebSocket connections