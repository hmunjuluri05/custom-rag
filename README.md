# Custom RAG System

A modular Retrieval-Augmented Generation (RAG) system with document upload capabilities and a web-based chat interface.

## System Workflow

The RAG system operates through a clear two-phase workflow:

### 📥 **Document Ingestion Workflow**
```
Documents → Embedding Model → Vector Database
```

1. **Document Upload**: PDF, DOCX, XLSX, TXT files uploaded via web interface
2. **Text Extraction**: Content extracted and processed from various file formats
3. **Text Chunking**: Documents split into manageable chunks using configurable strategies
4. **Embedding Generation**: **Embedding Model** converts each chunk into vector embeddings
5. **Vector Storage**: Embeddings stored in ChromaDB for fast similarity search

### 💬 **Query & Response Workflow**
```
User Question → Embedding Model → Vector Search → LLM Model → Response
```

1. **User Query**: Question submitted through chat interface
2. **Query Embedding**: **Embedding Model** converts question into vector representation
3. **Similarity Search**: System finds most relevant document chunks using vector similarity
4. **Context Assembly**: Retrieved chunks combined with user question
5. **Response Generation**: **LLM Model** generates intelligent, contextual response
6. **Answer Delivery**: Final answer delivered to user with source attribution

### 🔄 **Model Roles**

| Model Type | Usage Phase | Purpose | Can Change Without Reprocessing? |
|------------|-------------|---------|-----------------------------------|
| **Embedding Model** | Document Ingestion + Query | Convert text to vectors for search | ❌ No - requires reprocessing all documents |
| **LLM Model** | Response Generation Only | Generate intelligent responses | ✅ Yes - only affects future responses |

### 🎯 **Key Benefits of This Architecture**

- **Separation of Concerns**: Search and response generation are independent
- **Flexible LLM Selection**: Change response generation models without data migration
- **Efficient Search**: Embedding-based similarity search finds relevant content quickly
- **Source Attribution**: Responses linked back to original documents
- **Scalable**: Add documents incrementally without reprocessing existing ones

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

The system supports **16 different embedding models** from **3 providers**: local SentenceTransformers models, OpenAI API models, and Google API models. You can change the global embedding model through the Admin Panel Settings.

#### Understanding Embedding Models

**What are embeddings?**
Embeddings convert your documents into numerical representations (vectors) that capture semantic meaning. This allows the system to find relevant documents even when the exact words don't match.

**Why does the choice matter?**
- **Better models** = more accurate document retrieval
- **Faster models** = quicker search responses
- **Different models** excel at different tasks (general vs Q&A vs multilingual)

#### Provider Comparison

| Provider | Models | Cost | Privacy | Setup Required |
|----------|--------|------|---------|----------------|
| **Local (SentenceTransformers)** | 11 models | 🆓 Free | 🔒 Complete privacy | ✅ None |
| **OpenAI** | 3 models | 💰 Paid API | ⚠️ Data sent to OpenAI | 🔑 API Key |
| **Google** | 2 models | 🆓/💰 Free tier + Paid | ⚠️ Data sent to Google | 🔑 API Key |

#### 🌐 **External API Models** (Requires API Key)

##### **OpenAI Embedding Models**

| Model | Dimensions | Cost | Description | Best For |
|-------|------------|------|-------------|----------|
| **text-embedding-3-large** | 3072 | $0.13/1M tokens | 🏆 OpenAI's most capable model | Maximum accuracy, large documents |
| **text-embedding-3-small** | 1536 | $0.02/1M tokens | ⚡ Fast and efficient | Cost-effective, good performance |
| **text-embedding-ada-002** | 1536 | $0.10/1M tokens | 📊 Previous generation (legacy) | Legacy applications |

**Setup Required:** OpenAI API key from [platform.openai.com](https://platform.openai.com)

##### **Google Embedding Models**

| Model | Dimensions | Cost | Description | Best For |
|-------|------------|------|-------------|----------|
| **models/embedding-001** | 768 | Free quota + paid | 🌍 General-purpose embedding | Balanced performance |
| **models/text-embedding-004** | 768 | Free quota + paid | 🆕 Latest Google model | Latest features, improved accuracy |

**Setup Required:** Google AI API key from [ai.google.dev](https://ai.google.dev)

#### 💻 **Local Models** (Free, Private)

These models run entirely on your computer - no API keys needed, complete privacy.

##### **Recommended Local Models**

| Model | Dimensions | Size | Description | Best For |
|-------|------------|------|-------------|----------|
| **all-mpnet-base-v2** ⭐ | 768 | ~420MB | 🎯 Default - high quality | Most users, best accuracy |
| **all-MiniLM-L6-v2** | 384 | ~80MB | ⚡ Fastest processing | Speed-critical applications |
| **all-MiniLM-L12-v2** | 384 | ~130MB | ⚖️ Balanced speed/accuracy | Good compromise |

##### **Specialized Local Models**

| Model | Dimensions | Size | Description | Best For |
|-------|------------|------|-------------|----------|
| **multi-qa-mpnet-base-dot-v1** | 768 | ~420MB | ❓ Q&A optimized | Question-answering systems |
| **multi-qa-MiniLM-L6-cos-v1** | 384 | ~80MB | ❓ Fast Q&A | Quick Q&A applications |
| **msmarco-distilbert-base-v4** | 768 | ~250MB | 🔍 Search optimized | Document search, retrieval |

##### **Multilingual Local Models**

| Model | Dimensions | Size | Description | Languages |
|-------|------------|------|-------------|-----------|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | ~420MB | 🌍 50+ languages | Multi-language docs |
| **paraphrase-multilingual-mpnet-base-v2** | 768 | ~970MB | 🌍 Premium multilingual | High-quality multi-language |

##### **High Performance Local Models**

| Model | Dimensions | Size | Description | Best For |
|-------|------------|------|-------------|----------|
| **all-roberta-large-v1** | 1024 | ~1.3GB | 🚀 Maximum accuracy (slower) | When accuracy is critical |
| **all-distilroberta-v1** | 768 | ~290MB | 🔄 Alternative to MPNet | Different architecture option |

#### 🎯 **Choosing the Right Embedding Model**

**For beginners:**
- Start with **all-mpnet-base-v2** (default) - works great for most use cases

**For specific needs:**
- **Maximum accuracy**: OpenAI `text-embedding-3-large` (paid) or local `all-roberta-large-v1`
- **Speed priority**: Local `all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`
- **Cost-effective**: Local models (free) or Google models (free tier)
- **Privacy critical**: Any local model
- **Multilingual**: Local `paraphrase-multilingual-*` models
- **Q&A systems**: Local `multi-qa-*` models or any external model

**Quality vs Speed vs Cost:**
```
🏆 Quality:    External (OpenAI/Google) > Large Local > Standard Local
⚡ Speed:      Small Local > Standard Local > External APIs
💰 Cost:      Local (Free) > Google (Free tier) > OpenAI (Paid)
🔒 Privacy:   Local (Complete) > External (Data sent to provider)
```

### Large Language Models (LLMs)

The system supports **intelligent response generation** using Large Language Models. You can choose between simple context display or AI-powered responses.

#### 🤖 **Response Generation Options**

| Option | Description | Cost | Privacy | Quality |
|--------|-------------|------|---------|---------|
| **OpenAI (GPT)** | AI writes intelligent responses | 💰 Paid API | ⚠️ Data sent to OpenAI | 🎯 Excellent |
| **Google (Gemini)** | AI writes intelligent responses | 🆓/💰 Free tier + Paid | ⚠️ Data sent to Google | 🎯 Excellent |

#### 📋 **Response Examples**

**Question:** "What is our company's vacation policy?"

**LLM-Generated Response:**
```
Based on your Employee Handbook, here's your vacation policy:

**Entitlement:** You get 15 paid vacation days per year

**How to request:** Submit requests at least 2 weeks in advance

**Additional notes:** The policy mentions unused days don't roll over, so use them before year-end.

*Source: Employee_Handbook.pdf, pages 12-13*
```

#### 🌐 **External LLM Models**

##### **OpenAI Models**

| Model | Cost | Description | Best For |
|-------|------|-------------|----------|
| **gpt-4o** | $5.00/1M tokens in, $15.00/1M out | 🏆 Most capable model | Complex analysis, reasoning |
| **gpt-4o-mini** | $0.15/1M tokens in, $0.60/1M out | ⚡ Fast and cost-effective | Most use cases |
| **gpt-3.5-turbo** | $0.50/1M tokens in, $1.50/1M out | 📊 Reliable workhorse | General Q&A, summaries |

##### **Google Models**

| Model | Cost | Description | Best For |
|-------|------|-------------|----------|
| **gemini-1.5-flash** | Free tier available | ⚡ Fast responses | Quick answers |
| **gemini-1.5-pro** | Free tier available | 🏆 Advanced reasoning | Complex documents |
| **gemini-pro** | Free tier available | 📊 Balanced performance | General use |

#### 🎯 **Choosing the Right LLM**

**For beginners:**
- Start with **Google Gemini** (free tier) for intelligent responses without cost
- Upgrade to **OpenAI GPT-4o-mini** for more advanced capabilities

**For specific needs:**
- **Best quality responses**: OpenAI GPT-4o or Google Gemini Pro
- **Cost-effective**: Google models (free tier) or OpenAI GPT-4o-mini
- **Fast responses**: Google Gemini Flash or OpenAI GPT-4o-mini
- **Complex documents**: OpenAI GPT-4o or Google Gemini Pro

**Why use LLM models:**
- **Intelligent responses**: Get explanations, summaries, and contextual answers
- **Source attribution**: Responses include references to source documents
- **Better user experience**: Natural language responses instead of raw text chunks

### Document Chunking Strategies

The system offers **9 different chunking strategies** to split your documents optimally for search and retrieval.

#### 🧩 **Understanding Chunking**

**What is chunking?**
Large documents are split into smaller, overlapping pieces (chunks) that fit within the embedding model's limits. Good chunking improves search accuracy.

**Why does chunking matter?**
- **Better chunks** = more relevant search results
- **Right size** = captures complete thoughts without cutting them off
- **Proper overlap** = ensures important information isn't lost at boundaries

#### 📊 **Available Chunking Strategies**

##### **🏆 Recommended (LangChain-based)**

| Strategy | Description | Best For | Parameters |
|----------|-------------|----------|------------|
| **Recursive Character** ⭐ | Intelligently splits by multiple separators | Most documents, production use | chunk_size, overlap |
| **Token-based (GPT)** | Splits by token count (GPT-compatible) | LLM integration, API limits | chunk_size, overlap, model |
| **Character-based** | Simple character splitting | Basic needs | chunk_size, overlap |

##### **📝 Custom Strategies**

| Strategy | Description | Best For | Parameters |
|----------|-------------|----------|------------|
| **Word-based** | Splits by word count with sentence preservation | General purpose, balanced | chunk_size, overlap, preserve_sentences |
| **Sentence-based** | Splits at sentence boundaries | Preserving context, Q&A | chunk_size, overlap |
| **Paragraph-based** | Splits at paragraph boundaries | Structured documents | chunk_size, overlap |
| **Semantic-based** | Advanced semantic similarity splitting | Research, complex analysis | chunk_size, overlap |
| **Fixed Character Size** | Fixed character-length chunks | Consistent sizing needs | chunk_size, overlap |
| **SentenceTransformers Token** | Optimized for embedding models | Embedding optimization | tokens_per_chunk, overlap |

#### ⚙️ **Configuration Options**

**Chunk Size:**
- **Small (500-800)**: Better for precise answers, more chunks
- **Medium (1000-1500)**: Balanced approach (default: 1000)
- **Large (2000-3000)**: Better context, fewer chunks

**Chunk Overlap:**
- **Low (100-150)**: Less redundancy, faster processing
- **Medium (200-300)**: Balanced approach (default: 200)
- **High (400-500)**: Maximum context preservation

**Additional Options:**
- **Preserve Sentences**: Don't split mid-sentence (recommended)
- **Preserve Paragraphs**: Keep paragraph structure intact

#### 🎯 **Choosing the Right Strategy**

**For beginners:**
- Use **Recursive Character** (default) - works great for most documents

**For specific document types:**
- **Technical docs**: Word-based with sentence preservation
- **Legal documents**: Paragraph-based to preserve structure
- **Q&A content**: Sentence-based for clean answers
- **Research papers**: Semantic-based for intelligent splitting
- **API integration**: Token-based for precise token control

**Configuration in UI:**
1. Go to **Admin Panel → Upload Documents**
2. Expand **Chunking Configuration** section
3. Choose strategy and adjust parameters
4. Settings apply to newly uploaded documents

#### 🔄 **Changing Models & Settings**

**⚠️ Important Considerations:**

**Changing Embedding Models:**
- Existing documents become incompatible
- Options: Re-upload documents or clear collection
- Access: Admin Panel → Settings → Global Embedding Model

**Changing LLM Providers:**
- No impact on existing documents
- Only affects response generation
- Access: Admin Panel → Settings → LLM Provider

**Changing Chunking:**
- Only affects newly uploaded documents
- Existing documents keep their original chunking
- To re-chunk: Delete and re-upload documents

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

**Intelligent Response Generation**: The system combines document retrieval with LLM processing for natural, contextual answers.

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