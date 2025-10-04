# Custom RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with multi-agent capabilities, built on LangChain with support for multiple LLM providers and embedding models.

## Features

- **🤖 Multiple Query Processing Modes**: Vector Search, LLM Response, Agentic RAG
- **📊 Multiple Embedding Models**: OpenAI and Google embedding models
- **🔧 Multiple Chunking Strategies**: From simple character-based to semantic chunking
- **🌐 Web Interface**: Admin panel for testing and chat interface for queries
- **🔗 API Gateway Support**: Enterprise-ready with configurable API gateway integration
- **⚡ Agentic RAG System**: Advanced reasoning with specialized tools

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI or Google API key

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd custom-rag
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
```

Edit `.env` file:
```bash
# Required: Your API key (for API Gateway or direct provider access)
API_KEY=your_api_key_here

# Optional: Gateway Base URL (models.yaml defines specific gateway paths)
BASE_URL=https://api.your-gateway.com

# Optional: Default models (can override YAML config defaults)
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

### Run the Application

```bash
python main.py
```

Open your browser to:
- **Chat Interface**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/upload

## Usage

### 1. Upload Documents
1. Go to Admin Panel (http://localhost:8000/upload)
2. Upload PDF, DOCX, XLSX, or TXT files
3. Select chunking strategy (recommended: "Recursive Character")
4. Click "Upload Files"

### 2. Configure Models
In Admin Panel:
- **Embedding Model**: Choose from available OpenAI and Google models
- **LLM Model**: Select OpenAI (GPT) or Google (Gemini)

### 3. Query Your Documents

**Chat Interface** (http://localhost:8000):
- Select query mode in right sidebar:
  - **🔍 Vector Search**: Fast document retrieval (~100-200ms)
  - **🤖 LLM Response**: Intelligent answers (~1-3 seconds)
  - **🧠 Agentic RAG**: Multi-step analysis (~3-10 seconds)

**Admin Panel** - Test different modes with instant results

## Available Models

### Embedding Models
- **OpenAI Models**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Google Models**: models/embedding-001, models/text-embedding-004

### LLM Models
- **OpenAI**: GPT-4, GPT-3.5-turbo, and other GPT models
- **Google**: Gemini Pro, Gemini Flash models

### Chunking Strategies
- **recursive_character**: Best for most documents
- **character**: Simple character-based splitting
- **token_based**: Based on tokenizer limits
- **sentence_transformers_token**: Optimized for embeddings
- **word_based**: Word boundary splitting
- **sentence_based**: Sentence boundary splitting
- **paragraph_based**: Paragraph boundary splitting
- **semantic_based**: Meaning-based chunking
- **fixed_size**: Fixed character size chunks

## Configuration

### Environment Variables
- `API_KEY`: Your API key for API Gateway or direct provider access (required)
- `BASE_URL`: Optional gateway base URL (models.yaml defines specific paths)
- `DEFAULT_LLM_PROVIDER`: openai or google (defaults from config/models.yaml)
- `DEFAULT_LLM_MODEL`: Model name (defaults from config/models.yaml)
- `DEFAULT_EMBEDDING_MODEL`: Embedding model (defaults from config/models.yaml)

### Configuration Files
- `config/models.yaml`: Centralized model configuration with gateway URLs and custom headers
- `.env`: Environment variables for API keys and optional overrides

### API Gateway Support
The system supports API Gateway integration with configurable header format:
- **Headers**: Customizable via `config/models.yaml` (e.g., `{"api-key": "your_key", "ai-gateway-version": "v2"}`)
- **Configuration**: Automatically handled via `config/models.yaml`
- **Models**: All LLM and embedding models support API Gateway routing
- **Fallback**: Direct provider URLs when BASE_URL is not configured

### File Support
- **PDF**: Text extraction with layout preservation
- **DOCX**: Microsoft Word documents
- **XLSX**: Excel spreadsheets (text content)
- **TXT**: Plain text files

## Architecture

### System Design

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Chat Interface<br/>WebSocket]
        B[Admin Panel<br/>Upload & Config]
    end

    subgraph "API Layer"
        C[FastAPI Backend<br/>Async REST API]
    end

    subgraph "Processing Layer"
        D[Query Router]
        E[Vector Search Mode]
        F[LLM Response Mode]
        G[Agentic RAG Mode]
    end

    subgraph "RAG Core"
        H[Document Processor<br/>PDF, DOCX, XLSX, TXT]
        I[Chunking Engine<br/>9 Strategies]
        J[RAG Orchestrator]
    end

    subgraph "Storage & Retrieval"
        K[(ChromaDB<br/>Vector Store)]
        L[Embedding Models<br/>OpenAI, Google]
    end

    subgraph "Intelligence Layer"
        M[LLM Service<br/>LangChain]
        N[OpenAI GPT-4<br/>GPT-3.5]
        O[Google Gemini<br/>Pro, Flash]
        P[Agent Tools<br/>Knowledge Search<br/>Document Analysis]
    end

    subgraph "API Gateway"
        Q[API Gateway]
    end

    A --> C
    B --> C
    C --> D
    D --> E & F & G
    E --> J
    F --> J
    G --> P
    P --> J
    B --> H
    H --> I
    I --> K
    J --> K
    J --> M
    K <--> L
    M --> N & O
    N & O <--> Q
    L <--> Q

    style A fill:#4FC3F7,stroke:#0288D1,stroke-width:2px,color:#000
    style B fill:#4FC3F7,stroke:#0288D1,stroke-width:2px,color:#000
    style C fill:#FFB74D,stroke:#F57C00,stroke-width:2px,color:#000
    style D fill:#BA68C8,stroke:#7B1FA2,stroke-width:2px,color:#000
    style E fill:#BA68C8,stroke:#7B1FA2,stroke-width:2px,color:#000
    style F fill:#BA68C8,stroke:#7B1FA2,stroke-width:2px,color:#000
    style G fill:#BA68C8,stroke:#7B1FA2,stroke-width:2px,color:#000
    style H fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#000
    style I fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#000
    style J fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#000
    style K fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000
    style L fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000
    style M fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style N fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style O fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style P fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style Q fill:#BDBDBD,stroke:#616161,stroke-width:2px,color:#000
```

**Technology Stack:**
- **Backend**: FastAPI with async support
- **AI Framework**: LangChain for model integration
- **Vector Database**: ChromaDB for semantic search
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **Real-time**: WebSocket for chat functionality

### Design Principles
- **Interface-First Design**: Modular components with well-defined interfaces
- **Dependency Injection**: Loose coupling for enhanced testability
- **Factory Pattern**: Configuration-driven component creation
- **Mock Support**: Complete mock implementations for isolated testing
- **Provider Agnostic**: Easy to swap LLM and embedding providers

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `uv sync`

2. **API Key Errors**: Check your `.env` file has the correct `API_KEY`

3. **Model Not Found**: Verify the model name in your configuration

4. **Upload Failures**: Check file format is supported (PDF, DOCX, XLSX, TXT)

5. **Slow Responses**: Agentic RAG mode takes longer (~3-10 seconds)

### Getting Help

- Check the Admin Panel for system status
- Review logs in the terminal output
- Verify API key and model configuration
- Ensure documents are uploaded successfully before querying