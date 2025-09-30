# Custom RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with multi-agent capabilities, built on LangChain with support for multiple LLM providers and embedding models.

## Features

- **🤖 3 Query Processing Modes**: Vector Search, LLM Response, Agentic RAG
- **📊 5 Embedding Models**: OpenAI and Google embedding models
- **🔧 9 Chunking Strategies**: From simple character-based to semantic chunking
- **🌐 Web Interface**: Admin panel for testing and chat interface for queries
- **🔗 API Gateway Support**: Enterprise-ready with Kong integration
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
# Required: Your API key (for Kong Gateway or direct provider access)
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
- **Embedding Model**: Choose from 5 available models
- **LLM Model**: Select OpenAI (GPT) or Google (Gemini)

### 3. Query Your Documents

**Chat Interface** (http://localhost:8000):
- Select query mode in right sidebar:
  - **🔍 Vector Search**: Fast document retrieval (~100-200ms)
  - **🤖 LLM Response**: Intelligent answers (~1-3 seconds)
  - **🧠 Agentic RAG**: Multi-step analysis (~3-10 seconds)

**Admin Panel** - Test different modes with instant results

## Available Models

### Embedding Models (5)
- **text-embedding-3-small**: OpenAI's fast model
- **text-embedding-3-large**: OpenAI's most capable model
- **text-embedding-ada-002**: OpenAI legacy model
- **models/embedding-001**: Google's general-purpose model
- **models/text-embedding-004**: Google's latest model

### LLM Models
- **OpenAI**: GPT-4, GPT-3.5-turbo, and other GPT models
- **Google**: Gemini Pro, Gemini Flash models

### Chunking Strategies (9)
- **recursive_character**: Best for most documents
- **character**: Simple character-based splitting
- **token_based**: Based on tokenizer limits
- **sentence_transformers_token**: Optimized for embeddings
- **word_based**: Word boundary splitting
- **sentence_based**: Sentence boundary splitting
- **paragraph_based**: Paragraph boundary splitting
- **semantic_based**: Meaning-based chunking
- **fixed_size**: Fixed character size chunks

## API Endpoints

### Query Documents
```bash
# Vector search
curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "mode": "vector_search"}'

# LLM response (default)
curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "mode": "llm_response"}'

# Agentic RAG
curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "mode": "agentic_rag"}'
```

### Get Available Modes
```bash
curl -X GET "http://localhost:8000/api/query/modes"
```

### Upload Documents
```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -F "files=@document.pdf" \
  -F "chunking_strategy=recursive_character"
```

## Configuration

### Environment Variables
- `API_KEY`: Your API key for Kong Gateway or direct provider access (required)
- `BASE_URL`: Optional gateway base URL (models.yaml defines specific paths)
- `DEFAULT_LLM_PROVIDER`: openai or google (defaults from config/models.yaml)
- `DEFAULT_LLM_MODEL`: Model name (defaults from config/models.yaml)
- `DEFAULT_EMBEDDING_MODEL`: Embedding model (defaults from config/models.yaml)

### Configuration Files
- `config/models.yaml`: Centralized model configuration with gateway URLs and Kong headers
- `.env`: Environment variables for API keys and optional overrides

### Kong API Gateway Support
The system supports Kong API Gateway with the correct header format:
- **Headers**: `{"api-key": "your_key", "ai-gateway-version": "v2"}`
- **Configuration**: Automatically handled via `config/models.yaml`
- **Models**: All LLM and embedding models support Kong Gateway routing
- **Fallback**: Direct provider URLs when BASE_URL is not configured

### File Support
- **PDF**: Text extraction with layout preservation
- **DOCX**: Microsoft Word documents
- **XLSX**: Excel spreadsheets (text content)
- **TXT**: Plain text files

## Architecture

### System Design
- **Backend**: FastAPI with async support
- **AI Framework**: LangChain for model integration
- **Vector Database**: ChromaDB for semantic search
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **Real-time**: WebSocket for chat functionality

### Dependency Injection Architecture
The system implements **comprehensive dependency injection** for loose coupling and enhanced testability:

- **Interface-First Design**: Each component implements well-defined interfaces:
  - `src/llm/interfaces/` - LLM Model and Service abstractions
  - `src/embedding/interfaces/` - Embedding Model and Vector Store abstractions
  - `src/upload/interfaces/` - Document Processor abstractions
  - `src/agents/interfaces/` - Agent System abstractions
- **Factory Pattern**: Intelligent creation with configuration-driven defaults
- **Builder Pattern**: Fine-grained control over complex dependency graphs
- **Mock Support**: Complete mock implementations for isolated unit testing
- **Backward Compatibility**: Existing code continues to work unchanged

#### Creating RAG Systems

```python
# Simple creation (backward compatible)
from src.rag_system import create_rag_system
rag_system = create_rag_system(collection_name="docs")

# Factory pattern with dependency injection
from src.rag_factory import RAGSystemFactory
rag_system = RAGSystemFactory.create_default_rag_system(
    embedding_model="text-embedding-3-large"
)

# Builder pattern for complex configurations
from src.rag_factory import RAGSystemBuilder
rag_system = (RAGSystemBuilder()
              .with_config(collection_name="custom")
              .build())

# Custom dependency injection for testing (interfaces-based)
from src.rag_factory import RAGSystemFactory
rag_system = RAGSystemFactory.create_custom_rag_system(
    document_processor=mock_processor,
    vector_store=mock_vector_store,
    llm_service=mock_llm_service
)
```

#### Benefits
- **Easy Testing**: Mock dependencies for isolated unit tests via interface injection
- **Flexible Configuration**: Multiple creation patterns for different deployment scenarios
- **Maintainable Code**: Clear separation of concerns through interface segregation
- **Extensible**: Easy to swap implementations without breaking dependent code
- **Type Safety**: Full interface compliance with abstract base classes

### LLM and Embedding Model Independence

The system ensures **LLM and Embedding models are completely loosely coupled** from the rest of the system through comprehensive interfaces:

#### LLM Model Abstraction
- **Interface**: `ILLMModel` defines contracts for all LLM implementations
- **Factory**: `ILLMModelFactory` handles model creation and validation
- **Providers**: OpenAI, Google (easily extensible)
- **Features**: Response generation, streaming, token estimation, connection validation

#### Embedding Model Abstraction
- **Interface**: `IEmbeddingModel` defines contracts for all embedding implementations
- **Factory**: `IEmbeddingModelFactory` handles model creation and validation
- **Providers**: OpenAI, Google (easily extensible)
- **Features**: Batch encoding, similarity calculation, dimension info, connection validation

#### Independent Testing with Interface Compliance
```bash
# Test LLM models independently with interface validation
python test_llm_models.py --mock-only      # Test with mock implementations
python test_llm_models.py --provider openai # Test OpenAI models against interfaces
python test_llm_models.py --provider google # Test Google models against interfaces
python test_llm_models.py                   # Test all providers and interface compliance

# Test Embedding models independently with interface validation
python test_embedding_models.py --mock-only      # Test with mock implementations
python test_embedding_models.py --provider openai # Test OpenAI embeddings against interfaces
python test_embedding_models.py --provider google # Test Google embeddings against interfaces
python test_embedding_models.py                   # Test all providers and interface compliance
```

#### Interface Compliance and Testing
Both LLM and Embedding models implement comprehensive interfaces ensuring:
- **Loose Coupling**: Models depend only on interfaces, not concrete implementations
- **Interface Compliance**: All implementations conform to abstract base class contracts
- **Mock Testing**: Complete mock implementations for isolated unit testing
- **Swappable Implementations**: Change providers without affecting dependent code
- **Configuration Validation**: Built-in validation for API keys, models, and connections
- **Performance Monitoring**: Token limits, batch sizes, and response time tracking

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

## License

MIT License - see LICENSE file for details.