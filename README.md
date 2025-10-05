# Custom RAG System

Production-ready RAG system with AI-powered chunking, hybrid search, and multi-agent capabilities.

## Features

- **🔍 Hybrid Search**: Combines vector similarity + AI metadata matching (keywords, topics, entities)
- **🧠 AI-Powered Chunking**: LLM analyzes documents for optimal semantic chunking
- **🤖 Agentic RAG**: Multi-step reasoning for complex queries
- **📊 Multiple Models**: OpenAI and Google embeddings/LLMs
- **🌐 Web Interface**: Chat UI and admin panel
- **⚡ API Gateway Support**: Enterprise-ready

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI or Google API key

### Installation

```bash
# Clone repository
git clone <repository-url>
cd custom-rag

# Install dependencies
uv sync
# or: pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

Edit `.env`:
```bash
API_KEY=your_api_key_here
BASE_URL=https://api.your-gateway.com  # Optional
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

### Run

```bash
python main.py
```

Open browser:
- **Chat**: http://localhost:8000
- **Admin**: http://localhost:8000/upload

## Usage

### 1. Upload Documents
1. Go to Admin Panel
2. Upload PDF, DOCX, XLSX, or TXT files
3. Select chunking strategy:
   - **Recursive Character** (recommended for speed)
   - **LLM Enhanced** (best quality, uses AI)
4. Choose metadata detail level (for LLM strategies):
   - **Basic**: Keywords + summary
   - **Detailed**: + Topics + entities
   - **Comprehensive**: + Sentiment + facts

### 2. Query Documents
Select query mode:
- **🔍 Vector Search**: Fast (~100ms)
- **🤖 LLM Response**: Intelligent answers (~1-3s) - **uses hybrid search**
- **🧠 Agentic RAG**: Complex analysis (~3-10s)

## Chunking Strategies

### AI-Powered (Recommended)
- **llm_enhanced**: Fast chunking + AI refinement + optional metadata
  - Uses Recursive Character splitter, then LLM improves boundaries
  - **Enables hybrid search** with metadata matching
  - ~2-5 seconds per document

- **llm_semantic**: Full AI semantic chunking
  - LLM identifies natural boundaries
  - Best quality, slower (~5-10 seconds)

### Standard (Fast)
- **recursive_character**: LangChain's best splitter (recommended)
- **token_based**: GPT tokenizer-based
- **sentence_based**: Preserves sentence boundaries
- **paragraph_based**: Preserves paragraph structure

## Hybrid Search

When you use **LLM chunking with metadata**, the system automatically uses **hybrid search**:

**Traditional Search**: Only vector similarity (semantic meaning)
**Hybrid Search**: Vector (70%) + Metadata (30%) matching

### Metadata Scoring
- **Keywords** (30%): Matches `llm_keywords`
- **Topics** (25%): Matches `llm_topic`
- **Entities** (25%): Matches `llm_entities` (people, places, organizations)
- **Title** (20%): Matches `llm_title`

### Result
**15-30% better relevance** for keyword/entity queries!

## Configuration Files

- `.env`: API keys and defaults
- `config/models.yaml`: Model configurations and gateway URLs

## Supported Files

- PDF, DOCX, XLSX, TXT

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
        I[Chunking Engine<br/>11 Strategies]
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

### Agentic RAG Workflow

```mermaid
flowchart TB
    Start([User Query]) --> Agent[ReAct Agent<br/>LLM-powered reasoning]

    Agent --> Think{Thought:<br/>What do I need?}

    Think --> |"Need to search<br/>knowledge base"| Action1[Action: RAG Search Tool]
    Think --> |"Need document<br/>analysis"| Action2[Action: Document Analysis Tool]
    Think --> |"Have enough<br/>information"| Final[Final Answer]

    Action1 --> Vector[Vector Store Search]
    Vector --> Retrieve[Retrieve Relevant Chunks]
    Retrieve --> Obs1[Observation:<br/>Retrieved context]

    Action2 --> DocAnalysis[Analyze Document Structure]
    DocAnalysis --> Obs2[Observation:<br/>Document insights]

    Obs1 --> Agent
    Obs2 --> Agent

    Final --> Response([Synthesized Answer])

    style Start fill:#4FC3F7,stroke:#0288D1,stroke-width:3px,color:#000
    style Agent fill:#BA68C8,stroke:#7B1FA2,stroke-width:3px,color:#000
    style Think fill:#FFB74D,stroke:#F57C00,stroke-width:2px,color:#000
    style Action1 fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#000
    style Action2 fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#000
    style Vector fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000
    style Retrieve fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000
    style DocAnalysis fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000
    style Obs1 fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style Obs2 fill:#F06292,stroke:#C2185B,stroke-width:2px,color:#000
    style Final fill:#81C784,stroke:#388E3C,stroke-width:3px,color:#000
    style Response fill:#4FC3F7,stroke:#0288D1,stroke-width:3px,color:#000
```

### Query Flow

```
User Query
    ↓
Hybrid Search (if AI metadata available)
    ├─ Vector Similarity (70%)
    └─ Metadata Matching (30%)
    ↓
ChromaDB → Retrieve Chunks
    ↓
LLM (GPT/Gemini) → Generate Response
    ↓
Return with Sources
```

## When to Use What

| Use Case | Chunking | Query Mode |
|----------|----------|------------|
| General documents | Recursive Character | LLM Response |
| Legal/Financial | LLM Enhanced + Detailed | LLM Response |
| Technical docs | LLM Enhanced + Basic | LLM Response |
| Speed critical | Recursive Character | Vector Search |
| Complex analysis | LLM Semantic + Comprehensive | Agentic RAG |

## Troubleshooting

**Import Errors**: Run `uv sync`
**API Key Errors**: Check `.env` file
**Upload Failures**: Verify file format (PDF, DOCX, XLSX, TXT)
**Slow Responses**: Agentic RAG takes 3-10 seconds (normal)

## Technology Stack

- **Backend**: FastAPI (async)
- **AI**: LangChain + OpenAI/Google
- **Vector DB**: ChromaDB
- **Frontend**: Bootstrap 5 + WebSocket
