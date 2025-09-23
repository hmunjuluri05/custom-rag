"""
Demo Models for UI Testing
These models simulate LLM and embedding functionality without external dependencies.
"""

import os
import time
import random
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Demo responses for different types of queries
DEMO_RESPONSES = {
    "greeting": [
        "Hello! I'm a demo AI assistant. I can help you explore the features of this RAG system.",
        "Hi there! This is demo mode - all responses are simulated. Feel free to test the interface!",
        "Welcome to the demo! I can answer questions about the uploaded documents (simulated)."
    ],
    "summarize": [
        """📄 **Document Summary (Demo)**

This is a simulated summary of your uploaded documents. In the real system, I would analyze the actual content and provide:

• Key findings and main points
• Important data and statistics
• Relevant conclusions and recommendations
• Cross-references between documents

The demo shows how summaries would appear with proper formatting and source citations.""",
        """📊 **Analysis Summary (Demo)**

Based on the uploaded documents (simulated), here are the key insights:

1. **Primary Topics**: The documents cover various subjects relevant to your query
2. **Key Data Points**: Important metrics and figures are highlighted
3. **Conclusions**: Main takeaways and recommendations
4. **Next Steps**: Suggested actions based on the content

This demonstrates the summary capabilities of the full system."""
    ],
    "key_findings": [
        """🔍 **Key Findings (Demo)**

Here are the simulated key findings from your documents:

**Finding 1**: Important discovery or insight from Document A
- Supporting evidence and context
- Relevant data points

**Finding 2**: Critical observation from Document B
- Additional details and implications
- Cross-references to other sections

**Finding 3**: Significant conclusion from Document C
- Supporting analysis and recommendations
- Related findings and patterns

These findings demonstrate how the system extracts and presents key insights.""",
        """💡 **Key Insights (Demo)**

**Primary Insights:**
• Strategic recommendations based on document analysis
• Important trends and patterns identified
• Critical data points and their implications
• Actionable items and next steps

**Secondary Observations:**
• Supporting evidence from multiple sources
• Comparative analysis between documents
• Risk factors and considerations
• Opportunities for improvement

This shows how key findings would be structured and presented."""
    ],
    "general": [
        "This is a demo response showing how the AI would answer your question based on uploaded documents. In the real system, this would be a detailed analysis of your specific query.",
        "I'm running in demo mode, so this is a simulated response. The actual system would search through your documents and provide relevant, accurate information based on the content.",
        "Demo response: The AI system would analyze your uploaded documents and provide a comprehensive answer with source citations and relevant details.",
        "In demo mode, I can show you how responses would be formatted and presented. The real system would provide actual insights from your document content."
    ],
    "technical": [
        """🔧 **Technical Analysis (Demo)**

Based on the simulated document analysis:

**System Requirements:**
- Processing capabilities: Advanced
- Integration options: Multiple APIs supported
- Scalability: Horizontal scaling available

**Implementation Details:**
- Framework: Modern architecture
- Security: Enterprise-grade
- Performance: Optimized for speed

**Recommendations:**
- Follow best practices for deployment
- Consider load balancing for high traffic
- Implement proper monitoring and logging

This demonstrates technical documentation analysis capabilities.""",
        """⚙️ **Technical Specifications (Demo)**

**Architecture Overview:**
- Microservices-based design
- RESTful API integration
- Real-time processing capabilities

**Performance Metrics:**
- Response time: <2 seconds average
- Throughput: 1000+ requests/minute
- Availability: 99.9% uptime

**Security Features:**
- End-to-end encryption
- Role-based access control
- Audit logging and monitoring

This shows how technical content would be analyzed and presented."""
    ]
}

DEMO_SOURCES = [
    {
        "document_id": "demo_doc_1",
        "filename": "Strategic_Planning_2024.pdf",
        "relevance_score": 0.92,
        "chunk_count": 3
    },
    {
        "document_id": "demo_doc_2",
        "filename": "Market_Research_Report.docx",
        "relevance_score": 0.87,
        "chunk_count": 2
    },
    {
        "document_id": "demo_doc_3",
        "filename": "Technical_Specifications.pdf",
        "relevance_score": 0.81,
        "chunk_count": 1
    },
    {
        "document_id": "demo_doc_4",
        "filename": "Financial_Analysis_Q3.xlsx",
        "relevance_score": 0.76,
        "chunk_count": 2
    }
]

class DemoLLMModel:
    """Demo LLM model that simulates OpenAI/Google responses"""

    def __init__(self, model_name: str = "gpt-4", **kwargs):
        self.model_name = model_name
        self.response_delay = float(os.getenv('DEMO_RESPONSE_DELAY', 1))
        self.enable_sources = os.getenv('DEMO_ENABLE_SOURCES', 'true').lower() == 'true'

        logger.info(f"Initialized demo LLM model: {model_name}")

    async def generate_response(self, context: str, query: str) -> str:
        """Generate a demo response based on query type"""

        # Simulate processing time
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        query_lower = query.lower()

        # Determine response type based on query content
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greeting']):
            response_type = "greeting"
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'main points']):
            response_type = "summarize"
        elif any(word in query_lower for word in ['key findings', 'insights', 'important', 'findings']):
            response_type = "key_findings"
        elif any(word in query_lower for word in ['technical', 'system', 'architecture', 'implementation']):
            response_type = "technical"
        else:
            response_type = "general"

        # Get a random response of the appropriate type
        responses = DEMO_RESPONSES.get(response_type, DEMO_RESPONSES["general"])
        response = random.choice(responses)

        # Add query-specific context
        if "demo" not in response.lower():
            response += f"\n\n*Note: This is a demo response simulating how the AI would answer your query: '{query[:100]}...' based on uploaded document content.*"

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Return demo model information"""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "description": "Demo LLM model simulating OpenAI",
            "cost": "Free (Demo Mode)",
            "status": "Demo Mode Active"
        }

class DemoEmbeddingModel:
    """Demo embedding model that generates fake embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        self.dimension = 384  # Standard dimension for this model

        logger.info(f"Initialized demo embedding model: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate fake embeddings for texts"""
        if not texts:
            return np.array([])

        # Generate deterministic fake embeddings based on text content
        embeddings = []
        for text in texts:
            # Use hash of text for deterministic but realistic-looking embeddings
            seed = hash(text) % 2**32
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.dimension)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings"""
        # Simple dot product for unit vectors gives cosine similarity
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        return np.dot(embeddings1, embeddings2.T)

def is_demo_mode() -> bool:
    """Check if demo mode is enabled"""
    return os.getenv('DEMO_MODE', 'false').lower() == 'true'

def get_demo_sources(query: str = "") -> List[Dict[str, Any]]:
    """Get demo sources based on query"""
    if not os.getenv('DEMO_ENABLE_SOURCES', 'true').lower() == 'true':
        return []

    # Return 2-4 sources with some variation based on query
    num_sources = random.randint(2, min(4, len(DEMO_SOURCES)))
    selected_sources = random.sample(DEMO_SOURCES, num_sources)

    # Add some variation to relevance scores
    for source in selected_sources:
        base_score = source["relevance_score"]
        variation = random.uniform(-0.1, 0.1)
        source["relevance_score"] = max(0.5, min(1.0, base_score + variation))

    # Sort by relevance score descending
    selected_sources.sort(key=lambda x: x["relevance_score"], reverse=True)

    return selected_sources

def create_demo_documents() -> List[Dict[str, Any]]:
    """Create demo documents for the document list"""
    demo_documents = [
        {
            "document_id": "demo_doc_1",
            "filename": "Strategic_Planning_2024.pdf",
            "chunks": 15,
            "timestamp": "2024-01-15T10:30:00",
            "size": "2.3 MB",
            "type": "PDF",
            "status": "Processed"
        },
        {
            "document_id": "demo_doc_2",
            "filename": "Market_Research_Report.docx",
            "chunks": 22,
            "timestamp": "2024-01-14T15:45:00",
            "size": "1.8 MB",
            "type": "Word Document",
            "status": "Processed"
        },
        {
            "document_id": "demo_doc_3",
            "filename": "Technical_Specifications.pdf",
            "chunks": 8,
            "timestamp": "2024-01-13T09:15:00",
            "size": "950 KB",
            "type": "PDF",
            "status": "Processed"
        },
        {
            "document_id": "demo_doc_4",
            "filename": "Financial_Analysis_Q3.xlsx",
            "chunks": 12,
            "timestamp": "2024-01-12T14:20:00",
            "size": "1.2 MB",
            "type": "Excel Spreadsheet",
            "status": "Processed"
        },
        {
            "document_id": "demo_doc_5",
            "filename": "Meeting_Notes_Jan2024.txt",
            "chunks": 5,
            "timestamp": "2024-01-11T11:00:00",
            "size": "45 KB",
            "type": "Text File",
            "status": "Processed"
        }
    ]

    return demo_documents

def get_demo_document_content(document_id: str) -> Dict[str, Any]:
    """Get demo content for a specific document"""
    demo_content = {
        "demo_doc_1": {
            "title": "Strategic Planning 2024",
            "content": """Strategic Planning Document 2024

Executive Summary:
This document outlines our strategic initiatives for 2024, focusing on digital transformation, market expansion, and operational efficiency.

Key Objectives:
1. Increase market share by 15%
2. Implement new technology solutions
3. Optimize operational processes
4. Enhance customer experience

Market Analysis:
Current market conditions show strong growth potential in our target segments. Competition remains fierce, but our unique value proposition positions us well for expansion.

Financial Projections:
Expected revenue growth of 20% over the next fiscal year, with improved profit margins through cost optimization initiatives.

Implementation Timeline:
Q1: Foundation and planning
Q2: Technology rollout
Q3: Market expansion
Q4: Optimization and review

This is demo content to showcase document viewing capabilities."""
        },
        "demo_doc_2": {
            "title": "Market Research Report",
            "content": """Market Research Report - Consumer Trends 2024

Research Methodology:
Conducted comprehensive market analysis using surveys, focus groups, and data analytics across multiple demographics.

Key Findings:
• 78% of consumers prefer digital-first experiences
• Mobile usage has increased by 35% year-over-year
• Sustainability concerns influence 62% of purchasing decisions

Market Segments:
1. Digital Natives (Ages 18-35): High engagement, mobile-first
2. Traditional Consumers (Ages 36-55): Gradual digital adoption
3. Senior Market (Ages 55+): Growing digital literacy

Competitive Landscape:
Market leaders continue to invest heavily in technology and customer experience. New entrants are disrupting traditional models.

Recommendations:
Focus on mobile optimization, sustainable practices, and personalized customer experiences to capture market opportunities.

This demo content illustrates how market research would be presented in the system."""
        },
        "demo_doc_3": {
            "title": "Technical Specifications",
            "content": """System Technical Specifications

Architecture Overview:
Microservices-based architecture with containerized deployment using Docker and Kubernetes orchestration.

Technical Stack:
• Backend: Python 3.11+ with FastAPI framework
• Frontend: HTML5, CSS3, JavaScript ES6+
• Database: PostgreSQL with Redis caching
• Search: Elasticsearch for full-text search
• AI/ML: OpenAI GPT models with custom embeddings

System Requirements:
• CPU: 8+ cores recommended
• RAM: 16GB minimum, 32GB recommended
• Storage: SSD with 100GB+ available space
• Network: High-speed internet connection

Security Features:
• End-to-end encryption
• OAuth 2.0 authentication
• Role-based access control
• Audit logging and monitoring

Performance Metrics:
• Response time: <2 seconds average
• Throughput: 1000+ concurrent users
• Availability: 99.9% uptime SLA

This demonstrates how technical documentation would appear in the document viewer."""
        }
    }

    return demo_content.get(document_id, {
        "title": "Demo Document",
        "content": "This is demo document content showing how the document viewer would display actual file contents."
    })

