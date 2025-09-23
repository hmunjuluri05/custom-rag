"""
Demo Module for RAG System
Provides demonstration functionality without external dependencies.
"""

from .models import (
    DemoLLMModel,
    DemoEmbeddingModel,
    is_demo_mode,
    get_demo_sources,
    create_demo_documents,
    get_demo_document_content
)

from .rag_system import (
    DemoRAGSystem,
    create_demo_rag_system
)

__all__ = [
    'DemoLLMModel',
    'DemoEmbeddingModel',
    'DemoRAGSystem',
    'is_demo_mode',
    'get_demo_sources',
    'create_demo_documents',
    'get_demo_document_content',
    'create_demo_rag_system'
]