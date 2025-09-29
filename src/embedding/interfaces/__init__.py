"""
Embedding and Vector Store interfaces for loose coupling and dependency injection.
"""

from .embedding_model_interface import IEmbeddingModel, IEmbeddingModelFactory
from .vector_store_interface import IVectorStore

__all__ = [
    "IEmbeddingModel",
    "IEmbeddingModelFactory",
    "IVectorStore"
]