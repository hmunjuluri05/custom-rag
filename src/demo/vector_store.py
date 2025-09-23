"""
Demo Vector Store
A mock vector store that simulates embedding functionality without external dependencies.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import DemoEmbeddingModel, get_demo_sources

logger = logging.getLogger(__name__)

class DemoVectorStore:
    """Demo vector store that simulates ChromaDB functionality"""

    def __init__(self, collection_name: str = "documents", **kwargs):
        self.collection_name = collection_name
        self.embedding_model = DemoEmbeddingModel()
        self.documents = []  # Store demo documents
        self.embeddings = []  # Store fake embeddings

        logger.info(f"Demo vector store initialized: {collection_name}")

    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to demo vector store"""

        if not texts:
            return []

        if ids is None:
            ids = [f"demo_chunk_{len(self.documents) + i}" for i in range(len(texts))]

        # Generate fake embeddings
        embeddings = self.embedding_model.encode(texts)

        # Store documents
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "embedding": embeddings[i]
            })

        logger.info(f"Added {len(texts)} documents to demo vector store")
        return ids

    async def search(self, query_text: str, top_k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search demo vector store"""

        if not self.documents:
            # Return empty results with demo message
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]

        # Calculate similarities (fake but realistic)
        results = []
        for doc in self.documents:
            # Skip original content placeholders
            if doc["text"] == "ORIGINAL_CONTENT_PLACEHOLDER":
                continue

            # Apply filter if provided
            if where_filter:
                match = True
                for key, value in where_filter.items():
                    if doc["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Calculate fake similarity
            similarity = np.dot(query_embedding, doc["embedding"])
            # Add some randomness but keep it realistic
            similarity = max(0.3, min(1.0, similarity + np.random.normal(0, 0.1)))

            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": float(similarity),
                "distance": float(1.0 - similarity)
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from demo vector store"""

        original_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc["id"] not in ids]

        deleted_count = original_count - len(self.documents)
        logger.info(f"Deleted {deleted_count} documents from demo vector store")

        return deleted_count > 0

    async def get_documents_by_metadata(self, where_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get documents by metadata filter"""

        results = []
        for doc in self.documents:
            match = True
            for key, value in where_filter.items():
                if doc["metadata"].get(key) != value:
                    match = False
                    break

            if match:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                })

        return results

    async def clear_collection(self) -> bool:
        """Clear all documents from demo collection"""

        document_count = len(self.documents)
        self.documents = []
        self.embeddings = []

        logger.info(f"Cleared {document_count} documents from demo vector store")
        return True

    def get_current_model(self) -> str:
        """Get current embedding model name"""
        return self.embedding_model.get_model_name()

    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "model_name": self.embedding_model.get_model_name(),
            "dimension": self.embedding_model.get_dimension(),
            "provider": "demo",
            "description": "Demo embedding model for UI testing",
            "demo_mode": True
        }

    def change_embedding_model(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Change embedding model (demo)"""

        old_model = self.embedding_model.model_name
        self.embedding_model.model_name = f"demo-{new_model_name}" if not new_model_name.startswith("demo") else new_model_name

        logger.info(f"Demo embedding model changed from {old_model} to {self.embedding_model.model_name}")
        return True

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""

        # Calculate stats from stored documents
        total_documents = len(set(doc["metadata"].get("document_id") for doc in self.documents if doc["metadata"].get("document_id")))
        total_chunks = len(self.documents)

        return {
            "collection_name": self.collection_name,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_model.get_model_name(),
            "demo_mode": True,
            "last_updated": datetime.now().isoformat()
        }