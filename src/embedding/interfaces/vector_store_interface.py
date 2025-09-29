from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class IVectorStore(ABC):
    """Interface for vector storage and retrieval operations"""

    @abstractmethod
    async def add_documents(self,
                           texts: List[str],
                           metadatas: List[Dict[str, Any]] = None,
                           document_ids: List[str] = None) -> List[str]:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    async def similarity_search(self,
                               query: str,
                               k: int = 5,
                               filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        pass

    @abstractmethod
    async def similarity_search_with_score(self,
                                          query: str,
                                          k: int = 5,
                                          filter_dict: Dict[str, Any] = None) -> List[tuple]:
        """Perform similarity search with scores"""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        pass

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        pass

    @abstractmethod
    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        pass