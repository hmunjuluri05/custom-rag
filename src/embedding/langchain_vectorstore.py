from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import logging
from langchain.vectorstores.base import VectorStore as LangChainVectorStore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import chromadb
from chromadb.config import Settings
import numpy as np
import asyncio

logger = logging.getLogger(__name__)


class LangChainChromaEmbeddingWrapper(Embeddings):
    """Wrapper to make our LangChain embedding models compatible with LangChain VectorStore interface"""

    def __init__(self, langchain_embedding_model):
        self.embedding_model = langchain_embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs synchronously."""
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = loop.run_until_complete(self.embedding_model.encode(texts))
            # Convert numpy arrays to lists
            return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        finally:
            loop.close()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text synchronously."""
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = loop.run_until_complete(self.embedding_model.encode([text]))
            embedding = embeddings[0]
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        finally:
            loop.close()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs asynchronously."""
        embeddings = await self.embedding_model.encode(texts)
        # Convert numpy arrays to lists
        return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text asynchronously."""
        embeddings = await self.embedding_model.encode([text])
        embedding = embeddings[0]
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding


class LangChainChromaVectorStore(LangChainVectorStore):
    """LangChain-compatible ChromaDB vector store implementation"""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        **kwargs
    ):
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Loaded existing LangChain ChromaDB collection: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "LangChain compatible document embeddings for RAG system"}
            )
            logger.info(f"Created new LangChain ChromaDB collection: {collection_name}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings using the embedding function
        embeddings = self.embedding_function.embed_documents(texts)

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        # Convert embeddings to list format for ChromaDB
        embeddings_list = [emb if isinstance(emb, list) else emb.tolist() for emb in embeddings]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added {len(texts)} texts to LangChain ChromaDB vector store")
        return ids

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add_texts."""
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings using async embedding function
        embeddings = await self.embedding_function.aembed_documents(texts)

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        # Convert embeddings to list format for ChromaDB
        embeddings_list = [emb if isinstance(emb, list) else emb.tolist() for emb in embeddings]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added {len(texts)} texts to LangChain ChromaDB vector store (async)")
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)

        # Prepare search arguments
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter:
            search_kwargs["where"] = filter

        # Search in ChromaDB
        results = self.collection.query(**search_kwargs)

        # Convert to LangChain Document format
        documents = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = Document(
                    page_content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                )
                documents.append(doc)

        logger.info(f"LangChain similarity search returned {len(documents)} documents")
        return documents

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        """Async version of similarity_search."""
        # Generate query embedding
        query_embedding = await self.embedding_function.aembed_query(query)

        # Prepare search arguments
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter:
            search_kwargs["where"] = filter

        # Search in ChromaDB
        results = self.collection.query(**search_kwargs)

        # Convert to LangChain Document format
        documents = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = Document(
                    page_content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                )
                documents.append(doc)

        logger.info(f"LangChain async similarity search returned {len(documents)} documents")
        return documents

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1]."""
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)

        # Prepare search arguments
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter:
            search_kwargs["where"] = filter

        # Search in ChromaDB
        results = self.collection.query(**search_kwargs)

        # Convert to LangChain Document format with scores
        documents_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = Document(
                    page_content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                )

                # Convert distance to similarity score (0-1 range)
                distance = results["distances"][0][i]
                # For cosine distance, convert to similarity: similarity = 1 - distance
                similarity_score = max(0.0, min(1.0, 1.0 - distance))

                documents_with_scores.append((doc, similarity_score))

        logger.info(f"LangChain similarity search with scores returned {len(documents_with_scores)} documents")
        return documents_with_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        # This is a simplified implementation
        # For a full MMR implementation, you would need to implement the MMR algorithm
        # For now, we'll use regular similarity search
        return self.similarity_search(query, k=k, filter=filter, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from LangChain ChromaDB vector store")
            return True
        return False

    def get(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get documents by IDs or filter."""
        get_kwargs = {"include": ["documents", "metadatas"]}

        if ids:
            get_kwargs["ids"] = ids
        if filter:
            get_kwargs["where"] = filter

        results = self.collection.get(**get_kwargs)

        documents = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                doc = Document(
                    page_content=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {}
                )
                documents.append(doc)

        return {
            "ids": results["ids"],
            "documents": documents,
            "metadatas": results.get("metadatas", [])
        }

    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
        """Update documents in the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate new embeddings
        embeddings = self.embedding_function.embed_documents(texts)
        embeddings_list = [emb if isinstance(emb, list) else emb.tolist() for emb in embeddings]

        # Update in ChromaDB
        self.collection.update(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Updated {len(ids)} documents in LangChain ChromaDB vector store")

    def reset_collection(self) -> bool:
        """Reset (clear) the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "LangChain compatible document embeddings for RAG system"}
            )
            logger.info(f"Reset LangChain ChromaDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting LangChain ChromaDB collection: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_function": str(type(self.embedding_function).__name__),
                "collection_metadata": getattr(self.collection, 'metadata', {}),
                "vector_store_type": "LangChain ChromaDB"
            }
        except Exception as e:
            logger.error(f"Error getting LangChain ChromaDB collection info: {str(e)}")
            return {"error": str(e)}

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        **kwargs: Any,
    ) -> "LangChainChromaVectorStore":
        """Create a vector store from a list of texts."""
        vector_store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )

        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        **kwargs: Any,
    ) -> "LangChainChromaVectorStore":
        """Async version of from_texts."""
        vector_store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )

        await vector_store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_store

    def _get_retriever_tags(self) -> List[str]:
        """Get tags for retriever identification."""
        return ["LangChainChromaVectorStore", "ChromaDB", "RAG"]