import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import numpy as np
from .models import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database wrapper using ChromaDB"""

    def __init__(self,
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-mpnet-base-v2",
                 embedding_api_key: str = None,
                 embedding_base_url: str = None):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_service = EmbeddingService(embedding_model, embedding_api_key, embedding_base_url)

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
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for RAG system"}
            )
            logger.info(f"Created new collection: {collection_name}")

    async def add_documents(self,
                           texts: List[str],
                           metadatas: List[Dict[str, Any]],
                           ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to the vector store"""
        try:
            if not texts:
                raise ValueError("No texts provided")

            if len(texts) != len(metadatas):
                raise ValueError("Number of texts and metadatas must match")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Generate embeddings
            embeddings = self.embedding_service.encode_texts(texts)

            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Added {len(texts)} documents to vector store")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise Exception(f"Failed to add documents: {str(e)}")

    async def search(self,
                    query_text: str,
                    top_k: int = 5,
                    where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.model.encode([query_text])[0].tolist()

            # Search in ChromaDB
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }

            if where_filter:
                search_kwargs["where"] = where_filter

            results = self.collection.query(**search_kwargs)

            # Format results
            formatted_results = []

            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]

                    # For cosine distance, values range from 0 (identical) to 2 (opposite)
                    # Convert to relevance: 0 distance = 100% relevance, 2 distance = 0% relevance
                    relevance_score = max(0, min(100, (1 - distance/2) * 100))

                    logger.debug(f"Distance: {distance}, Relevance: {relevance_score}")

                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "relevance_score": round(relevance_score, 2)
                    }
                    formatted_results.append(result)

            logger.info(f"Search for '{query_text}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")

    async def get_documents(self,
                           ids: Optional[List[str]] = None,
                           where_filter: Optional[Dict[str, Any]] = None,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents from the vector store"""
        try:
            get_kwargs = {"include": ["documents", "metadatas"]}

            if ids:
                get_kwargs["ids"] = ids
            if where_filter:
                get_kwargs["where"] = where_filter
            if limit:
                get_kwargs["limit"] = limit

            results = self.collection.get(**get_kwargs)

            documents = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    doc = {
                        "id": results["ids"][i],
                        "content": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    }
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            raise Exception(f"Failed to get documents: {str(e)}")

    async def update_documents(self,
                              ids: List[str],
                              texts: Optional[List[str]] = None,
                              metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Update existing documents"""
        try:
            update_kwargs = {"ids": ids}

            if texts:
                # Generate new embeddings
                embeddings = self.embedding_service.encode_texts(texts)
                embeddings_list = [emb.tolist() for emb in embeddings]
                update_kwargs["embeddings"] = embeddings_list
                update_kwargs["documents"] = texts

            if metadatas:
                update_kwargs["metadatas"] = metadatas

            self.collection.update(**update_kwargs)

            logger.info(f"Updated {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            return False

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    async def delete_by_filter(self, where_filter: Dict[str, Any]) -> bool:
        """Delete documents matching a filter"""
        try:
            self.collection.delete(where=where_filter)
            logger.info(f"Deleted documents matching filter: {where_filter}")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents by filter: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            model_info = self.embedding_service.get_model_info()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": model_info,
                "collection_metadata": self.collection.metadata if hasattr(self.collection, 'metadata') else {}
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}

    def reset_collection(self) -> bool:
        """Reset (clear) the collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG system"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False

    def get_current_model(self) -> str:
        """Get the current embedding model name"""
        return self.embedding_service.model_name

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return self.embedding_service.get_model_info()

    async def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Get all document IDs
            all_docs = await self.get_documents()
            if all_docs:
                all_ids = [doc["id"] for doc in all_docs]
                await self.delete_documents(all_ids)
            logger.info(f"Cleared all documents from collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def change_embedding_model(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Change the embedding model (note: existing embeddings will become incompatible)"""
        try:
            success = self.embedding_service.change_model(new_model_name, api_key, base_url)
            if success:
                logger.warning(f"Embedding model changed to {new_model_name}. "
                             f"Existing embeddings may be incompatible and should be regenerated.")
            return success

        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            return False