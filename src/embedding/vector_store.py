import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import logging
from .models import EmbeddingService
from .interfaces.vector_store_interface import IVectorStore

logger = logging.getLogger(__name__)

class VectorStore(IVectorStore):
    """Vector database wrapper using ChromaDB with optional LangChain integration"""

    def __init__(self,
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = None,
                 use_langchain_vectorstore: bool = False):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_langchain_vectorstore = use_langchain_vectorstore

        # Use modern embedding implementation
        # EmbeddingService now handles its own configuration, just pass model_name
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        logger.info("Using modern embedding implementation")

        # Choose vector store implementation
        if use_langchain_vectorstore:
            # Use modern vector store abstraction
            from .langchain_vectorstore import LangChainChromaVectorStore, LangChainChromaEmbeddingWrapper
            from .models import EmbeddingModelFactory

            # Create embedding function
            embedding_model_instance = EmbeddingModelFactory.create_model(model_name=embedding_model)
            self.langchain_embedding = LangChainChromaEmbeddingWrapper(embedding_model_instance)

            self.langchain_vectorstore = LangChainChromaVectorStore(
                embedding_function=self.langchain_embedding,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            logger.info("Using LangChain vector store implementation")
        else:
            self.langchain_vectorstore = None

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
                           metadatas: List[Dict[str, Any]] = None,
                           document_ids: List[str] = None) -> List[str]:
        """Add documents to the vector store"""
        try:
            if not texts:
                raise ValueError("No texts provided")

            # Handle optional metadatas
            if metadatas is None:
                metadatas = [{} for _ in texts]
            elif len(texts) != len(metadatas):
                raise ValueError("Number of texts and metadatas must match")

            # Generate IDs if not provided
            if document_ids is None:
                document_ids = [str(uuid.uuid4()) for _ in texts]

            # Generate embeddings (handle both sync and async)
            embeddings = await self.embedding_service.encode_texts(texts)

            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to ChromaDB
            self.collection.add(
                ids=document_ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Added {len(texts)} documents to vector store")
            return document_ids

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
            query_embeddings = await self.embedding_service.encode_texts([query_text])
            query_embedding = query_embeddings[0].tolist()

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
                    relevance_score = max(0.0, min(100.0, (1 - distance/2) * 100))

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
                embeddings = await self.embedding_service.encode_texts(texts)
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

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
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

    # Interface implementation methods
    async def similarity_search(self,
                               query: str,
                               k: int = 5,
                               filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            results = await self.search(
                query_text=query,
                top_k=k,
                where_filter=filter_dict
            )

            # Convert to expected interface format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'id': result.get('id', ''),
                    'distance': result.get('distance', 0)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    async def similarity_search_with_score(self,
                                          query: str,
                                          k: int = 5,
                                          filter_dict: Dict[str, Any] = None) -> List[tuple]:
        """Perform similarity search with scores"""
        try:
            results = await self.search(
                query_text=query,
                top_k=k,
                where_filter=filter_dict
            )

            # Convert to expected interface format (doc, score) tuples
            scored_results = []
            for result in results:
                doc = {
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'id': result.get('id', '')
                }
                score = 1 - result.get('distance', 0)  # Convert distance to similarity score
                scored_results.append((doc, score))

            return scored_results

        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.get_collection_info()

            # Get document count
            document_count = collection_info.get('document_count', 0)

            # Get unique document IDs count (for unique documents)
            all_docs = await self.get_documents()
            unique_doc_ids = set()
            for doc in all_docs:
                doc_id = doc.get('metadata', {}).get('document_id')
                if doc_id:
                    unique_doc_ids.add(doc_id)

            return {
                'total_chunks': document_count,
                'unique_documents': len(unique_doc_ids),
                'collection_name': self.collection_name,
                'embedding_model': self.get_current_model(),
                'persist_directory': self.persist_directory
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return self.get_model_info()