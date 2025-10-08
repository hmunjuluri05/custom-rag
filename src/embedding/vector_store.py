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
        """Add documents to the vector store with validation"""
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

            logger.info(f"Generating embeddings for {len(texts)} text chunks...")

            # Generate embeddings (handle both sync and async)
            embeddings = await self.embedding_service.encode_texts(texts)

            # Validate embeddings were generated for all texts
            if len(embeddings) != len(texts):
                logger.error(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
                raise ValueError(f"Generated {len(embeddings)} embeddings for {len(texts)} texts. Some embeddings failed to generate.")

            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Final validation before adding to ChromaDB
            if not (len(document_ids) == len(embeddings_list) == len(texts) == len(metadatas)):
                logger.error(f"Length mismatch - ids:{len(document_ids)}, embeddings:{len(embeddings_list)}, texts:{len(texts)}, metadatas:{len(metadatas)}")
                raise ValueError(f"Unequal lengths: ids:{len(document_ids)}, embeddings:{len(embeddings_list)}, texts:{len(texts)}, metadatas:{len(metadatas)}")

            logger.info(f"Adding {len(texts)} documents to vector store...")

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

    def _calculate_metadata_score(self, query: str, metadata: Dict[str, Any]) -> float:
        """Calculate metadata matching score for AI-generated metadata"""
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check if this chunk has AI-generated metadata
        has_ai_metadata = metadata.get('chunking_method') in ['llm_semantic', 'llm_enhanced']

        if not has_ai_metadata:
            return 0.0  # No AI metadata to match

        # Score based on keyword matching (30% of metadata score)
        keywords = metadata.get('llm_keywords', '')
        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.split(',')]
            matched_keywords = sum(1 for kw in keyword_list if any(word in kw or kw in word for word in query_words))
            if keyword_list:
                score += (matched_keywords / len(keyword_list)) * 0.3

        # Score based on topic matching (25% of metadata score)
        topic = metadata.get('llm_topic', '').lower()
        if topic:
            topic_match = sum(1 for word in query_words if word in topic or topic in word)
            if topic_match > 0:
                score += 0.25

        # Score based on entity matching (25% of metadata score)
        entities = metadata.get('llm_entities', '')
        if entities:
            entity_list = [e.strip().lower() for e in entities.split(',')]
            matched_entities = sum(1 for ent in entity_list if any(word in ent or ent in word for word in query_words))
            if entity_list:
                score += (matched_entities / len(entity_list)) * 0.25

        # Score based on title matching (20% of metadata score)
        title = metadata.get('llm_title', '').lower()
        if title:
            title_match = sum(1 for word in query_words if word in title)
            if title_match > 0:
                score += (title_match / len(query_words)) * 0.2

        return min(1.0, score)  # Normalize to [0, 1]

    async def hybrid_search(self,
                           query: str,
                           k: int = 5,
                           filter_dict: Dict[str, Any] = None,
                           use_metadata: bool = True,
                           vector_weight: float = 0.7,
                           metadata_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and AI-generated metadata matching.

        Args:
            query: Search query string
            k: Number of results to return (will fetch more internally for reranking)
            filter_dict: Optional filter for document selection
            use_metadata: Whether to use metadata matching (only works with AI-chunked documents)
            vector_weight: Weight for vector similarity score (0-1)
            metadata_weight: Weight for metadata matching score (0-1)

        Returns:
            List of search results ranked by combined score
        """
        try:
            # Normalize weights
            total_weight = vector_weight + metadata_weight
            if total_weight > 0:
                vector_weight = vector_weight / total_weight
                metadata_weight = metadata_weight / total_weight
            else:
                vector_weight = 0.7
                metadata_weight = 0.3

            # Fetch more results than requested for better reranking
            fetch_k = min(k * 3, 50)  # Fetch 3x results or max 50

            # Step 1: Get vector similarity results
            vector_results = await self.search(query, top_k=fetch_k, where_filter=filter_dict)

            if not use_metadata:
                # Return top k vector results only
                return vector_results[:k]

            # Step 2: Calculate metadata scores and combine
            hybrid_results = []
            for result in vector_results:
                metadata = result.get('metadata', {})

                # Get vector similarity score (normalized 0-1)
                vector_score = result.get('similarity_score', 0.0)

                # Calculate metadata matching score
                metadata_score = self._calculate_metadata_score(query, metadata)

                # Combine scores
                hybrid_score = (vector_weight * vector_score) + (metadata_weight * metadata_score)

                # Add to results with hybrid score
                hybrid_result = {
                    **result,
                    'vector_score': vector_score,
                    'metadata_score': metadata_score,
                    'hybrid_score': hybrid_score,
                    'relevance_score': round(hybrid_score * 100, 2)  # Update relevance to hybrid score
                }
                hybrid_results.append(hybrid_result)

            # Step 3: Re-rank by hybrid score
            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # Return top k results
            top_results = hybrid_results[:k]

            logger.info(f"Hybrid search for '{query}' returned {len(top_results)} results "
                       f"(vector_weight={vector_weight:.2f}, metadata_weight={metadata_weight:.2f})")

            return top_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to regular vector search
            logger.warning("Falling back to vector-only search")
            return await self.search(query, top_k=k, where_filter=filter_dict)

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
                    'distance': result.get('distance', 0),
                    'similarity_score': result.get('similarity_score', 0),
                    'relevance_score': result.get('relevance_score', 0)
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