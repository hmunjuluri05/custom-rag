import uuid
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from embedding.vector_store import VectorStore
from embedding.chunking import ChunkerFactory, ChunkingConfig, ChunkingStrategy
from upload.document_processor import DocumentProcessor
from llm.models import LLMService
from config import LLMProvider
from config import get_default_llm_config

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system combining document processing, embeddings, and retrieval"""

    def __init__(self,
                 collection_name: str = "documents",
                 embedding_model: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 chunking_config: Optional[ChunkingConfig] = None,
                 llm_provider: LLMProvider = None,
                 llm_model: str = None,
                 use_langchain: bool = True,
                 use_langchain_vectorstore: bool = False):

        # Initialize chunking configuration
        self.chunking_config = chunking_config or ChunkingConfig()

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            use_langchain=use_langchain,
            use_langchain_vectorstore=use_langchain_vectorstore
        )

        # Initialize LLM service with default configuration if not provided
        if llm_provider is None:
            default_provider, default_model, default_api_key, default_base_url = get_default_llm_config()
            llm_provider = default_provider
            llm_model = llm_model or default_model
            api_key = api_key or default_api_key
            base_url = base_url or default_base_url

        # Choose between LangChain and custom implementation
        if use_langchain:
            from llm.langchain_models import LangChainLLMService
            self.llm_service = LangChainLLMService(
                provider=llm_provider,
                model_name=llm_model,
                api_key=api_key,
                base_url=base_url
            )
            logger.info("Using LangChain LLM implementation")
        else:
            from llm.models import LLMService
            self.llm_service = LLMService(
                provider=llm_provider,
                model_name=llm_model,
                api_key=api_key,
                base_url=base_url
            )
            logger.info("Using custom LLM implementation")

        # Initialize agent system for advanced workflows
        self.agent_system = None
        if use_langchain:
            try:
                from agents import MultiAgentRAGSystem
                self.agent_system = MultiAgentRAGSystem(self, self.llm_service)
                logger.info("LangChain agent system initialized for advanced workflows")
            except ImportError as e:
                logger.warning(f"Could not initialize agent system: {e}")

        logger.info(f"RAG system initialized with embedding model: {embedding_model}, "
                   f"chunking: {self.chunking_config.strategy.value}, "
                   f"LLM: {llm_provider.value}")

    def _chunk_text(self, text: str, metadata: Dict[str, Any], custom_config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
        """Split text into chunks using configured chunking strategy"""
        config = custom_config or self.chunking_config
        chunker = ChunkerFactory.create_chunker(config)
        return chunker.chunk_text(text, metadata)

    async def add_document(self, text_content: str, file_path: str, filename: str, custom_chunking_config: Optional[ChunkingConfig] = None) -> str:
        """Add a document to the RAG system"""
        try:
            document_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # Create base metadata
            base_metadata = {
                "document_id": document_id,
                "filename": filename,
                "file_path": file_path,
                "timestamp": timestamp,
                "document_type": filename.split('.')[-1].lower() if '.' in filename else "unknown"
            }

            # Store original content in the first chunk's metadata for document viewing
            original_content_metadata = base_metadata.copy()
            original_content_metadata["original_content"] = text_content
            original_content_metadata["is_original_content"] = True

            # Split text into chunks using custom config if provided
            chunks = self._chunk_text(text_content, base_metadata, custom_chunking_config)

            if not chunks:
                raise ValueError("No content to process after chunking")

            # Prepare data for vector store
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

            # Also store original content for document viewing (non-searchable)
            texts.append("ORIGINAL_CONTENT_PLACEHOLDER")  # Won't be searched
            metadatas.append(original_content_metadata)
            chunk_ids.append(f"{document_id}_original")

            # Add to vector store
            await self.vector_store.add_documents(texts, metadatas, chunk_ids)

            logger.info(f"Added document {filename} with {len(chunks)} chunks")
            return document_id

        except Exception as e:
            logger.error(f"Error adding document {filename}: {str(e)}")
            raise Exception(f"Failed to add document: {str(e)}")

    async def query(self, query_text: str, top_k: int = 5, document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the RAG system for relevant documents"""
        try:
            # Build filter if document_filter is provided
            where_filter = None
            if document_filter:
                where_filter = {"document_id": document_filter}

            # Search in vector store
            results = await self.vector_store.search(
                query_text=query_text,
                top_k=top_k,
                where_filter=where_filter
            )

            # Format results for RAG response
            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": result["content"],
                    "filename": result["metadata"].get("filename", "Unknown"),
                    "document_id": result["metadata"].get("document_id", ""),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "similarity_score": result["similarity_score"],
                    "relevance_score": result["relevance_score"],
                    "source_info": {
                        "filename": result["metadata"].get("filename", "Unknown"),
                        "chunk": f"{result['metadata'].get('chunk_index', 0) + 1}/{result['metadata'].get('total_chunks', 1)}",
                        "document_type": result["metadata"].get("document_type", "unknown")
                    }
                }
                formatted_results.append(formatted_result)

            logger.info(f"Query '{query_text}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise Exception(f"Query failed: {str(e)}")

    async def query_with_llm(self, query_text: str, top_k: int = 5, document_filter: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system and generate response using LLM with source references"""
        try:
            # Get relevant documents
            results = await self.query(query_text, top_k, document_filter)

            if not results:
                response = await self.llm_service.generate_response("", query_text)
                return {
                    "response": response,
                    "sources": [],
                    "query": query_text
                }

            # Combine context from retrieved documents
            context_parts = []
            for result in results:
                context_parts.append(f"From {result['filename']}: {result['content']}")

            context = "\n\n".join(context_parts)

            # Generate response using LLM
            response = await self.llm_service.generate_response(context, query_text)

            # Prepare source information - group by document and show best relevance score
            document_sources = {}
            for result in results:
                doc_id = result["document_id"]
                relevance_score = result.get("relevance_score", 0)

                if doc_id not in document_sources or relevance_score > document_sources[doc_id]["relevance_score"]:
                    document_sources[doc_id] = {
                        "document_id": doc_id,
                        "filename": result["filename"],
                        "relevance_score": relevance_score,
                        "source_info": result.get("source_info", {}),
                        "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                        "chunk_count": 1
                    }
                else:
                    # Increment chunk count for this document
                    document_sources[doc_id]["chunk_count"] += 1

            # Convert to list and sort by relevance score
            sources = list(document_sources.values())
            sources.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.info(f"Generated LLM response for query: '{query_text}'")
            return {
                "response": response,
                "sources": sources,
                "query": query_text
            }

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": [],
                "query": query_text
            }

    async def query_with_llm_stream(self, query_text: str, top_k: int = 5, document_filter: Optional[str] = None):
        """Query the RAG system and generate streaming response using LLM"""
        try:
            # Get relevant documents
            results = await self.query(query_text, top_k, document_filter)

            # Prepare source information first
            document_sources = {}
            context_parts = []

            if results:
                for result in results:
                    context_parts.append(f"From {result['filename']}: {result['content']}")

                    doc_id = result["document_id"]
                    relevance_score = result.get("relevance_score", 0)

                    if doc_id not in document_sources:
                        document_sources[doc_id] = {
                            "document_id": doc_id,
                            "filename": result["filename"],
                            "relevance_score": relevance_score,
                            "source_info": result.get("source_info", {}),
                            "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                            "chunk_count": 1
                        }
                    else:
                        document_sources[doc_id]["chunk_count"] += 1

                # Convert to list and sort by relevance score
                sources = list(document_sources.values())
                sources.sort(key=lambda x: x["relevance_score"], reverse=True)
                context = "\n\n".join(context_parts)
            else:
                sources = []
                context = ""

            # Yield sources information first
            yield {
                "type": "sources",
                "content": sources,
                "query": query_text
            }

            # Stream the response using LLM
            if hasattr(self.llm_service, 'generate_response_stream') and context:
                async for chunk in self.llm_service.generate_response_stream(context, query_text):
                    yield {
                        "type": "response_chunk",
                        "content": chunk
                    }
            else:
                # Fallback to non-streaming
                response = await self.llm_service.generate_response(context, query_text)
                yield {
                    "type": "response",
                    "content": response
                }

            logger.info(f"Generated streaming LLM response for query: '{query_text}'")

        except Exception as e:
            logger.error(f"Error in query_with_llm_stream: {str(e)}")
            yield {
                "type": "error",
                "content": f"Error generating response: {str(e)}"
            }

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the RAG system"""
        try:
            # Get all documents from vector store
            all_docs = await self.vector_store.get_documents()

            # Group by document_id to get unique documents
            documents = {}

            for doc in all_docs:
                metadata = doc["metadata"]
                doc_id = metadata.get("document_id")

                if doc_id and doc_id not in documents:
                    file_size = metadata.get("file_size", 0)

                    # For older documents without file_size, try to get it from file path
                    if file_size == 0:
                        file_path = metadata.get("file_path", "")
                        if file_path:
                            try:
                                import os
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path)
                            except Exception:
                                pass  # If file doesn't exist or can't read, keep file_size as 0

                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_path": metadata.get("file_path", ""),
                        "document_type": metadata.get("document_type", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "file_size": file_size
                    }

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise Exception(f"Failed to list documents: {str(e)}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the RAG system and file storage"""
        try:
            # First, get the document metadata to find the file path
            documents = await self.list_documents()
            file_path = None
            logger.info(f"Looking for document {document_id} among {len(documents)} documents")

            for doc in documents:
                if doc["document_id"] == document_id:
                    file_path = doc.get("file_path")
                    logger.info(f"Found document {document_id} with file_path: {file_path}")
                    break

            if not file_path:
                logger.warning(f"No file_path found for document {document_id}")

            # Delete from vector store using correct ChromaDB filter syntax
            success = await self.vector_store.delete_by_filter({"document_id": {"$eq": document_id}})

            # If vector deletion was successful and we have a file path, delete the file
            if success and file_path:
                try:
                    import os
                    logger.info(f"Attempting to delete file: {file_path}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Successfully deleted file: {file_path}")
                    else:
                        logger.warning(f"File not found for deletion: {file_path}")
                except Exception as file_error:
                    logger.error(f"Error deleting file {file_path}: {str(file_error)}")
                    # Continue execution - vector deletion was successful
            elif success and not file_path:
                logger.warning(f"Vector deletion successful but no file_path to delete for document {document_id}")

            if success:
                logger.info(f"Deleted document {document_id} from vector store and file system")
            else:
                logger.warning(f"Document {document_id} not found in vector store")

            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise Exception(f"Failed to delete document: {str(e)}")

    async def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents from both vector store and file system"""
        try:
            # Get all documents first to know which files to delete
            documents = await self.list_documents()
            file_paths = [doc.get("file_path") for doc in documents if doc.get("file_path")]

            # Clear vector store
            success = await self.vector_store.clear_collection()

            # Delete all files
            deleted_files = []
            failed_files = []

            if success and file_paths:
                import os
                for file_path in file_paths:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            deleted_files.append(file_path)
                            logger.info(f"Deleted file: {file_path}")
                        else:
                            failed_files.append(f"{file_path} (not found)")
                    except Exception as file_error:
                        failed_files.append(f"{file_path} (error: {str(file_error)})")
                        logger.error(f"Error deleting file {file_path}: {str(file_error)}")

            result = {
                "success": success,
                "vector_store_cleared": success,
                "total_documents_removed": len(documents),
                "files_deleted": len(deleted_files),
                "files_failed": len(failed_files),
                "deleted_file_paths": deleted_files,
                "failed_file_paths": failed_files
            }

            if success:
                logger.info(f"Cleared all documents: {len(documents)} from vector store, {len(deleted_files)} files deleted")

            return result

        except Exception as e:
            logger.error(f"Error clearing all documents: {str(e)}")
            raise Exception(f"Failed to clear all documents: {str(e)}")

    async def get_original_document(self, document_id: str) -> Dict[str, Any]:
        """Get the original document content for viewing"""
        try:
            # For now, let's just return chunks for existing documents
            # since they don't have original content stored
            # This will be fixed when new documents are uploaded
            return await self.get_document_chunks(document_id)

        except Exception as e:
            logger.error(f"Error getting original document {document_id}: {str(e)}")
            # Fallback to chunks
            return await self.get_document_chunks(document_id)

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            # Get chunks from vector store using correct ChromaDB filter syntax
            chunks = await self.vector_store.get_documents(
                where_filter={"document_id": {"$eq": document_id}}
            )

            # Sort by chunk index
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            formatted_chunks = []
            for chunk in chunks:
                formatted_chunk = {
                    "chunk_id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "chunk_index": chunk["metadata"].get("chunk_index", 0),
                    "word_range": f"{chunk['metadata'].get('start_word', 0)}-{chunk['metadata'].get('end_word', 0)}"
                }
                formatted_chunks.append(formatted_chunk)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            raise Exception(f"Failed to get document chunks: {str(e)}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        try:
            collection_info = self.vector_store.get_collection_info()

            # Calculate unique documents synchronously
            unique_documents_count = 0
            try:
                # Get all documents synchronously from ChromaDB
                results = self.vector_store.collection.get(include=["metadatas"])
                unique_doc_ids = set()

                if results["metadatas"]:
                    for metadata in results["metadatas"]:
                        doc_id = metadata.get("document_id")
                        if doc_id:
                            unique_doc_ids.add(doc_id)

                unique_documents_count = len(unique_doc_ids)
            except Exception as e:
                logger.warning(f"Error calculating unique documents: {str(e)}")
                unique_documents_count = 0

            # Get actual current model info from the vector store (not cached)
            current_model_info = self.vector_store.get_model_info()

            # Handle serialization of enum values
            def serialize_model_info(info):
                if isinstance(info, dict):
                    serialized = {}
                    for key, value in info.items():
                        if hasattr(value, 'value'):  # This is an enum
                            serialized[key] = value.value
                        elif isinstance(value, dict):
                            serialized[key] = serialize_model_info(value)  # Recursive for nested dicts
                        elif isinstance(value, list):
                            serialized[key] = [serialize_model_info(item) for item in value]  # Handle lists
                        else:
                            serialized[key] = value
                    return serialized
                elif isinstance(info, list):
                    return [serialize_model_info(item) for item in info]
                elif hasattr(info, 'value'):  # This is an enum
                    return info.value
                return info

            # Serialize model info to handle enum values
            serializable_model_info = serialize_model_info(current_model_info)

            # Also serialize chunking strategies
            chunking_strategies = serialize_model_info(self.get_chunking_strategies())

            return {
                "total_chunks": collection_info.get("document_count", 0),
                "unique_documents": unique_documents_count,
                "embedding_model": serializable_model_info,  # Use serialized model info
                "chunking_config": {
                    "strategy": self.chunking_config.strategy.value,
                    "chunk_size": self.chunking_config.chunk_size,
                    "chunk_overlap": self.chunking_config.chunk_overlap,
                    "preserve_sentences": self.chunking_config.preserve_sentences,
                    "preserve_paragraphs": self.chunking_config.preserve_paragraphs
                },
                "collection_name": collection_info.get("collection_name", ""),
                "supported_formats": list(self.document_processor.supported_formats.keys()),
                "available_chunking_strategies": chunking_strategies
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}

    async def process_and_add_document(self, file_path: str, filename: str) -> str:
        """Process a document file and add it to the RAG system"""
        try:
            from pathlib import Path

            # Extract text using document processor
            text_content = self.document_processor.extract_text(Path(file_path))

            # Add to RAG system
            document_id = await self.add_document(text_content, file_path, filename)

            return document_id

        except Exception as e:
            logger.error(f"Error processing and adding document {filename}: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")

    async def update_chunking_config(self, new_config: ChunkingConfig):
        """Update chunking configuration"""
        self.chunking_config = new_config
        logger.info(f"Updated chunking config: strategy={new_config.strategy.value}, size={new_config.chunk_size}")

    def get_chunking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get available chunking strategies"""
        return ChunkerFactory.get_available_strategies()

    def get_current_embedding_model(self) -> str:
        """Get the current embedding model name"""
        return self.vector_store.get_current_model()

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return self.vector_store.get_model_info()

    async def change_embedding_model(self, new_model_name: str, api_key: str = None, base_url: str = None, force_reprocess: bool = False) -> Dict[str, Any]:
        """Change the embedding model (requires reprocessing documents)"""
        try:
            # Get current stats before change
            current_stats = self.get_system_stats()
            current_model = self.get_current_embedding_model()

            if current_model == new_model_name:
                return {
                    "success": True,
                    "message": f"Model {new_model_name} is already active",
                    "current_model": current_model,
                    "documents_affected": 0
                }

            # Check if there are existing documents
            documents = await self.list_documents()

            if documents and not force_reprocess:
                return {
                    "success": False,
                    "message": f"Cannot change model from {current_model} to {new_model_name}. "
                              f"{len(documents)} existing documents would become incompatible. "
                              f"Set force_reprocess=true to reprocess all documents.",
                    "current_model": current_model,
                    "new_model": new_model_name,
                    "documents_affected": len(documents),
                    "requires_reprocessing": True
                }

            # Change the model
            success = self.vector_store.change_embedding_model(new_model_name, api_key, base_url)

            if not success:
                return {
                    "success": False,
                    "message": f"Failed to change model to {new_model_name}",
                    "current_model": current_model
                }

            result = {
                "success": True,
                "message": f"Successfully changed embedding model from {current_model} to {new_model_name}",
                "previous_model": current_model,
                "current_model": new_model_name,
                "documents_affected": len(documents)
            }

            # If there were documents and force_reprocess is True, reprocess them
            if documents and force_reprocess:
                # Clear existing documents first
                await self.vector_store.clear_collection()

                reprocessed = 0
                failed = []

                # Note: This is a simplified reprocessing - in a real system,
                # you'd want to store original document content or file paths
                logger.warning(f"Reprocessing {len(documents)} documents with new model {new_model_name}")
                result["reprocessing"] = {
                    "total_documents": len(documents),
                    "reprocessed": reprocessed,
                    "failed": failed,
                    "note": "Document reprocessing requires original files to be re-uploaded"
                }

            return result

        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            return {
                "success": False,
                "message": f"Error changing embedding model: {str(e)}",
                "current_model": self.get_current_embedding_model()
            }

    def change_embedding_model_sync(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Synchronous version - Change the embedding model (requires reprocessing documents)"""
        try:
            success = self.vector_store.change_embedding_model(new_model_name, api_key, base_url)
            if success:
                logger.warning(f"Embedding model changed to {new_model_name}. "
                             f"Existing documents should be reprocessed for optimal results.")
            return success

        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            return False

    def change_llm(self, provider: LLMProvider, model_name: str = None, api_key: str = None, base_url: str = None) -> bool:
        """Change the LLM provider and model"""
        try:
            success = self.llm_service.change_model(provider, model_name, api_key, base_url)
            if success:
                logger.info(f"LLM changed to {provider.value}: {model_name}")
            return success
        except Exception as e:
            logger.error(f"Error changing LLM: {str(e)}")
            return False

    def get_available_llms(self) -> Dict[str, Dict[str, Any]]:
        """Get available LLM models"""
        from llm.models import LLMFactory
        return LLMFactory.get_available_models()

    def get_llm_info(self) -> Dict[str, Any]:
        """Get current LLM model information"""
        return self.llm_service.get_model_info()

    async def query_with_agent(self, query_text: str, agent_type: str = "general") -> Dict[str, Any]:
        """Query using LangChain agents for advanced reasoning"""
        if not self.agent_system:
            return {
                "response": "Agent system not available. Ensure use_langchain=True when creating the RAG system.",
                "agent_type": agent_type,
                "query": query_text,
                "fallback": True
            }

        try:
            result = await self.agent_system.route_query(query_text, agent_type)
            logger.info(f"Agent query completed for: '{query_text}'")
            return result

        except Exception as e:
            logger.error(f"Error in agent query: {str(e)}")
            return {
                "response": f"Agent error: {str(e)}. Falling back to standard RAG.",
                "agent_type": agent_type,
                "query": query_text,
                "fallback": True
            }

    async def query_with_agent_stream(self, query_text: str, agent_type: str = "general"):
        """Stream agent response for real-time reasoning"""
        if not self.agent_system:
            yield {
                "type": "error",
                "content": "Agent system not available. Ensure use_langchain=True when creating the RAG system.",
                "agent_type": agent_type
            }
            return

        try:
            async for chunk in self.agent_system.route_query_stream(query_text, agent_type):
                yield chunk

            logger.info(f"Agent streaming query completed for: '{query_text}'")

        except Exception as e:
            logger.error(f"Error in agent streaming query: {str(e)}")
            yield {
                "type": "error",
                "content": f"Agent streaming error: {str(e)}",
                "agent_type": agent_type
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent system"""
        if not self.agent_system:
            return {"error": "Agent system not available"}

        return self.agent_system.get_system_info()


def create_rag_system(**kwargs) -> 'RAGSystem':
    """
    Factory function to create RAG system with LangChain as the default configuration.

    For maximum performance and features, use LangChain implementations:
    - use_langchain=True (default): Uses LangChain for LLM and embeddings
    - use_langchain_vectorstore=True: Uses LangChain-compatible vector store

    Legacy mode (not recommended for new development):
    - use_langchain=False: Uses deprecated custom implementations
    """
    logger.info("Creating RAG system with LangChain integration")

    # Provide defaults if not specified in kwargs
    if 'api_key' not in kwargs:
        from config.model_config import get_kong_config
        kwargs['api_key'] = get_kong_config()

    if 'base_url' not in kwargs:
        # base_url will be derived by the factories, so no need to set it here
        pass

    # Use LangChain by default unless explicitly disabled
    if 'use_langchain' not in kwargs:
        kwargs['use_langchain'] = True

    # Warn if legacy mode is explicitly requested
    if kwargs.get('use_langchain') is False:
        import warnings
        warnings.warn(
            "Legacy mode (use_langchain=False) is deprecated and not recommended for new development. "
            "Consider migrating to LangChain implementations for better features and performance.",
            DeprecationWarning,
            stacklevel=2
        )

    return RAGSystem(**kwargs)
