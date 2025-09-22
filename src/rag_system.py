import uuid
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from .embedding.vector_store import VectorStore
from .embedding.chunking import ChunkerFactory, ChunkingConfig, ChunkingStrategy
from .upload.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class RAGSystem:
    """Complete RAG system combining document processing, embeddings, and retrieval"""

    def __init__(self,
                 collection_name: str = "documents",
                 embedding_model: str = "all-mpnet-base-v2",
                 chunking_config: Optional[ChunkingConfig] = None):

        # Initialize chunking configuration
        self.chunking_config = chunking_config or ChunkingConfig()

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model
        )

        logger.info(f"RAG system initialized with model: {embedding_model}, chunking: {self.chunking_config.strategy.value}")

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

            # Split text into chunks using custom config if provided
            chunks = self._chunk_text(text_content, base_metadata, custom_chunking_config)

            if not chunks:
                raise ValueError("No content to process after chunking")

            # Prepare data for vector store
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

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
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_path": metadata.get("file_path", ""),
                        "document_type": metadata.get("document_type", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "total_chunks": metadata.get("total_chunks", 0)
                    }

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise Exception(f"Failed to list documents: {str(e)}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the RAG system"""
        try:
            # Delete from vector store using filter
            success = await self.vector_store.delete_by_filter({"document_id": document_id})

            if success:
                logger.info(f"Deleted document {document_id}")
            else:
                logger.warning(f"Document {document_id} not found")

            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise Exception(f"Failed to delete document: {str(e)}")

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            # Get chunks from vector store
            chunks = await self.vector_store.get_documents(
                where_filter={"document_id": document_id}
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

            # Get actual current model info from the vector store (not cached)
            current_model_info = self.vector_store.get_model_info()

            return {
                "total_chunks": collection_info.get("document_count", 0),
                "unique_documents": collection_info.get("unique_documents", 0),
                "embedding_model": current_model_info,  # Use actual current model info
                "chunking_config": {
                    "strategy": self.chunking_config.strategy.value,
                    "chunk_size": self.chunking_config.chunk_size,
                    "chunk_overlap": self.chunking_config.chunk_overlap,
                    "preserve_sentences": self.chunking_config.preserve_sentences,
                    "preserve_paragraphs": self.chunking_config.preserve_paragraphs
                },
                "collection_name": collection_info.get("collection_name", ""),
                "supported_formats": list(self.document_processor.supported_formats.keys()),
                "available_chunking_strategies": self.get_chunking_strategies()
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

    async def change_embedding_model(self, new_model_name: str, force_reprocess: bool = False) -> Dict[str, Any]:
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
            success = self.vector_store.change_embedding_model(new_model_name)

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

    def change_embedding_model_sync(self, new_model_name: str) -> bool:
        """Synchronous version - Change the embedding model (requires reprocessing documents)"""
        try:
            success = self.vector_store.change_embedding_model(new_model_name)
            if success:
                logger.warning(f"Embedding model changed to {new_model_name}. "
                             f"Existing documents should be reprocessed for optimal results.")
            return success

        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            return False