"""
Demo RAG System
A demonstration version of the RAG system that works without external dependencies.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    DemoLLMModel,
    DemoEmbeddingModel,
    is_demo_mode,
    get_demo_sources,
    create_demo_documents,
    get_demo_document_content
)
from .vector_store import DemoVectorStore

logger = logging.getLogger(__name__)

class DemoRAGSystem:
    """Demo RAG system for UI demonstrations"""

    def __init__(self):
        self.llm_model = DemoLLMModel()
        self.embedding_model = DemoEmbeddingModel()
        self.vector_store = DemoVectorStore()
        self.demo_documents = create_demo_documents()

        # Pre-populate with demo documents
        self._initialize_demo_data()

        logger.info("Demo RAG system initialized")

    def _initialize_demo_data(self):
        """Initialize demo data in vector store"""
        try:
            # Add demo document chunks to vector store
            for doc in self.demo_documents:
                content_data = get_demo_document_content(doc["document_id"])
                content = content_data.get("content", "Demo content")

                # Split into chunks
                sentences = content.split('. ')
                chunk_size = 3

                texts = []
                metadatas = []
                ids = []

                for i in range(0, len(sentences), chunk_size):
                    chunk_text = '. '.join(sentences[i:i + chunk_size])
                    if chunk_text and not chunk_text.endswith('.'):
                        chunk_text += '.'

                    chunk_id = f"{doc['document_id']}_chunk_{i // chunk_size}"
                    texts.append(chunk_text)
                    metadatas.append({
                        "document_id": doc["document_id"],
                        "filename": doc["filename"],
                        "chunk_index": i // chunk_size,
                        "timestamp": doc["timestamp"],
                        "document_type": doc["type"].lower()
                    })
                    ids.append(chunk_id)

                # Add to vector store (synchronous call for initialization)
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.vector_store.add_documents(texts, metadatas, ids))
                loop.close()

            logger.info("Demo data initialized in vector store")

        except Exception as e:
            logger.warning(f"Could not initialize demo data: {e}")
            # Continue without demo data - it's not critical

    async def query(self, query: str, document_filter: str = None) -> Dict[str, Any]:
        """Process a query and return demo response with sources"""

        # Simulate processing time
        await asyncio.sleep(0.5)

        # Generate response using demo LLM
        response = await self.llm_model.generate_response("", query)

        # Get demo sources
        sources = get_demo_sources(query)

        return {
            "response": response,
            "sources": sources,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True
        }

    async def add_document(self, text_content: str, file_path: str, filename: str) -> str:
        """Simulate adding a document"""

        # Simulate processing time
        await asyncio.sleep(1)

        # Create a demo document entry
        document_id = f"demo_doc_{len(self.demo_documents) + 1}"

        demo_doc = {
            "document_id": document_id,
            "filename": filename,
            "chunks": len(text_content.split()) // 50 + 1,  # Rough chunk estimate
            "timestamp": datetime.now().isoformat(),
            "size": f"{len(text_content) // 1024} KB",
            "type": filename.split('.')[-1].upper() if '.' in filename else "Unknown",
            "status": "Processed"
        }

        self.demo_documents.append(demo_doc)

        logger.info(f"Demo document added: {filename}")
        return document_id

    async def list_documents(self) -> List[Dict[str, Any]]:
        """Return list of demo documents"""
        return self.demo_documents

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get demo document chunks"""

        # Find the document
        doc = next((d for d in self.demo_documents if d["document_id"] == document_id), None)
        if not doc:
            return []

        # Get demo content
        content_data = get_demo_document_content(document_id)

        # Split content into chunks for demo
        content = content_data.get("content", "Demo content for document viewing.")
        sentences = content.split('. ')

        chunks = []
        chunk_size = 3  # Sentences per chunk

        for i in range(0, len(sentences), chunk_size):
            chunk_text = '. '.join(sentences[i:i + chunk_size])
            if chunk_text and not chunk_text.endswith('.'):
                chunk_text += '.'

            chunks.append({
                "chunk_id": f"{document_id}_chunk_{i // chunk_size}",
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "filename": doc["filename"],
                    "chunk_index": i // chunk_size,
                    "timestamp": doc["timestamp"],
                    "document_type": doc["type"].lower()
                }
            })

        return chunks

    async def get_original_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get original demo document content"""

        # Find the document
        doc = next((d for d in self.demo_documents if d["document_id"] == document_id), None)
        if not doc:
            return None

        # Get demo content
        content_data = get_demo_document_content(document_id)

        return [{
            "text": content_data.get("content", "Demo document content"),
            "metadata": {
                "document_id": document_id,
                "filename": doc["filename"],
                "timestamp": doc["timestamp"],
                "document_type": doc["type"].lower(),
                "is_original_content": True,
                "title": content_data.get("title", doc["filename"])
            }
        }]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a demo document"""

        # Find and remove the document
        original_count = len(self.demo_documents)
        self.demo_documents = [d for d in self.demo_documents if d["document_id"] != document_id]

        deleted = len(self.demo_documents) < original_count
        if deleted:
            logger.info(f"Demo document deleted: {document_id}")

        return deleted

    def get_system_stats(self) -> Dict[str, Any]:
        """Get demo system statistics"""

        total_chunks = sum(doc.get("chunks", 0) for doc in self.demo_documents)

        return {
            "total_documents": len(self.demo_documents),
            "total_chunks": total_chunks,
            "embedding_model": {
                "model_name": self.embedding_model.get_model_name(),
                "provider": "local",
                "dimension": self.embedding_model.get_dimension(),
                "demo_mode": True
            },
            "llm_model": self.llm_model.get_model_info()["model_name"],
            "demo_mode": True,
            "system_status": "Demo Mode Active",
            "last_updated": datetime.now().isoformat()
        }

    def get_current_llm_model(self) -> str:
        """Get current LLM model name"""
        return self.llm_model.get_model_info()["model_name"]

    def get_current_embedding_model(self) -> str:
        """Get current embedding model name"""
        return self.embedding_model.get_model_name()

    def get_llm_model_info(self) -> Dict[str, Any]:
        """Get LLM model information"""
        return self.llm_model.get_model_info()

    def get_llm_info(self) -> Dict[str, Any]:
        """Get LLM model information (alias for API compatibility)"""
        return self.get_llm_model_info()

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "model_name": self.embedding_model.get_model_name(),
            "dimension": self.embedding_model.get_dimension(),
            "provider": "local",
            "description": "Demo embedding model simulating Hugging Face",
            "demo_mode": True
        }

    async def change_llm_model(self, provider: str, model_name: str = None, **kwargs) -> Dict[str, Any]:
        """Simulate changing LLM model"""

        if is_demo_mode():
            # Just update the model name for demo
            old_model = self.llm_model.model_name
            new_model = model_name or "gpt-4"
            self.llm_model.model_name = new_model

            return {
                "success": True,
                "message": f"LLM model changed from {old_model} to {new_model}",
                "old_model": old_model,
                "new_model": new_model,
                "demo_mode": True
            }
        else:
            return {
                "success": False,
                "message": "Non-demo models not available in demo mode",
                "demo_mode": True
            }

    async def change_embedding_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Simulate changing embedding model"""

        old_model = self.embedding_model.model_name
        new_model = model_name if "demo" in model_name else f"demo-{model_name}"
        self.embedding_model.model_name = new_model

        return {
            "success": True,
            "message": f"Demo embedding model changed from {old_model} to {new_model}",
            "old_model": old_model,
            "new_model": new_model,
            "demo_mode": True,
            "documents_affected": 0  # No reprocessing needed in demo
        }

    def get_available_llms(self) -> Dict[str, Dict[str, Any]]:
        """Get available LLM models (delegates to real models for demo)"""
        return {
            "openai": {
                "gpt-4": {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "description": "GPT-4 - Most capable model",
                    "cost": "Free (Demo)",
                    "recommended": True
                },
                "gpt-3.5-turbo": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "description": "GPT-3.5 Turbo - Fast and efficient",
                    "cost": "Free (Demo)",
                    "recommended": False
                }
            }
        }

    def get_available_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Get available embedding models (delegates to real factory)"""
        from ..embedding.models import EmbeddingModelFactory
        return EmbeddingModelFactory.get_available_models()

    def change_llm(self, provider, model_name: str = None, api_key: str = None, base_url: str = None) -> bool:
        """Change LLM model (demo version)"""
        try:
            old_model = self.llm_model.model_name
            new_model = model_name or "gpt-4"
            self.llm_model.model_name = new_model
            logger.info(f"Demo LLM changed from {old_model} to {new_model}")
            return True
        except Exception as e:
            logger.error(f"Error changing demo LLM: {e}")
            return False

    async def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents (demo version)"""
        try:
            document_count = len(self.demo_documents)
            self.demo_documents = []
            await self.vector_store.clear_collection()

            logger.info(f"Demo: Cleared {document_count} documents")
            return {
                "success": True,
                "message": f"Cleared {document_count} demo documents",
                "documents_removed": document_count,
                "demo_mode": True
            }
        except Exception as e:
            logger.error(f"Error clearing demo documents: {e}")
            return {
                "success": False,
                "message": f"Error clearing documents: {str(e)}",
                "demo_mode": True
            }

def create_demo_rag_system() -> DemoRAGSystem:
    """Factory function to create demo RAG system"""
    return DemoRAGSystem()