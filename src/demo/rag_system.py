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

logger = logging.getLogger(__name__)

class DemoRAGSystem:
    """Demo RAG system for UI demonstrations"""

    def __init__(self):
        self.llm_model = DemoLLMModel()
        self.embedding_model = DemoEmbeddingModel()
        self.demo_documents = create_demo_documents()

        logger.info("Demo RAG system initialized")

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
            "embedding_model": self.embedding_model.get_model_name(),
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

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "model_name": self.embedding_model.get_model_name(),
            "dimension": self.embedding_model.get_dimension(),
            "provider": "demo",
            "description": "Demo embedding model for UI testing",
            "demo_mode": True
        }

    async def change_llm_model(self, provider: str, model_name: str = None, **kwargs) -> Dict[str, Any]:
        """Simulate changing LLM model"""

        if provider == "demo" or is_demo_mode():
            # Just update the model name for demo
            old_model = self.llm_model.model_name
            new_model = model_name or "demo-gpt-4"
            self.llm_model.model_name = new_model

            return {
                "success": True,
                "message": f"Demo LLM model changed from {old_model} to {new_model}",
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
        """Get available demo LLM models"""
        return {
            "demo": {
                "demo-gpt-4": {
                    "provider": "demo",
                    "model_name": "demo-gpt-4",
                    "description": "Demo GPT-4 model for testing",
                    "cost": "Free (Demo)",
                    "recommended": True
                },
                "demo-gpt-3.5": {
                    "provider": "demo",
                    "model_name": "demo-gpt-3.5",
                    "description": "Demo GPT-3.5 model for testing",
                    "cost": "Free (Demo)",
                    "recommended": False
                }
            }
        }

    def get_available_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Get available demo embedding models"""
        return {
            "demo-embeddings": {
                "provider": "demo",
                "dimension": 384,
                "description": "Demo embedding model for UI testing",
                "size": "Demo",
                "category": "Demo",
                "recommended": True,
                "requires_api_key": False,
                "cost": "Free"
            },
            "demo-large-embeddings": {
                "provider": "demo",
                "dimension": 1536,
                "description": "Demo large embedding model",
                "size": "Demo",
                "category": "Demo",
                "recommended": False,
                "requires_api_key": False,
                "cost": "Free"
            }
        }

def create_demo_rag_system() -> DemoRAGSystem:
    """Factory function to create demo RAG system"""
    return DemoRAGSystem()