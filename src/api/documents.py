from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from pydantic import BaseModel
from ..embedding.chunking import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)

class ReprocessRequest(BaseModel):
    embedding_model: Optional[str] = None
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    preserve_sentences: Optional[bool] = None
    preserve_paragraphs: Optional[bool] = None

class BulkReprocessRequest(BaseModel):
    document_ids: List[str]
    embedding_model: Optional[str] = None
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    preserve_sentences: Optional[bool] = None
    preserve_paragraphs: Optional[bool] = None

def create_documents_router(rag_system):
    """Create documents API router with RAG system dependency"""
    router = APIRouter()

    @router.get("/documents/")
    async def list_documents():
        """List all processed documents"""
        try:
            documents = await rag_system.list_documents()
            return JSONResponse(content={"documents": documents})

        except Exception as e:
            logger.error(f"Error in list_documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/documents/{doc_id}/chunks")
    async def get_document_chunks(doc_id: str):
        """Get chunks for a specific document"""
        try:
            chunks = await rag_system.get_document_chunks(doc_id)
            return JSONResponse(content={"chunks": chunks})

        except Exception as e:
            logger.error(f"Error getting chunks for document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str):
        """Delete a document from the system"""
        try:
            success = await rag_system.delete_document(doc_id)
            if success:
                return JSONResponse(content={"message": "Document deleted successfully"})
            else:
                raise HTTPException(status_code=404, detail="Document not found")

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/documents/{doc_id}/reprocess")
    async def reprocess_document(doc_id: str, request: ReprocessRequest):
        """Reprocess a document with new settings"""
        try:
            # Get document chunks to extract the original file path
            chunks = await rag_system.get_document_chunks(doc_id)

            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")

            # Get the file path from metadata
            file_path = chunks[0]["metadata"].get("file_path")
            filename = chunks[0]["metadata"].get("filename")

            if not file_path or not filename:
                raise HTTPException(status_code=400, detail="Cannot reprocess: missing file information")

            # Delete old document
            await rag_system.delete_document(doc_id)

            # Create chunking configuration if provided
            chunking_config = None
            if any([request.chunking_strategy, request.chunk_size, request.chunk_overlap]):
                try:
                    strategy = ChunkingStrategy(request.chunking_strategy) if request.chunking_strategy else ChunkingStrategy.WORD_BASED
                except ValueError:
                    strategy = ChunkingStrategy.WORD_BASED

                chunking_config = ChunkingConfig(
                    strategy=strategy,
                    chunk_size=request.chunk_size or 1000,
                    chunk_overlap=request.chunk_overlap or 200,
                    preserve_sentences=request.preserve_sentences if request.preserve_sentences is not None else True,
                    preserve_paragraphs=request.preserve_paragraphs if request.preserve_paragraphs is not None else False
                )

            # Update embedding model if provided
            if request.embedding_model:
                rag_system.change_embedding_model(request.embedding_model)

            # Reprocess the document
            from pathlib import Path
            text_content = rag_system.document_processor.extract_text(Path(file_path))
            new_doc_id = await rag_system.add_document(text_content, file_path, filename, chunking_config)

            return JSONResponse(content={
                "message": "Document reprocessed successfully",
                "old_document_id": doc_id,
                "new_document_id": new_doc_id,
                "chunking_config": {
                    "strategy": chunking_config.strategy.value if chunking_config else "default",
                    "chunk_size": chunking_config.chunk_size if chunking_config else "default",
                    "chunk_overlap": chunking_config.chunk_overlap if chunking_config else "default"
                }
            })

        except Exception as e:
            logger.error(f"Error reprocessing document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/documents/bulk-delete")
    async def bulk_delete_documents(document_ids: List[str]):
        """Delete multiple documents"""
        try:
            results = []

            for doc_id in document_ids:
                try:
                    success = await rag_system.delete_document(doc_id)
                    results.append({"document_id": doc_id, "success": success})
                except Exception as e:
                    results.append({"document_id": doc_id, "success": False, "error": str(e)})

            successful_count = len([r for r in results if r["success"]])

            return JSONResponse(content={
                "message": f"Deleted {successful_count}/{len(document_ids)} documents",
                "results": results
            })

        except Exception as e:
            logger.error(f"Error in bulk delete: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/documents/bulk-reprocess")
    async def bulk_reprocess_documents(request: BulkReprocessRequest):
        """Reprocess multiple documents with new settings"""
        try:
            results = []

            # Create chunking configuration if provided
            chunking_config = None
            if any([request.chunking_strategy, request.chunk_size, request.chunk_overlap]):
                try:
                    strategy = ChunkingStrategy(request.chunking_strategy) if request.chunking_strategy else ChunkingStrategy.WORD_BASED
                except ValueError:
                    strategy = ChunkingStrategy.WORD_BASED

                chunking_config = ChunkingConfig(
                    strategy=strategy,
                    chunk_size=request.chunk_size or 1000,
                    chunk_overlap=request.chunk_overlap or 200,
                    preserve_sentences=request.preserve_sentences if request.preserve_sentences is not None else True,
                    preserve_paragraphs=request.preserve_paragraphs if request.preserve_paragraphs is not None else False
                )

            # Update embedding model if provided
            if request.embedding_model:
                rag_system.change_embedding_model(request.embedding_model)

            for doc_id in request.document_ids:
                try:
                    # Get document info
                    chunks = await rag_system.get_document_chunks(doc_id)

                    if not chunks:
                        results.append({"document_id": doc_id, "success": False, "error": "Document not found"})
                        continue

                    file_path = chunks[0]["metadata"].get("file_path")
                    filename = chunks[0]["metadata"].get("filename")

                    if not file_path or not filename:
                        results.append({"document_id": doc_id, "success": False, "error": "Missing file information"})
                        continue

                    # Delete and reprocess
                    await rag_system.delete_document(doc_id)

                    # Extract text and add with chunking config
                    from pathlib import Path
                    text_content = rag_system.document_processor.extract_text(Path(file_path))
                    new_doc_id = await rag_system.add_document(text_content, file_path, filename, chunking_config)

                    results.append({
                        "document_id": doc_id,
                        "success": True,
                        "new_document_id": new_doc_id
                    })

                except Exception as e:
                    results.append({"document_id": doc_id, "success": False, "error": str(e)})

            successful_count = len([r for r in results if r["success"]])

            return JSONResponse(content={
                "message": f"Reprocessed {successful_count}/{len(request.document_ids)} documents",
                "results": results,
                "chunking_config": {
                    "strategy": chunking_config.strategy.value if chunking_config else "default",
                    "chunk_size": chunking_config.chunk_size if chunking_config else "default",
                    "chunk_overlap": chunking_config.chunk_overlap if chunking_config else "default"
                }
            })

        except Exception as e:
            logger.error(f"Error in bulk reprocess: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router