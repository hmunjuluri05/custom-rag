from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import json
from ..embedding.chunking import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)

def create_upload_router(file_service, rag_system):
    """Create upload API router with dependencies"""
    router = APIRouter()

    @router.post("/upload-documents/")
    async def upload_documents(
        files: List[UploadFile] = File(...),
        chunking_strategy: str = Form("word_based"),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(200),
        preserve_sentences: bool = Form(True),
        preserve_paragraphs: bool = Form(False)
    ):
        """Upload and process multiple documents for RAG system"""
        try:
            # Create chunking configuration from form data
            try:
                strategy = ChunkingStrategy(chunking_strategy)
            except ValueError:
                strategy = ChunkingStrategy.WORD_BASED

            chunking_config = ChunkingConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                preserve_sentences=preserve_sentences,
                preserve_paragraphs=preserve_paragraphs
            )

            # Process files using file service
            processed_files = await file_service.process_uploaded_files(files)

            # Add successfully processed files to RAG system
            for file_info in processed_files:
                if file_info["status"] == "success":
                    try:
                        doc_id = await rag_system.add_document(
                            file_info["text_content"],
                            file_info["file_path"],
                            file_info["filename"],
                            custom_chunking_config=chunking_config
                        )
                        file_info["document_id"] = doc_id
                        file_info["status"] = "processed"
                        file_info["chunking_config"] = {
                            "strategy": strategy.value,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        }
                        # Remove text_content from response (too large)
                        del file_info["text_content"]

                    except Exception as e:
                        logger.error(f"Error adding {file_info['filename']} to RAG system: {str(e)}")
                        file_info["status"] = "failed"
                        file_info["error"] = f"Failed to add to RAG system: {str(e)}"

            successful_count = len([f for f in processed_files if f["status"] == "processed"])

            return JSONResponse(content={
                "message": f"Processed {successful_count}/{len(files)} files successfully",
                "files": processed_files,
                "chunking_config": {
                    "strategy": strategy.value,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "preserve_sentences": preserve_sentences,
                    "preserve_paragraphs": preserve_paragraphs
                }
            })

        except Exception as e:
            logger.error(f"Error in upload_documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router