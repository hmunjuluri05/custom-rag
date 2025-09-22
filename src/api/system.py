from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class EmbeddingModelChange(BaseModel):
    model_name: str
    force_reprocess: bool = False

def create_system_router(rag_system, file_service):
    """Create system/stats API router with dependencies"""
    router = APIRouter()

    @router.get("/stats")
    async def get_system_stats():
        """Get system statistics"""
        try:
            rag_stats = rag_system.get_system_stats()
            upload_stats = file_service.get_upload_stats()

            return JSONResponse(content={
                "rag_system": rag_stats,
                "file_uploads": upload_stats
            })

        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Custom RAG System",
            "version": "2.0.0",
            "components": {
                "file_service": "operational",
                "rag_system": "operational",
                "ui": "operational"
            }
        }

    @router.get("/embedding/models")
    async def get_available_models():
        """Get available embedding models"""
        try:
            from src.embedding.models import EmbeddingModelFactory
            models = EmbeddingModelFactory.get_available_models()
            return JSONResponse(content=models)
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/embedding/current")
    async def get_current_model():
        """Get current embedding model"""
        try:
            current_model = rag_system.get_current_embedding_model()
            return JSONResponse(content={
                "current_model": current_model,
                "model_info": rag_system.get_embedding_model_info()
            })
        except Exception as e:
            logger.error(f"Error getting current model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/embedding/change")
    async def change_embedding_model(model_change: EmbeddingModelChange):
        """Change the global embedding model"""
        try:
            result = await rag_system.change_embedding_model(
                model_change.model_name,
                model_change.force_reprocess
            )
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router