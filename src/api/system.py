from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from enum import Enum
from src.llm.models import LLMProvider

logger = logging.getLogger(__name__)

class EmbeddingModelChange(BaseModel):
    model_name: str
    api_key: str = None
    force_reprocess: bool = False

class LLMModelChange(BaseModel):
    provider: str
    model_name: str = None
    api_key: str = None

def create_system_router(rag_system, file_service):
    """Create system/stats API router with dependencies"""
    router = APIRouter(prefix="/system")

    @router.get("/stats")
    async def get_system_stats():
        """Get system statistics"""
        try:
            rag_stats = rag_system.get_system_stats()
            upload_stats = file_service.get_upload_stats()

            # Handle serialization of enum values in nested structure
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

            # Serialize rag_stats to handle enum values
            serializable_rag_stats = serialize_model_info(rag_stats)

            return JSONResponse(content={
                "rag_system": serializable_rag_stats,
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

    @router.get("/test")
    async def test_endpoint():
        """Test endpoint to verify route registration"""
        return {"message": "Test endpoint working"}

    @router.get("/embedding/models")
    async def get_available_models():
        """Get available embedding models"""
        try:
            from src.embedding.models import EmbeddingModelFactory
            models = EmbeddingModelFactory.get_available_models()
            # Convert enum values to strings for JSON serialization
            serializable_models = {}
            for model_name, model_info in models.items():
                serializable_info = model_info.copy()
                serializable_info['provider'] = model_info['provider'].value
                serializable_models[model_name] = serializable_info
            return JSONResponse(content=serializable_models)
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/embedding/current")
    async def get_current_model():
        """Get current embedding model"""
        try:
            current_model = rag_system.get_current_embedding_model()
            model_info = rag_system.get_embedding_model_info()

            # Handle serialization of enum values in nested structure
            def serialize_model_info(info):
                if isinstance(info, dict):
                    serialized = {}
                    for key, value in info.items():
                        if hasattr(value, 'value'):  # This is an enum
                            serialized[key] = value.value
                        elif isinstance(value, dict):
                            serialized[key] = serialize_model_info(value)  # Recursive for nested dicts
                        else:
                            serialized[key] = value
                    return serialized
                return info

            serializable_info = serialize_model_info(model_info)

            return JSONResponse(content={
                "current_model": current_model,
                "model_info": serializable_info
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
                model_change.api_key,
                model_change.force_reprocess
            )
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Error changing embedding model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/llm/models")
    async def get_available_llm_models():
        """Get available LLM models"""
        try:
            models = rag_system.get_available_llms()
            # Convert enum keys to strings for JSON serialization
            serializable_models = {provider.value: info for provider, info in models.items()}
            return JSONResponse(content=serializable_models)
        except Exception as e:
            logger.error(f"Error getting available LLM models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/llm/debug")
    async def debug_llm():
        """Debug endpoint to test LLM integration"""
        return {"status": "LLM debug endpoint working", "routes_registered": True}

    @router.get("/llm/current")
    async def get_current_llm():
        """Get current LLM model information"""
        try:
            llm_info = rag_system.get_llm_info()
            return JSONResponse(content=llm_info)
        except Exception as e:
            logger.error(f"Error getting current LLM: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/llm/change")
    async def change_llm_model(model_change: LLMModelChange):
        """Change the LLM provider and model"""
        try:
            # Convert string provider to enum
            try:
                provider = LLMProvider(model_change.provider.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid provider: {model_change.provider}")

            success = rag_system.change_llm(
                provider=provider,
                model_name=model_change.model_name,
                api_key=model_change.api_key
            )

            if success:
                return JSONResponse(content={
                    "success": True,
                    "message": f"Successfully changed LLM to {provider.value}: {model_change.model_name}",
                    "new_llm": rag_system.get_llm_info()
                })
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": "Failed to change LLM model"
                    }
                )
        except Exception as e:
            logger.error(f"Error changing LLM model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router