from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def create_query_router(rag_system):
    """Create query API router with RAG system dependency"""
    router = APIRouter()

    @router.post("/query/")
    async def query_documents(query: str, top_k: int = 5, document_id: Optional[str] = None):
        """Query the RAG system"""
        try:
            results = await rag_system.query(
                query_text=query,
                top_k=top_k,
                document_filter=document_id
            )

            return JSONResponse(content={
                "query": query,
                "results": results,
                "total_results": len(results)
            })

        except Exception as e:
            logger.error(f"Error in query_documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router