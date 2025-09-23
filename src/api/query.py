from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    document_filter: Optional[str] = None
    use_llm: bool = True

def create_query_router(rag_system):
    """Create query API router with RAG system dependency"""
    router = APIRouter()

    @router.post("/query/")
    async def query_documents(request: QueryRequest):
        """Query the RAG system with optional LLM response generation"""
        try:
            if request.use_llm:
                # Use LLM for intelligent response generation
                response = await rag_system.query_with_llm(
                    query_text=request.query,
                    top_k=request.top_k,
                    document_filter=request.document_filter
                )
                return JSONResponse(content={
                    "query": request.query,
                    "response": response.get("response", ""),
                    "sources": response.get("sources", []),
                    "type": "llm_response"
                })
            else:
                # Basic vector search without LLM
                results = await rag_system.query(
                    query_text=request.query,
                    top_k=request.top_k,
                    document_filter=request.document_filter
                )
                return JSONResponse(content={
                    "query": request.query,
                    "results": results,
                    "total_results": len(results),
                    "type": "vector_search"
                })

        except Exception as e:
            logger.error(f"Error in query_documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router