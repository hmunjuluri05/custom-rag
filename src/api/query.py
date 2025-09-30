from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryMode(str, Enum):
    """Query processing modes"""
    VECTOR_SEARCH = "vector_search"      # Basic vector search without LLM
    LLM_RESPONSE = "llm_response"        # Standard RAG with LLM
    AGENTIC_RAG = "agentic_rag"          # Agentic RAG with multi-step reasoning

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    document_filter: Optional[str] = None
    mode: QueryMode = QueryMode.LLM_RESPONSE  # Default to LLM response
    agent_type: str = "general"  # Agent type for multi-agent mode

    # Backward compatibility
    use_llm: Optional[bool] = None

    def __init__(self, **data):
        # Handle backward compatibility with use_llm parameter
        if data.get('use_llm') is not None:
            if data['use_llm']:
                data['mode'] = QueryMode.LLM_RESPONSE
            else:
                data['mode'] = QueryMode.VECTOR_SEARCH
            data.pop('use_llm', None)
        super().__init__(**data)

def create_query_router(rag_system):
    """Create query API router with RAG system dependency"""
    router = APIRouter()

    @router.post("/query/")
    async def query_documents(request: QueryRequest):
        """
        Query the RAG system with multiple processing modes:
        - vector_search: Basic vector search without LLM
        - llm_response: Standard RAG with LLM response generation
        - agentic_rag: Multi-agent intelligent reasoning with tools
        """
        try:
            if request.mode == QueryMode.VECTOR_SEARCH:
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
                    "mode": request.mode.value,
                    "type": "vector_search"
                })

            elif request.mode == QueryMode.LLM_RESPONSE:
                # Standard RAG with LLM response generation
                response = await rag_system.query_with_llm(
                    query_text=request.query,
                    top_k=request.top_k,
                    document_filter=request.document_filter
                )
                return JSONResponse(content={
                    "query": request.query,
                    "response": response.get("response", ""),
                    "sources": response.get("sources", []),
                    "mode": request.mode.value,
                    "type": "llm_response"
                })

            elif request.mode == QueryMode.AGENTIC_RAG:
                # Agentic RAG with multi-step reasoning
                response = await rag_system.query_with_agent(
                    query_text=request.query,
                    agent_type=request.agent_type
                )

                # Check if agent system is available
                if response.get("fallback"):
                    # Fallback to standard LLM response if agents not available
                    logger.warning("Agent system not available, falling back to LLM response")
                    fallback_response = await rag_system.query_with_llm(
                        query_text=request.query,
                        top_k=request.top_k,
                        document_filter=request.document_filter
                    )
                    return JSONResponse(content={
                        "query": request.query,
                        "response": fallback_response.get("response", ""),
                        "sources": fallback_response.get("sources", []),
                        "mode": "llm_response_fallback",
                        "type": "llm_response",
                        "agent_fallback": True,
                        "agent_error": response.get("response", "Agent system unavailable")
                    })

                return JSONResponse(content={
                    "query": request.query,
                    "response": response.get("response", ""),
                    "agent_type": response.get("agent_type", request.agent_type),
                    "agent_reasoning": response.get("agent_reasoning", ""),
                    "tools_used": response.get("tools_used", []),
                    "mode": request.mode.value,
                    "type": "agent_response"
                })

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid query mode: {request.mode}. Supported modes: {[mode.value for mode in QueryMode]}"
                )

        except Exception as e:
            logger.error(f"Error in query_documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/query/modes")
    async def get_query_modes():
        """Get available query modes and agent types"""
        try:
            # Get agent system info if available
            agent_info = {}
            if hasattr(rag_system, 'agent_system') and rag_system.agent_system:
                agent_info = rag_system.get_agent_info()

            return JSONResponse(content={
                "available_modes": [
                    {
                        "mode": QueryMode.VECTOR_SEARCH.value,
                        "description": "Basic vector search without LLM processing",
                        "use_case": "Fast document retrieval, similarity search"
                    },
                    {
                        "mode": QueryMode.LLM_RESPONSE.value,
                        "description": "Standard RAG with LLM response generation",
                        "use_case": "Intelligent answers based on document context"
                    },
                    {
                        "mode": QueryMode.AGENTIC_RAG.value,
                        "description": "Agentic RAG with multi-step reasoning and specialized tools",
                        "use_case": "Complex queries requiring multi-step reasoning and analysis",
                        "available": bool(hasattr(rag_system, 'agent_system') and rag_system.agent_system),
                        "workflow": "Uses ReAct (Reasoning + Acting) pattern: analyzes question → selects appropriate tool → retrieves information → reasons about results → provides comprehensive answer",
                        "tools": ["Knowledge Search Tool (searches vector database)", "Document Analysis Tool (analyzes document structure and metadata)"]
                    }
                ],
                "agent_system": agent_info,
                "default_mode": QueryMode.LLM_RESPONSE.value,
                "backward_compatibility": {
                    "use_llm": "Still supported for backward compatibility",
                    "use_llm=true": "Maps to llm_response mode",
                    "use_llm=false": "Maps to vector_search mode"
                }
            })

        except Exception as e:
            logger.error(f"Error getting query modes: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router