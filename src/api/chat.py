import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat query processing"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    async def process_query(self, query: str, mode: str = "llm_response", **kwargs) -> Dict[str, Any]:
        """Process chat query using RAG system with specified mode"""
        try:
            top_k = kwargs.get('top_k', 3)
            document_filter = kwargs.get('document_filter')
            agent_type = kwargs.get('agent_type', 'general')

            if mode == "vector_search":
                # Basic vector search without LLM
                results = await self.rag_system.query(
                    query_text=query,
                    top_k=top_k,
                    document_filter=document_filter
                )
                return {
                    "response": f"Found {len(results)} relevant documents",
                    "results": results,
                    "sources": [],
                    "query": query,
                    "mode": mode
                }

            elif mode == "agentic_rag":
                # Agentic RAG with multi-step reasoning
                result = await self.rag_system.query_with_agent(
                    query_text=query,
                    agent_type=agent_type
                )
                return result

            else:
                # Default: LLM response mode
                result = await self.rag_system.query_with_llm(
                    query_text=query,
                    top_k=top_k,
                    document_filter=document_filter
                )
                return result

        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query": query,
                "mode": mode
            }

    async def process_query_stream(self, query: str):
        """Process chat query using RAG system with streaming LLM response"""
        try:
            # Use streaming LLM-powered query method
            async for chunk in self.rag_system.query_with_llm_stream(query, top_k=3):
                yield chunk

        except Exception as e:
            logger.error(f"Error processing streaming chat query: {str(e)}")
            yield {
                "type": "error",
                "content": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "query": query
            }

