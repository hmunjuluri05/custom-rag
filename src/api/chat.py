import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat query processing"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process chat query using RAG system with LLM"""
        try:
            # Use the new LLM-powered query method which returns response with sources
            result = await self.rag_system.query_with_llm(query, top_k=3)
            return result

        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query": query
            }

