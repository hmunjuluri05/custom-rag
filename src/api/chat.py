import logging

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat query processing"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    async def process_query(self, query: str) -> str:
        """Process chat query using RAG system"""
        try:
            results = await self.rag_system.query(query, top_k=3)

            if not results:
                return "I don't have any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded some documents first."

            # Format response with sources
            response_parts = []

            # Create a summary response based on the retrieved chunks
            context = "\n\n".join([result["content"] for result in results])

            # For now, return the most relevant chunk with source info
            # In a production system, you'd use an LLM to generate a proper response
            best_result = results[0]

            response_parts.append(f"Based on the document '{best_result['filename']}', here's what I found:")
            response_parts.append(f"\n{best_result['content']}")

            if len(results) > 1:
                response_parts.append(f"\n\nI also found {len(results)-1} other relevant sections in your documents.")

            return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return f"I encountered an error while searching your documents: {str(e)}"