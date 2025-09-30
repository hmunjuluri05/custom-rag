import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

def format_response_text(text: str) -> str:
    """Format response text for better readability"""
    if not text:
        return text

    # Split into sentences and paragraphs for better readability
    formatted_text = text.strip()

    # Add line breaks after periods followed by capital letters (new sentences)
    # But be careful with numbered lists
    formatted_text = re.sub(r'(\.)(\s+)([A-Z][^0-9])', r'\1\n\n\3', formatted_text)

    # Add line breaks after colons followed by capital letters (lists/explanations)
    formatted_text = re.sub(r'(:)(\s+)([A-Z])', r'\1\n\2\3', formatted_text)

    # Format numbered/bulleted lists - add line breaks before numbers
    formatted_text = re.sub(r'(\.)(\s+)(\d+\.)', r'\1\n\n\3', formatted_text)
    formatted_text = re.sub(r'(\w)(\s+)(\d+\.)', r'\1\n\n\3', formatted_text)
    formatted_text = re.sub(r'(\w)(\s+)(- )', r'\1\n\n\3', formatted_text)

    # Clean up multiple newlines
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)

    return formatted_text.strip()

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

                # Format response with document details
                if results:
                    response_parts = [f"Found {len(results)} relevant documents:\n"]
                    for i, result in enumerate(results, 1):
                        filename = result.get("filename", "Unknown")
                        similarity = result.get("similarity_score", 0)
                        content_preview = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")

                        response_parts.append(f"\n**{i}. {filename}** (Similarity: {similarity:.2f})")
                        response_parts.append(f"{content_preview}\n")

                    response_text = "\n".join(response_parts)
                else:
                    response_text = "No relevant documents found for your query."

                return {
                    "response": response_text,
                    "results": results,
                    "sources": [{"filename": r.get("filename", "Unknown"), "relevance_score": r.get("relevance_score", 0)} for r in results],
                    "query": query,
                    "mode": mode
                }

            elif mode == "agentic_rag":
                # Agentic RAG with multi-step reasoning
                result = await self.rag_system.query_with_agent(
                    query_text=query,
                    agent_type=agent_type
                )

                # Format the response for better readability
                if "response" in result:
                    result["response"] = format_response_text(result["response"])

                return result

            else:
                # Default: LLM response mode
                result = await self.rag_system.query_with_llm(
                    query_text=query,
                    top_k=top_k,
                    document_filter=document_filter
                )

                # Format the response for better readability
                if "response" in result:
                    result["response"] = format_response_text(result["response"])

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

