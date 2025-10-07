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
                return result

            else:
                # Default: LLM response mode with custom system prompt for profile-id extraction
                profile_extraction_prompt = """You are an intelligent assistant analyzing structured Excel data from a knowledge base containing multiple sheets and columns, including a 'profile-id' column.

PRIMARY OBJECTIVE: Extract and return the matching profile-id(s) from the search results.

INSTRUCTIONS:
1. Carefully analyze the provided context which contains structured data in the format [column_name=value, ...]
2. Search for the 'profile-id' field in the context
3. Extract ALL matching profile-id values that relate to the user's query
4. Present the profile-id(s) prominently in your response

OUTPUT FORMAT:
**Profile ID(s):** [list all matching profile-ids here]

Then provide relevant details from the matching records.

If no profile-id is found in the results, state: "No Profile ID found for this query."

CRITICAL FORMATTING:
- Start with the Profile ID(s) line
- Use clear formatting with line breaks
- Include relevant context from the matched records
- If multiple matches exist, list all profile-ids"""

                result = await self.rag_system.query_with_llm(
                    query_text=query,
                    top_k=top_k,
                    document_filter=document_filter,
                    system_prompt=profile_extraction_prompt
                )
                result["mode"] = mode
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

