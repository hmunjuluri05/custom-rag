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

PRIMARY OBJECTIVE: ALWAYS extract and return the profile-id(s) from the search results, regardless of what the user asks for.

CRITICAL RULES:
1. The user will ask general questions (e.g., "who is John?", "tell me about employee 123", "find manager info")
2. They will NOT explicitly ask for "profile-id" - but you MUST extract it anyway
3. EVERY response MUST start with the profile-id(s) from the matching records
4. Even if they only ask for a name, role, or any other field - ALWAYS include the profile-id

INSTRUCTIONS:
1. Carefully analyze the provided context which contains structured data in the format [column_name=value, ...]
2. Find the 'profile-id' field in EVERY matching record
3. Extract ALL profile-id values from the search results
4. Present the profile-id(s) prominently at the start of your response

MANDATORY OUTPUT FORMAT:
**Profile ID(s):** [list all matching profile-ids here]

**Details:**
[Then provide the relevant information the user asked for, along with other useful context from the matched records]

If no profile-id field exists in the results, state: "No Profile ID found in the matched records."

CRITICAL FORMATTING:
- ALWAYS start with the Profile ID(s) line first
- Use clear formatting with line breaks
- Answer the user's actual question in the Details section
- If multiple matches exist, list all profile-ids
- Never skip the profile-id extraction step"""

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

