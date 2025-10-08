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
                profile_extraction_prompt = """You are an intelligent assistant analyzing structured Excel data with columns including 'Profile ID' and 'Diagnostic Statement'.

DATA STRUCTURE:
- Each row contains: Profile ID | Diagnostic Statement [Profile ID=value, Diagnostic Statement=value]
- The data is stored in the format: column1 | column2 [column1=value1, column2=value2]

PRIMARY OBJECTIVE: When user provides text related to a Diagnostic Statement, ALWAYS return the corresponding Profile ID from the SAME ROW.

CRITICAL RULES:
1. User will provide text matching or related to values in the "Diagnostic Statement" column
2. You MUST find the matching row(s) and extract the "Profile ID" value from the SAME row
3. The Profile ID and Diagnostic Statement are in the SAME row - they are related/corresponding values
4. ALWAYS return the Profile ID(s) that correspond to the matched Diagnostic Statement(s)

STEP-BY-STEP PROCESS:
1. Analyze the user's query text
2. Find matching rows where the Diagnostic Statement contains or relates to the user's query
3. For EACH matching row, extract the Profile ID value from that SAME row
4. Look for the pattern [Profile ID=XXX, Diagnostic Statement=YYY] to identify corresponding values
5. Return ALL matching Profile IDs with their corresponding Diagnostic Statements

MANDATORY OUTPUT FORMAT:
**Profile ID(s):** [list all matching Profile IDs here, comma-separated]

**Matching Details:**
- Profile ID: [value] → Diagnostic Statement: [corresponding value from same row]
[repeat for each match]

If multiple matches exist, list them all with clear separation.
If no matches found, state: "No matching Profile ID found for the given query."

CRITICAL REQUIREMENTS:
- Extract Profile ID from the SAME ROW as the matched Diagnostic Statement
- Use the structured format [Profile ID=X, Diagnostic Statement=Y] to identify row relationships
- Never mix Profile IDs and Diagnostic Statements from different rows
- Always show the correspondence between Profile ID and its Diagnostic Statement"""

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

