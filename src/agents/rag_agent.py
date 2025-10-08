from typing import Any, Dict, List, Optional, Union
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import Field
import asyncio
from .interfaces.agent_system_interface import IAgentSystem

logger = logging.getLogger(__name__)


class RAGSearchTool(BaseTool):
    """Tool for searching the RAG system knowledge base"""

    name: str = "knowledge_search"
    description: str = """
    Search the knowledge base for relevant information to answer questions.
    Use this tool when you need to find specific information from documents.
    Input should be a clear, specific search query.
    """
    rag_system: Any = Field(description="RAG system instance")

    def __init__(self, rag_system, **kwargs):
        super().__init__(rag_system=rag_system, **kwargs)

    def _run(self, query: str) -> str:
        """Search the knowledge base synchronously"""
        try:
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.rag_system.query(query, top_k=5))

                if not results:
                    return "No relevant information found in the knowledge base."

                # Format results for the agent
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. From {result['filename']}: {result['content'][:300]}..."
                    )

                return "\n\n".join(formatted_results)
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error in RAG search tool: {str(e)}")
            # Return a more helpful message that doesn't make the agent think there's a persistent issue
            return "The knowledge base is currently not available, but I can still help answer your question based on my general knowledge. Please feel free to ask your question and I'll do my best to provide a helpful response."

    async def _arun(self, query: str) -> str:
        """Search the knowledge base asynchronously"""
        try:
            results = await self.rag_system.query(query, top_k=5)

            if not results:
                return "No relevant information found in the knowledge base."

            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. From {result['filename']}: {result['content'][:300]}..."
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in RAG search tool: {str(e)}")
            # Return a more helpful message that doesn't make the agent think there's a persistent issue
            return "The knowledge base is currently not available, but I can still help answer your question based on my general knowledge. Please feel free to ask your question and I'll do my best to provide a helpful response."


class DocumentAnalysisTool(BaseTool):
    """Tool for analyzing specific documents in the knowledge base"""

    name: str = "document_analysis"
    description: str = """
    Analyze specific documents or get document details from the knowledge base.
    Use this tool when you need to understand document structure, metadata, or get document summaries.
    Input should specify what kind of analysis or which documents to examine.
    """
    rag_system: Any = Field(description="RAG system instance")

    def __init__(self, rag_system, **kwargs):
        super().__init__(rag_system=rag_system, **kwargs)

    def _run(self, query: str) -> str:
        """Analyze documents synchronously"""
        try:
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Get document list
                documents = loop.run_until_complete(self.rag_system.list_documents())

                if not documents:
                    return "No documents found in the knowledge base."

                # Create summary
                doc_summary = []
                for doc in documents[:10]:  # Limit to first 10 documents
                    doc_summary.append(
                        f"• {doc.get('filename', 'unknown')} ({doc.get('document_type', 'unknown')}) - "
                        f"{doc.get('total_chunks', 0)} chunks, uploaded {doc.get('timestamp', 'unknown')}"
                    )

                summary = f"Knowledge base contains {len(documents)} documents:\n\n"
                summary += "\n".join(doc_summary)

                if len(documents) > 10:
                    summary += f"\n\n... and {len(documents) - 10} more documents"

                return summary
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error in document analysis tool: {str(e)}")
            return f"Error analyzing documents: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Analyze documents asynchronously"""
        try:
            # Get document list
            documents = await self.rag_system.list_documents()

            if not documents:
                return "No documents found in the knowledge base."

            # Create summary
            doc_summary = []
            for doc in documents[:10]:  # Limit to first 10 documents
                doc_summary.append(
                    f"• {doc.get('filename', 'unknown')} ({doc.get('document_type', 'unknown')}) - "
                    f"{doc.get('total_chunks', 0)} chunks, uploaded {doc.get('timestamp', 'unknown')}"
                )

            summary = f"Knowledge base contains {len(documents)} documents:\n\n"
            summary += "\n".join(doc_summary)

            if len(documents) > 10:
                summary += f"\n\n... and {len(documents) - 10} more documents"

            return summary

        except Exception as e:
            logger.error(f"Error in document analysis tool: {str(e)}")
            return f"Error analyzing documents: {str(e)}"


class RAGAgent:
    """Advanced RAG agent with reasoning capabilities"""

    def __init__(self, rag_system, llm_service):
        self.rag_system = rag_system
        self.llm_service = llm_service

        # Initialize tools
        self.tools = [
            RAGSearchTool(rag_system),
            DocumentAnalysisTool(rag_system)
        ]

        # Create ReAct prompt template
        self.prompt = PromptTemplate.from_template("""You are an intelligent research assistant with access to a knowledge base that contains Excel files with columns including 'Profile ID' and 'Diagnostic Statement'.

DATA STRUCTURE UNDERSTANDING:
- Each row contains: Profile ID | Diagnostic Statement [Profile ID=value, Diagnostic Statement=value]
- The data is stored in the format: column1 | column2 [column1=value1, column2=value2]
- Profile ID and Diagnostic Statement in the same row are CORRESPONDING/RELATED values

PRIMARY OBJECTIVE: When user provides text related to a Diagnostic Statement, ALWAYS return the corresponding Profile ID from the SAME ROW.

CRITICAL UNDERSTANDING:
- User will provide text matching or related to values in the "Diagnostic Statement" column
- You MUST find matching rows and extract the "Profile ID" value from the SAME row as the matched Diagnostic Statement
- The Profile ID and Diagnostic Statement are in the SAME row - they are related/corresponding values
- Look for the pattern [Profile ID=XXX, Diagnostic Statement=YYY] to identify corresponding values from the same row

IMPORTANT: You MUST use the knowledge_search tool to search for information before answering questions. Do not assume the knowledge base is empty or unavailable - always try searching first.

STRICT RULES:
1. For EVERY question, your FIRST action MUST be to use knowledge_search tool with a relevant search query
2. Search for content matching or related to the user's query text
3. After receiving search results, identify rows where the Diagnostic Statement matches the user's query
4. For EACH matching row, extract the Profile ID value from that SAME row
5. Look for the structured format [Profile ID=X, Diagnostic Statement=Y] to ensure you're extracting corresponding values from the same row
6. Your Final Answer MUST list all matching Profile IDs with their corresponding Diagnostic Statements
7. NEVER mix Profile IDs and Diagnostic Statements from different rows

MANDATORY OUTPUT FORMAT for Final Answer:
**Profile ID(s):** [list all matching Profile IDs here, comma-separated]

**Matching Details:**
- Profile ID: [value] → Diagnostic Statement: [corresponding value from same row]
[repeat for each match]

If multiple matches exist, list them all with clear separation.
If no matches found, state: "No matching Profile ID found for the given query."

You have access to these tools:

{tools}

Use this EXACT format (follow it precisely):

Question: the input question you must answer
Thought: I need to search the knowledge base for information about [topic]
Action: knowledge_search
Action Input: [your search query here]
Observation: [the search results will appear here]
Thought: Based on the search results, I can extract the profile-id and provide the answer
Final Answer: **Profile ID(s):** [extracted profile-ids]

**Details:**
[Answer to the user's question with relevant information]

CRITICAL:
- The Action must be EXACTLY "knowledge_search" (one of [{tool_names}])
- The Action Input must be a clear search query
- Wait for Observation before Final Answer
- ALWAYS extract Profile ID from the SAME ROW as the matched Diagnostic Statement
- Use the structured format [Profile ID=X, Diagnostic Statement=Y] to identify row relationships
- Do NOT skip steps or assume you already searched

FORMATTING REQUIREMENTS FOR FINAL ANSWER:
- Use TWO line breaks (\\n\\n) between different sections or items
- Use clear bullet points (-) or numbered lists when presenting multiple matches
- Add line breaks after each Profile ID entry for readability
- NEVER write everything as one big paragraph - break it up!
- Make it easy to scan and read

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        logger.info("RAG Agent initialized with knowledge base tools")

    def _handle_parsing_error(self, error) -> str:
        """Custom handler for parsing errors - guides the agent back on track"""
        error_str = str(error)
        logger.warning(f"Agent parsing error occurred: {error_str}")

        # Return a helpful message that guides the agent to use the correct format
        return """Invalid format detected. You MUST follow this exact format:

Thought: I need to search the knowledge base for [topic]
Action: knowledge_search
Action Input: [search query]

Please try again using the EXACT format above. Start with 'Thought:', then 'Action: knowledge_search', then 'Action Input: [your query]'"""

    async def create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and LLM"""
        try:
            # Get the LangChain LLM from our service
            if hasattr(self.llm_service, 'llm_model') and hasattr(self.llm_service.llm_model, 'llm'):
                llm = self.llm_service.llm_model.llm
            else:
                raise ValueError("LLM service does not have compatible LangChain LLM")

            # Create the ReAct agent (works with any LLM, not just OpenAI)
            agent = create_react_agent(llm, self.tools, self.prompt)

            # Create agent executor with custom error handler
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=self._handle_parsing_error,
                max_iterations=8,  # Increased to allow for retries after parsing errors
                return_intermediate_steps=True
            )

            return agent_executor

        except Exception as e:
            logger.error(f"Error creating agent executor: {str(e)}")
            raise

    async def query(self, question: str) -> Dict[str, Any]:
        """Process a query using the RAG agent"""
        try:
            logger.info(f"Starting RAG agent query for: {question}")
            agent_executor = await self.create_agent_executor()
            logger.info("Agent executor created successfully")

            # Execute the agent
            logger.info("Invoking agent executor...")
            result = await agent_executor.ainvoke({"input": question})
            logger.info(f"Agent execution completed. Result keys: {result.keys()}")

            # Extract tools used from intermediate steps if available
            tools_used = []
            knowledge_search_used = False
            if "intermediate_steps" in result:
                logger.info(f"Intermediate steps count: {len(result['intermediate_steps'])}")
                for step in result["intermediate_steps"]:
                    tool_name = step[0].tool
                    tool_input = step[0].tool_input
                    tool_output = step[1]
                    logger.info(f"Tool used: {tool_name}, Input: {tool_input}, Output length: {len(str(tool_output))}")
                    tools_used.append(tool_name)
                    if tool_name == "knowledge_search":
                        knowledge_search_used = True

            # If knowledge_search was NOT used, force a search as fallback
            if not knowledge_search_used:
                logger.warning("Agent did not use knowledge_search tool! Forcing manual search...")
                search_result = await self.tools[0]._arun(question)
                logger.info(f"Forced search result length: {len(search_result)}")

                # If we got actual results, regenerate the answer with this context
                if "No relevant information found" not in search_result:
                    logger.info("Forced search found results! Regenerating answer with context...")
                    enhanced_prompt = f"""Based on the following information from the knowledge base, please answer the question.

Knowledge Base Information:
{search_result}

Question: {question}

Please provide a comprehensive answer based on the information above."""

                    enhanced_answer = await self.llm_service.generate_response("", enhanced_prompt)

                    return {
                        "answer": enhanced_answer,
                        "agent_reasoning": "Agent bypassed search, manually retrieved knowledge base results",
                        "query": question,
                        "tools_used": ["knowledge_search (forced)"],
                        "forced_search": True
                    }

            return {
                "answer": result["output"],
                "agent_reasoning": "Used intelligent reasoning with knowledge base tools",
                "query": question,
                "tools_used": tools_used if tools_used else []
            }

        except Exception as e:
            logger.error(f"Error in RAG agent query: {str(e)}", exc_info=True)
            # Fall back to using the LLM directly to answer with general knowledge
            try:
                logger.info("Agent executor failed, falling back to direct LLM query")

                # Use the LLM service directly to answer the question
                fallback_prompt = f"""The knowledge base is currently unavailable, but please answer the following question using your general knowledge:

{question}

Provide a clear, helpful answer based on your training. If you use general knowledge rather than specific documents, make that clear in your response."""

                llm_response = await self.llm_service.generate_response(fallback_prompt)

                return {
                    "answer": llm_response,
                    "agent_reasoning": "Knowledge base unavailable, answered using general knowledge",
                    "query": question,
                    "tools_used": [],
                    "fallback": True
                }
            except Exception as fallback_error:
                logger.error(f"Fallback LLM query also failed: {str(fallback_error)}")
                return {
                    "answer": f"I apologize, but I'm currently unable to process your question due to technical issues. Please try again later or contact support if the problem persists.",
                    "agent_reasoning": "Both agent and fallback failed",
                    "query": question,
                    "tools_used": [],
                    "fallback": True,
                    "error": str(e)
                }

    async def query_stream(self, question: str):
        """Stream the agent's reasoning and response"""
        try:
            yield {
                "type": "agent_start",
                "content": "🤖 Agent starting analysis..."
            }

            agent_executor = await self.create_agent_executor()

            # Note: Streaming agent execution is more complex in LangChain
            # For now, we'll provide step-by-step updates
            yield {
                "type": "agent_thinking",
                "content": "🔍 Searching knowledge base..."
            }

            result = await agent_executor.ainvoke({"input": question})

            yield {
                "type": "agent_complete",
                "content": result["output"]
            }

        except Exception as e:
            logger.error(f"Error in streaming RAG agent query: {str(e)}")
            yield {
                "type": "error",
                "content": f"Error during agent execution: {str(e)}"
            }

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history"""
        return self.memory.chat_memory.messages

    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Agent conversation memory cleared")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            "agent_type": "RAG Agent",
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "memory_type": type(self.memory).__name__,
            "conversation_turns": len(self.memory.chat_memory.messages),
            "max_iterations": 5,
            "capabilities": [
                "Knowledge base search",
                "Document analysis",
                "Multi-step reasoning",
                "Conversation memory",
                "Source citation"
            ]
        }


class AgenticRAGSystem(IAgentSystem):
    """Agentic RAG system with multi-step reasoning and tool orchestration"""

    def __init__(self, rag_system, llm_service):
        self.rag_system = rag_system
        self.llm_service = llm_service

        # Initialize different specialized agents
        self.general_agent = RAGAgent(rag_system, llm_service)

        # Could add more specialized agents here
        # self.technical_agent = TechnicalRAGAgent(rag_system, llm_service)
        # self.research_agent = ResearchRAGAgent(rag_system, llm_service)

        logger.info("Multi-agent RAG system initialized")

    async def route_query(self, question: str, agent_type: str = "general") -> Dict[str, Any]:
        """Route query to appropriate agent"""
        agents = {
            "general": self.general_agent
        }

        agent = agents.get(agent_type)
        if not agent:
            return {
                "answer": f"Unknown agent type: {agent_type}. Available agents: {list(agents.keys())}",
                "agent_type": agent_type,
                "query": question
            }

        result = await agent.query(question)
        result["agent_type"] = agent_type

        # Normalize response format for chat interface
        # Convert "answer" field to "response" if present
        if "answer" in result and "response" not in result:
            result["response"] = result["answer"]

        # Ensure sources field exists for UI compatibility
        if "sources" not in result:
            result["sources"] = []

        return result

    async def route_query_stream(self, question: str, agent_type: str = "general"):
        """Stream routed query response"""
        agents = {
            "general": self.general_agent
        }

        agent = agents.get(agent_type)
        if not agent:
            yield {
                "type": "error",
                "content": f"Unknown agent type: {agent_type}. Available agents: {list(agents.keys())}"
            }
            return

        async for chunk in agent.query_stream(question):
            chunk["agent_type"] = agent_type
            yield chunk

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the multi-agent system"""
        return {
            "system_type": "Agentic RAG",
            "available_agents": ["general"],
            "total_tools": len(self.general_agent.tools),
            "capabilities": [
                "Intelligent agent routing",
                "Specialized agent workflows",
                "Tool-based reasoning",
                "Multi-step problem solving",
                "Conversation memory across agents"
            ],
            "agent_details": {
                "general": self.general_agent.get_agent_info()
            }
        }

    # Interface implementation methods
    async def query_with_agent(self,
                              query_text: str,
                              agent_type: str = "general",
                              **kwargs) -> Dict[str, Any]:
        """Query using agent with specified type - Interface implementation"""
        try:
            # Use existing route_query method
            result = await self.route_query(query_text, agent_type)
            return result
        except Exception as e:
            logger.error(f"Error in agent query: {str(e)}")
            return {
                "response": f"Agent query failed: {str(e)}",
                "agent_type": agent_type,
                "tools_used": [],
                "agent_reasoning": "Error occurred during processing",
                "error": str(e),
                "fallback": True
            }

    def get_available_agents(self) -> list:
        """Get list of available agent types"""
        return ["general"]

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent system"""
        return self.get_system_info()

    def is_available(self) -> bool:
        """Check if agent system is available and functional"""
        try:
            return self.general_agent is not None and self.llm_service is not None
        except Exception:
            return False