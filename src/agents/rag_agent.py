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
        self.prompt = PromptTemplate.from_template("""You are an intelligent research assistant with access to a knowledge base of documents.
You can search for information and analyze documents to provide comprehensive answers.

When answering questions:
1. Try to search the knowledge base for relevant information using available tools
2. If the knowledge base is unavailable or contains no relevant information, use your general knowledge to provide helpful answers
3. Analyze the results critically and provide clear, well-sourced answers
4. Always be helpful and respond to the user's question, even if you cannot access the knowledge base
5. If you cite sources from the knowledge base, mention the document names
6. If using general knowledge, make it clear that the information is from your training rather than the knowledge base

Your goal is to be as helpful as possible. Never refuse to answer a question due to technical issues.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        logger.info("RAG Agent initialized with knowledge base tools")

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

            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
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
            if "intermediate_steps" in result:
                tools_used = [step[0].tool for step in result["intermediate_steps"]]

            return {
                "answer": result["output"],
                "agent_reasoning": "Used intelligent reasoning with knowledge base tools",
                "query": question,
                "tools_used": tools_used if tools_used else [tool.name for tool in self.tools]
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
            # Use existing agent_query method
            result = await self.agent_query(query_text, agent_type)
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