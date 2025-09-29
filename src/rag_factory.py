"""
RAG System Factory with Dependency Injection

This module provides factory classes to create and assemble RAG system components
with proper dependency injection for loose coupling and testability.
"""

from typing import Optional, Dict, Any
import logging

from src.upload.interfaces import IDocumentProcessor
from src.embedding.interfaces import IVectorStore
from src.llm.interfaces import ILLMService
from src.agents.interfaces import IAgentSystem
from src.rag_system import RAGSystem
from src.upload.document_processor import DocumentProcessor
from src.embedding.vector_store import VectorStore
from src.llm.models import LLMService
from src.embedding.chunking import ChunkingConfig, ChunkingStrategy
from src.config import LLMProvider, get_default_llm_config

logger = logging.getLogger(__name__)


class RAGSystemFactory:
    """
    Factory for creating RAG systems with dependency injection.

    This factory handles the assembly of all dependencies and provides
    convenient methods for creating configured RAG systems.
    """

    @staticmethod
    def create_default_rag_system(
        collection_name: str = "documents",
        embedding_model: str = None,
        api_key: str = None,
        base_url: str = None,
        chunking_config: Optional[ChunkingConfig] = None,
        llm_provider: LLMProvider = None,
        llm_model: str = None,
        use_langchain_vectorstore: bool = False,
        include_agents: bool = True
    ) -> RAGSystem:
        """
        Create a RAG system with default dependencies.

        Args:
            collection_name: Name for the document collection
            embedding_model: Embedding model to use
            api_key: API key for services
            base_url: Base URL for API gateway
            chunking_config: Configuration for text chunking
            llm_provider: LLM provider (OpenAI, Google, etc.)
            llm_model: Specific LLM model name
            use_langchain_vectorstore: Whether to use LangChain vector store
            include_agents: Whether to include agent system

        Returns:
            Configured RAG system with injected dependencies
        """
        # Create document processor
        document_processor = RAGSystemFactory.create_document_processor()

        # Create vector store
        vector_store = RAGSystemFactory.create_vector_store(
            collection_name=collection_name,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            use_langchain_vectorstore=use_langchain_vectorstore
        )

        # Create LLM service
        llm_service = RAGSystemFactory.create_llm_service(
            provider=llm_provider,
            model_name=llm_model,
            api_key=api_key,
            base_url=base_url
        )

        # Create agent system if requested
        agent_system = None
        if include_agents:
            agent_system = RAGSystemFactory.create_agent_system(
                rag_system=None,  # Will be set after RAG system creation
                llm_service=llm_service
            )

        # Create chunking config
        if chunking_config is None:
            chunking_config = ChunkingConfig()

        # Create RAG system with injected dependencies
        rag_system = RAGSystem(
            document_processor=document_processor,
            vector_store=vector_store,
            llm_service=llm_service,
            chunking_config=chunking_config,
            agent_system=agent_system
        )

        # Set RAG system reference in agent system if it exists
        if agent_system:
            agent_system.rag_system = rag_system

        logger.info("RAG system created with dependency injection")
        return rag_system

    @staticmethod
    def create_document_processor() -> IDocumentProcessor:
        """Create document processor instance."""
        return DocumentProcessor()

    @staticmethod
    def create_vector_store(
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = None,
        api_key: str = None,
        base_url: str = None,
        use_langchain_vectorstore: bool = False
    ) -> IVectorStore:
        """Create vector store instance."""
        return VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            use_langchain_vectorstore=use_langchain_vectorstore
        )

    @staticmethod
    def create_llm_service(
        provider: LLMProvider = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None
    ) -> ILLMService:
        """Create LLM service instance."""
        # Get defaults if not provided
        if provider is None:
            default_provider, default_model, default_api_key, default_base_url = get_default_llm_config()
            provider = provider or default_provider
            model_name = model_name or default_model
            api_key = api_key or default_api_key
            base_url = base_url or default_base_url

        return LLMService(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

    @staticmethod
    def create_agent_system(
        rag_system,
        llm_service: ILLMService
    ) -> Optional[IAgentSystem]:
        """Create agent system instance."""
        try:
            from src.agents import AgenticRAGSystem
            return AgenticRAGSystem(rag_system, llm_service)
        except ImportError as e:
            logger.warning(f"Could not create agent system: {e}")
            return None

    @staticmethod
    def create_custom_rag_system(
        document_processor: IDocumentProcessor,
        vector_store: IVectorStore,
        llm_service: ILLMService,
        chunking_config: Optional[ChunkingConfig] = None,
        agent_system: Optional[IAgentSystem] = None
    ) -> RAGSystem:
        """
        Create RAG system with custom dependencies.

        This method allows complete control over dependency injection
        for testing and advanced use cases.

        Args:
            document_processor: Custom document processor
            vector_store: Custom vector store
            llm_service: Custom LLM service
            chunking_config: Custom chunking configuration
            agent_system: Custom agent system

        Returns:
            RAG system with custom injected dependencies
        """
        return RAGSystem(
            document_processor=document_processor,
            vector_store=vector_store,
            llm_service=llm_service,
            chunking_config=chunking_config or ChunkingConfig(),
            agent_system=agent_system
        )


class RAGSystemBuilder:
    """
    Builder pattern for RAG system creation.

    Provides a fluent interface for configuring and building RAG systems
    with dependency injection.
    """

    def __init__(self):
        self._document_processor: Optional[IDocumentProcessor] = None
        self._vector_store: Optional[IVectorStore] = None
        self._llm_service: Optional[ILLMService] = None
        self._chunking_config: Optional[ChunkingConfig] = None
        self._agent_system: Optional[IAgentSystem] = None
        self._config: Dict[str, Any] = {}

    def with_document_processor(self, processor: IDocumentProcessor) -> 'RAGSystemBuilder':
        """Set custom document processor."""
        self._document_processor = processor
        return self

    def with_vector_store(self, vector_store: IVectorStore) -> 'RAGSystemBuilder':
        """Set custom vector store."""
        self._vector_store = vector_store
        return self

    def with_llm_service(self, llm_service: ILLMService) -> 'RAGSystemBuilder':
        """Set custom LLM service."""
        self._llm_service = llm_service
        return self

    def with_chunking_config(self, config: ChunkingConfig) -> 'RAGSystemBuilder':
        """Set chunking configuration."""
        self._chunking_config = config
        return self

    def with_agent_system(self, agent_system: IAgentSystem) -> 'RAGSystemBuilder':
        """Set custom agent system."""
        self._agent_system = agent_system
        return self

    def with_config(self, **kwargs) -> 'RAGSystemBuilder':
        """Set configuration parameters."""
        self._config.update(kwargs)
        return self

    def build(self) -> RAGSystem:
        """Build the RAG system with configured dependencies."""
        # Create missing dependencies with defaults
        if self._document_processor is None:
            self._document_processor = RAGSystemFactory.create_document_processor()

        if self._vector_store is None:
            self._vector_store = RAGSystemFactory.create_vector_store(
                **{k: v for k, v in self._config.items()
                   if k in ['collection_name', 'persist_directory', 'embedding_model',
                           'api_key', 'base_url', 'use_langchain_vectorstore']}
            )

        if self._llm_service is None:
            self._llm_service = RAGSystemFactory.create_llm_service(
                **{k: v for k, v in self._config.items()
                   if k in ['provider', 'model_name', 'api_key', 'base_url']}
            )

        if self._chunking_config is None:
            self._chunking_config = ChunkingConfig()

        # Create agent system if not set and not explicitly disabled
        if self._agent_system is None and self._config.get('include_agents', True):
            self._agent_system = RAGSystemFactory.create_agent_system(
                rag_system=None,  # Will be set after creation
                llm_service=self._llm_service
            )

        # Build RAG system
        rag_system = RAGSystemFactory.create_custom_rag_system(
            document_processor=self._document_processor,
            vector_store=self._vector_store,
            llm_service=self._llm_service,
            chunking_config=self._chunking_config,
            agent_system=self._agent_system
        )

        # Set RAG system reference in agent system
        if self._agent_system:
            self._agent_system.rag_system = rag_system

        return rag_system


def create_rag_system(**kwargs) -> 'RAGSystem':
    """
    Factory function to create a RAG system with dependency injection.

    This is a convenient wrapper around RAGSystemFactory.create_default_rag_system()
    that provides the same interface as the original create_rag_system function.

    Args:
        **kwargs: Configuration parameters for the RAG system

    Returns:
        RAG system with injected dependencies
    """
    return RAGSystemFactory.create_default_rag_system(**kwargs)