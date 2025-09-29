"""
Interface for LLM models - defines the contract for individual LLM implementations.

This separates the LLM model interface from the LLM service interface,
allowing for better abstraction and testing of the core LLM functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional


class ILLMModel(ABC):
    """
    Interface for individual LLM model implementations.

    This interface defines the core contract that all LLM models must implement,
    regardless of provider (OpenAI, Google, etc.).
    """

    @abstractmethod
    async def generate_response(self, context: str, query: str, **kwargs) -> str:
        """
        Generate a response based on context and query.

        Args:
            context: The context information (e.g., from retrieved documents)
            query: The user's question or prompt
            **kwargs: Additional model-specific parameters

        Returns:
            Generated response as a string
        """
        pass

    @abstractmethod
    async def generate_response_stream(self, context: str, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for real-time interaction.

        Args:
            context: The context information
            query: The user's question or prompt
            **kwargs: Additional model-specific parameters

        Yields:
            Response chunks as strings
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model metadata such as:
            - provider: Model provider (e.g., 'openai', 'google')
            - model_name: Specific model name
            - description: Human-readable description
            - capabilities: List of supported features
            - cost_info: Pricing or usage information
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate that the model can be reached and is properly configured.

        Returns:
            True if connection is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_token_limit(self) -> Optional[int]:
        """
        Get the token limit for this model.

        Returns:
            Maximum number of tokens the model can process, or None if unknown
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.

        Args:
            text: Text to analyze

        Returns:
            Estimated token count
        """
        pass


class ILLMModelFactory(ABC):
    """
    Interface for LLM model factories.

    This interface defines how LLM models should be created and managed.
    """

    @abstractmethod
    def create_model(self, provider: str, model_name: str, **config) -> ILLMModel:
        """
        Create an LLM model instance.

        Args:
            provider: Provider name (e.g., 'openai', 'google')
            model_name: Specific model to create
            **config: Configuration parameters (api_key, base_url, etc.)

        Returns:
            Configured LLM model instance
        """
        pass

    @abstractmethod
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models.

        Returns:
            Dictionary mapping model names to their information
        """
        pass

    @abstractmethod
    def get_supported_providers(self) -> list:
        """
        Get list of supported providers.

        Returns:
            List of provider names
        """
        pass

    @abstractmethod
    def validate_model_config(self, provider: str, model_name: str, **config) -> bool:
        """
        Validate model configuration without creating the model.

        Args:
            provider: Provider name
            model_name: Model name
            **config: Configuration parameters

        Returns:
            True if configuration is valid, False otherwise
        """
        pass