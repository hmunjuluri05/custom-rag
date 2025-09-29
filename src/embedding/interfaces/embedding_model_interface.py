"""
Interface for Embedding models - defines the contract for individual embedding implementations.

This separates the embedding model interface from higher-level services,
allowing for better abstraction and testing of the core embedding functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class IEmbeddingModel(ABC):
    """
    Interface for individual embedding model implementations.

    This interface defines the core contract that all embedding models must implement,
    regardless of provider (OpenAI, Google, HuggingFace, etc.).
    """

    @abstractmethod
    async def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional model-specific parameters

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    async def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Text string to encode
            **kwargs: Additional model-specific parameters

        Returns:
            Numpy array embedding with shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension as integer
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name/identifier of this model.

        Returns:
            Model name as string
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Dictionary containing model metadata such as:
            - provider: Model provider (e.g., 'openai', 'google')
            - model_name: Specific model name
            - dimension: Embedding dimension
            - max_input_tokens: Maximum tokens per input
            - description: Human-readable description
            - use_cases: Recommended use cases
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
    def get_max_batch_size(self) -> int:
        """
        Get the maximum number of texts that can be processed in one batch.

        Returns:
            Maximum batch size as integer
        """
        pass

    @abstractmethod
    def get_max_input_length(self) -> int:
        """
        Get the maximum input length (in characters or tokens) for this model.

        Returns:
            Maximum input length as integer
        """
        pass

    @abstractmethod
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (typically cosine similarity)
        """
        pass


class IEmbeddingModelFactory(ABC):
    """
    Interface for embedding model factories.

    This interface defines how embedding models should be created and managed.
    """

    @abstractmethod
    def create_model(self, model_name: str, **config) -> IEmbeddingModel:
        """
        Create an embedding model instance.

        Args:
            model_name: Name of the model to create
            **config: Configuration parameters (api_key, base_url, etc.)

        Returns:
            Configured embedding model instance
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
    def get_recommended_model(self, use_case: str = "general") -> str:
        """
        Get recommended model for a specific use case.

        Args:
            use_case: Use case description (e.g., 'general', 'code', 'multilingual')

        Returns:
            Recommended model name
        """
        pass

    @abstractmethod
    def validate_model_config(self, model_name: str, **config) -> bool:
        """
        Validate model configuration without creating the model.

        Args:
            model_name: Model name
            **config: Configuration parameters

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_model_requirements(self, model_name: str) -> Dict[str, Any]:
        """
        Get requirements for a specific model.

        Args:
            model_name: Model name

        Returns:
            Dictionary containing requirements such as:
            - required_packages: List of required Python packages
            - api_key_required: Whether API key is required
            - base_url_required: Whether base URL is required
            - minimum_memory: Minimum memory requirements
        """
        pass