from typing import List, Dict, Any, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
import asyncio
import os

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model wrapper with API Gateway support"""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None, base_url: str = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        logger.info(f"Initializing OpenAI embedding model: {model_name}")

        try:
            import openai

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                logger.info(f"Using API Gateway for embeddings: {self.base_url}")
            else:
                logger.info("Using direct OpenAI API for embeddings")

            self.client = openai.OpenAI(**client_kwargs)

            # Model dimensions mapping
            self.model_dimensions = {
                "text-embedding-3-large": 3072,
                "text-embedding-3-small": 1536,
                "text-embedding-ada-002": 1536
            }

            if model_name not in self.model_dimensions:
                logger.warning(f"Unknown OpenAI model {model_name}, defaulting to 1536 dimensions")
                self.dimensions = 1536
            else:
                self.dimensions = self.model_dimensions[model_name]

            logger.info(f"Successfully initialized OpenAI embedding model: {model_name}")

        except ImportError:
            raise Exception("OpenAI library not installed. Please install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise Exception(f"Failed to initialize OpenAI embedding model: {str(e)}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using OpenAI API"""
        try:
            if not texts:
                return np.array([])

            # OpenAI API has a limit on batch size, so we process in batches
            batch_size = 100  # Conservative batch size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings)

        except Exception as e:
            logger.error(f"Error encoding texts with OpenAI: {str(e)}")
            raise Exception(f"Failed to encode texts with OpenAI: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name

class GoogleEmbeddingModel(EmbeddingModel):
    """Google embedding model wrapper with API Gateway support"""

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None, base_url: str = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        logger.info(f"Initializing Google embedding model: {model_name}")

        try:
            import google.generativeai as genai

            if self.base_url:
                # For API Gateway, we might need to configure transport or use custom client
                logger.info(f"Using API Gateway for Google embeddings: {self.base_url}")
                # Note: Google's library might need additional configuration for custom base URLs
                logger.warning("Google embedding API Gateway support is limited. Consider using OpenAI-compatible endpoints.")

            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                logger.info("Using default Google API configuration")

            # Model dimensions mapping for Google models
            self.model_dimensions = {
                "models/embedding-001": 768,
                "models/text-embedding-004": 768
            }

            if model_name not in self.model_dimensions:
                logger.warning(f"Unknown Google model {model_name}, defaulting to 768 dimensions")
                self.dimensions = 768
            else:
                self.dimensions = self.model_dimensions[model_name]

            logger.info(f"Successfully initialized Google embedding model: {model_name}")

        except ImportError:
            raise Exception("Google Generative AI library not installed. Please install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {str(e)}")
            raise Exception(f"Failed to initialize Google embedding model: {str(e)}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using Google API"""
        try:
            if not texts:
                return np.array([])

            import google.generativeai as genai

            # Google API processes one text at a time for embeddings
            embeddings = []

            for text in texts:
                response = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error encoding texts with Google: {str(e)}")
            raise Exception(f"Failed to encode texts with Google: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name

class EmbeddingModelFactory:
    """Factory for creating embedding models"""

    @classmethod
    def create_model(cls, model_name: str = None, api_key: str = None, base_url: str = None) -> EmbeddingModel:
        """Create an embedding model with optional API Gateway support"""
        from ..config.model_config import get_model_config, EmbeddingProvider
        config = get_model_config()

        # Use default model if none specified
        if model_name is None:
            model_name = config.get_default_embedding_model()

        # Get model info from YAML config
        model_info = config.get_embedding_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown embedding model: {model_name}. Available models: {list(config.get_embedding_models().keys())}")

        provider = config.get_embedding_provider(model_name)
        if not provider:
            raise ValueError(f"Invalid provider for model {model_name}")

        if provider.value == "openai":
            return OpenAIEmbeddingModel(model_name, api_key, base_url)
        elif provider.value == "google":
            return GoogleEmbeddingModel(model_name, api_key, base_url)
        else:
            raise ValueError(f"Unsupported provider {provider} for model {model_name}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        from ..config.model_config import get_model_config
        config = get_model_config()
        return config.get_embedding_models()

    @classmethod
    def get_recommended_model(cls, use_case: str = "general") -> str:
        """Get recommended model for specific use case"""
        from ..config.model_config import get_model_config
        config = get_model_config()

        # Find recommended models from config
        models = config.get_embedding_models()
        recommended_models = [name for name, info in models.items() if info.get('recommended', False)]

        if recommended_models:
            return recommended_models[0]  # Return first recommended model

        # Fallback to default
        return config.get_default_embedding_model()

class EmbeddingService:
    """Service for managing embeddings and similarity search"""

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model = EmbeddingModelFactory.create_model(model_name, api_key, base_url)
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts in batches"""
        if not texts:
            return []

        embeddings = []

        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def find_similar_texts(self,
                          query_text: str,
                          candidate_texts: List[str],
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts to query"""
        try:
            # Encode query
            query_embedding = self.model.encode([query_text])[0]

            # Encode candidates
            candidate_embeddings = self.model.encode(candidate_texts)

            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                candidate_embeddings
            )[0]

            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "text": candidate_texts[idx],
                    "similarity_score": float(similarities[idx]),
                    "index": int(idx)
                })

            return results

        except Exception as e:
            logger.error(f"Error finding similar texts: {str(e)}")
            raise Exception(f"Failed to find similar texts: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model.get_model_name(),
            "dimension": self.model.get_dimension(),
            "model_info": self._get_model_info() or
                        {"description": "Custom model", "size": "Unknown"}
        }

    def change_model(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Change the embedding model"""
        try:
            old_model = self.model_name
            self.model = EmbeddingModelFactory.create_model(new_model_name, api_key, base_url)
            self.model_name = new_model_name
            self.api_key = api_key
            self.base_url = base_url
            logger.info(f"Changed embedding model from {old_model} to {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to change model to {new_model_name}: {str(e)}")
            return False

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model info from config"""
        try:
            from ..config.model_config import get_model_config
            config = get_model_config()
            return config.get_embedding_model_info(self.model_name)
        except Exception:
            return None