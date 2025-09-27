from typing import List, Dict, Any, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
import asyncio
import os

logger = logging.getLogger(__name__)


class LangChainEmbeddingModel(ABC):
    """Abstract base class for LangChain embedding models"""

    @abstractmethod
    async def encode(self, texts: List[str]) -> np.ndarray:
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


class LangChainOpenAIEmbeddingModel(LangChainEmbeddingModel):
    """LangChain OpenAI embedding model wrapper with Kong API Gateway support"""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None, base_url: str = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        logger.info(f"Initializing LangChain OpenAI embedding model: {model_name}")

        try:
            from langchain_openai import OpenAIEmbeddings

            # Initialize LangChain OpenAI embeddings with Kong API Gateway
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                # Additional parameters for better performance
                chunk_size=1000,  # Process up to 1000 texts at once
                max_retries=3,
                request_timeout=30
            )

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

            logger.info(f"Successfully initialized LangChain OpenAI embedding model: {model_name} via API Gateway: {self.base_url}")

        except ImportError:
            raise Exception("LangChain OpenAI library not installed. Please install with: uv add langchain-openai")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain OpenAI embeddings: {str(e)}")
            raise Exception(f"Failed to initialize LangChain OpenAI embedding model: {str(e)}")

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using LangChain OpenAI API"""
        try:
            if not texts:
                return np.array([])

            # Use LangChain's async embed_documents method
            embeddings_list = await self.embeddings.aembed_documents(texts)

            # Convert to numpy array
            return np.array(embeddings_list)

        except Exception as e:
            logger.error(f"Error encoding texts with LangChain OpenAI: {str(e)}")
            raise Exception(f"Failed to encode texts with LangChain OpenAI: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name


class LangChainGoogleEmbeddingModel(LangChainEmbeddingModel):
    """LangChain Google embedding model wrapper with Kong API Gateway support"""

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None, base_url: str = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        logger.info(f"Initializing LangChain Google embedding model: {model_name}")

        try:
            # For Kong API Gateway, we'll use OpenAI-compatible embeddings
            # since Kong often provides unified OpenAI-compatible endpoints
            from langchain_openai import OpenAIEmbeddings

            # Use OpenAI embeddings for Google models via Kong Gateway (OpenAI-compatible format)
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                chunk_size=100,  # Google models might have lower rate limits
                max_retries=3,
                request_timeout=30
            )

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

            logger.info(f"Successfully initialized LangChain Google embedding model: {model_name} via Kong API Gateway: {self.base_url}")

        except ImportError:
            raise Exception("LangChain OpenAI library not installed. Please install with: uv add langchain-openai")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Google embeddings: {str(e)}")
            raise Exception(f"Failed to initialize LangChain Google embedding model: {str(e)}")

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using LangChain via Kong Gateway"""
        try:
            if not texts:
                return np.array([])

            # Use LangChain's async embed_documents method
            embeddings_list = await self.embeddings.aembed_documents(texts)

            # Convert to numpy array
            return np.array(embeddings_list)

        except Exception as e:
            logger.error(f"Error encoding texts with LangChain Google: {str(e)}")
            raise Exception(f"Failed to encode texts with LangChain Google: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name


class LangChainEmbeddingModelFactory:
    """Factory for creating LangChain embedding models"""

    @classmethod
    def create_model(cls, model_name: str = None, api_key: str = None, base_url: str = None) -> LangChainEmbeddingModel:
        """Create a LangChain embedding model with Kong API Gateway support"""
        from config.model_config import get_model_config, EmbeddingProvider, get_kong_config, derive_embedding_url
        config = get_model_config()

        # Use default model if none specified
        if model_name is None:
            model_name = config.get_default_embedding_model()

        # Get defaults for api_key and base_url if not provided
        if api_key is None:
            api_key = get_kong_config()

        if base_url is None:
            base_url = derive_embedding_url(model_name)

        # Get model info from YAML config
        model_info = config.get_embedding_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown embedding model: {model_name}. Available models: {list(config.get_embedding_models().keys())}")

        provider = config.get_embedding_provider(model_name)
        if not provider:
            raise ValueError(f"Invalid provider for model {model_name}")

        if provider.value == "openai":
            return LangChainOpenAIEmbeddingModel(model_name, api_key, base_url)
        elif provider.value == "google":
            return LangChainGoogleEmbeddingModel(model_name, api_key, base_url)
        else:
            raise ValueError(f"Unsupported provider {provider} for model {model_name}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        from config.model_config import get_model_config
        config = get_model_config()
        return config.get_embedding_models()

    @classmethod
    def get_recommended_model(cls, use_case: str = "general") -> str:
        """Get recommended model for specific use case"""
        from config.model_config import get_model_config
        config = get_model_config()

        # Find recommended models from config
        models = config.get_embedding_models()
        recommended_models = [name for name, info in models.items() if info.get('recommended', False)]

        if recommended_models:
            return recommended_models[0]  # Return first recommended model

        # Fallback to default
        return config.get_default_embedding_model()


class LangChainEmbeddingService:
    """Service for managing LangChain embeddings and similarity search"""

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model = LangChainEmbeddingModelFactory.create_model(model_name, api_key, base_url)
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    async def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts using LangChain embeddings"""
        if not texts:
            return []

        try:
            # LangChain handles batching internally, so we can pass all texts at once
            # But we'll still respect the batch_size for very large datasets
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self.model.encode(batch)

                # Convert numpy array to list of individual embeddings
                if len(batch_embeddings.shape) == 2:
                    all_embeddings.extend([emb for emb in batch_embeddings])
                else:
                    all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise Exception(f"Failed to encode texts: {str(e)}")

    async def find_similar_texts(self,
                                query_text: str,
                                candidate_texts: List[str],
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts to query using LangChain embeddings"""
        try:
            # Encode query
            query_embeddings = await self.model.encode([query_text])
            query_embedding = query_embeddings[0]

            # Encode candidates
            candidate_embeddings = await self.model.encode(candidate_texts)

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
        """Get information about the current LangChain model"""
        return {
            "model_name": self.model.get_model_name(),
            "dimension": self.model.get_dimension(),
            "framework": "langchain",
            "model_info": self._get_model_info() or
                        {"description": "LangChain embedding model", "size": "Unknown"}
        }

    def change_model(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Change the LangChain embedding model"""
        try:
            old_model = self.model_name
            self.model = LangChainEmbeddingModelFactory.create_model(new_model_name, api_key, base_url)
            self.model_name = new_model_name
            self.api_key = api_key
            self.base_url = base_url
            logger.info(f"Changed LangChain embedding model from {old_model} to {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to change LangChain model to {new_model_name}: {str(e)}")
            return False

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model info from config"""
        try:
            from config.model_config import get_model_config
            config = get_model_config()
            return config.get_embedding_model_info(self.model_name)
        except Exception:
            return None