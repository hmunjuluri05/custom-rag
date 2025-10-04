from typing import List, Dict, Any
import numpy as np
import logging
from abc import abstractmethod
from .interfaces.embedding_model_interface import IEmbeddingModel, IEmbeddingModelFactory

logger = logging.getLogger(__name__)


class EmbeddingModel(IEmbeddingModel):
    """Abstract base class for embedding models"""

    @abstractmethod
    async def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts into embeddings"""
        pass

    async def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """Encode a single text into an embedding"""
        result = await self.encode([text], **kwargs)
        return result[0]

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the model can be reached"""
        pass

    @abstractmethod
    def get_max_batch_size(self) -> int:
        """Get the maximum batch size"""
        pass

    @abstractmethod
    def get_max_input_length(self) -> int:
        """Get the maximum input length"""
        pass

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


class OpenAIEmbeddingModel(EmbeddingModel):
    """Modern OpenAI embedding model wrapper with API Gateway support"""

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name

        # Get configuration from config system
        from ..config.model_config import get_model_config, get_api_config
        config = get_model_config()

        # Get API key and base URL from configuration
        self.api_key = get_api_config()
        self.base_url = config.get_embedding_model_gateway_url(model_name)

        # Validate required configuration
        if not self.api_key:
            raise ValueError("API_KEY is required. Set API_KEY environment variable.")
        if not self.base_url:
            raise ValueError(f"No gateway URL found for embedding model: {model_name}")

        logger.info(f"Initializing Modern OpenAI embedding model: {model_name}")

        try:
            from langchain_openai import OpenAIEmbeddings
            import httpx

            # Create custom HTTP client with API Gateway headers
            # Headers configured via config/models.yaml (e.g., {"api-key": key, "ai-gateway-version": "v2"})
            from ..config.model_config import get_model_config
            config = get_model_config()
            async_client = httpx.AsyncClient(headers=config.get_gateway_headers(self.api_key))

            # Initialize Modern OpenAI embeddings with API Gateway
            # Use "dummy" for api_key since API Gateway handles authentication via custom header
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                api_key="dummy",  # API Gateway handles auth via custom header
                base_url=self.base_url,
                # Additional parameters for better performance
                chunk_size=1000,  # Process up to 1000 texts at once
                max_retries=3,
                request_timeout=30,
                http_async_client=async_client
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

            logger.info(f"Successfully initialized Modern OpenAI embedding model: {model_name} via API Gateway: {self.base_url}")

        except ImportError:
            raise Exception("LangChain OpenAI library not installed. Please install with: uv add langchain-openai")
        except Exception as e:
            logger.error(f"Failed to initialize Modern OpenAI embeddings: {str(e)}")
            raise Exception(f"Failed to initialize Modern OpenAI embedding model: {str(e)}")

    async def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts into embeddings using Modern OpenAI API"""
        try:
            if not texts:
                return np.array([])

            # Use Modern's async embed_documents method
            embeddings_list = await self.embeddings.aembed_documents(texts)

            # Convert to numpy array
            return np.array(embeddings_list)

        except Exception as e:
            logger.error(f"Error encoding texts with OpenAI: {str(e)}")
            raise Exception(f"Failed to encode texts with OpenAI: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "dimension": self.dimensions,
            "max_input_tokens": 8192,
            "description": f"OpenAI {self.model_name} embedding model with API Gateway support",
            "use_cases": ["document_search", "similarity", "clustering", "classification"],
            "framework": "langchain",
            "gateway_url": self.base_url
        }

    def validate_connection(self) -> bool:
        """Validate OpenAI connection"""
        try:
            return bool(self.api_key and self.base_url and self.embeddings)
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size"""
        return 1000  # OpenAI's default chunk_size

    def get_max_input_length(self) -> int:
        """Get the maximum input length in tokens"""
        return 8192  # OpenAI's token limit for embeddings


class GoogleEmbeddingModel(EmbeddingModel):
    """Modern Google embedding model wrapper with API Gateway support"""

    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name

        # Get configuration from config system
        from ..config.model_config import get_model_config, get_api_config
        config = get_model_config()

        # Get API key and base URL from configuration
        self.api_key = get_api_config()
        self.base_url = config.get_embedding_model_gateway_url(model_name)

        # Validate required configuration
        if not self.api_key:
            raise ValueError("API_KEY is required. Set API_KEY environment variable.")
        if not self.base_url:
            raise ValueError(f"No gateway URL found for embedding model: {model_name}")

        logger.info(f"Initializing Modern Google embedding model: {model_name}")

        try:
            # Use Google's native embedding implementation
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            import httpx

            # Create custom HTTP client with API Gateway headers
            # Headers configured via config/models.yaml (e.g., {"api-key": key, "ai-gateway-version": "v2"})
            from ..config.model_config import get_model_config
            config = get_model_config()
            async_client = httpx.AsyncClient(headers=config.get_gateway_headers(self.api_key))

            # Initialize Google embeddings with API Gateway
            # Use "dummy" for google_api_key since API Gateway handles authentication via custom header
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key="dummy",  # API Gateway handles auth via custom header
                task_type="retrieval_document",
                # Note: Google LangChain may not support custom HTTP client
                # If it doesn't work, we'll need to use a different approach
                client=async_client if hasattr(GoogleGenerativeAIEmbeddings, 'client') else None
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

            logger.info(f"Successfully initialized Google embedding model: {model_name}")

        except ImportError:
            raise Exception("LangChain Google GenAI library not installed. Please install with: uv add langchain-google-genai")
        except Exception as e:
            logger.error(f"Failed to initialize Modern Google embeddings: {str(e)}")
            raise Exception(f"Failed to initialize Modern Google embedding model: {str(e)}")

    async def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts into embeddings using Modern via API Gateway"""
        try:
            if not texts:
                return np.array([])

            # Use Modern's async embed_documents method
            embeddings_list = await self.embeddings.aembed_documents(texts)

            # Convert to numpy array
            return np.array(embeddings_list)

        except Exception as e:
            logger.error(f"Error encoding texts with Google: {str(e)}")
            raise Exception(f"Failed to encode texts with Google: {str(e)}")

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        return {
            "provider": "google",
            "model_name": self.model_name,
            "dimension": self.dimensions,
            "max_input_tokens": 2048,
            "description": f"Google {self.model_name} embedding model",
            "use_cases": ["document_search", "similarity", "clustering", "multilingual"],
            "framework": "langchain"
        }

    def validate_connection(self) -> bool:
        """Validate Google connection"""
        try:
            return bool(self.api_key and self.embeddings)
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size"""
        return 100  # Google's batch limit is typically smaller

    def get_max_input_length(self) -> int:
        """Get the maximum input length in tokens"""
        return 2048  # Google's token limit for embeddings


class EmbeddingModelFactory(IEmbeddingModelFactory):
    """Factory for creating Modern embedding models"""

    @classmethod
    def create_model(cls, provider: str = None, model_name: str = None, **config) -> EmbeddingModel:
        """Create a Modern embedding model - models handle their own configuration"""
        from ..config.model_config import get_model_config
        model_config = get_model_config()

        # Use default model if none specified
        if model_name is None:
            model_name = model_config.get_default_embedding_model()

        # Determine provider - use explicit provider if given, otherwise derive from model
        provider_enum = None
        if provider is not None:
            # Use explicitly provided provider
            from ..config.model_config import EmbeddingProvider
            try:
                provider_enum = EmbeddingProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        else:
            # Derive provider from model name
            provider_enum = model_config.get_embedding_provider(model_name)
            if not provider_enum:
                raise ValueError(f"Cannot determine provider for model {model_name}")

        # Get model info from YAML config to validate model exists
        model_info = model_config.get_embedding_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown embedding model: {model_name}. Available models: {list(model_config.get_embedding_models().keys())}")

        # Validate that the model belongs to the determined provider
        model_provider = model_info.get('provider', '').lower()
        if model_provider != provider_enum.value:
            raise ValueError(f"Model {model_name} belongs to provider {model_provider}, not {provider_enum.value}")

        # Create model based on provider - models handle their own config
        if provider_enum.value == "openai":
            return OpenAIEmbeddingModel(model_name)
        elif provider_enum.value == "google":
            return GoogleEmbeddingModel(model_name)
        else:
            raise ValueError(f"Unsupported provider {provider_enum.value} for model {model_name}")

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

    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported providers"""
        return ["openai", "google"]

    @classmethod
    def validate_model_config(cls, provider: str = None, model_name: str = None, **config) -> bool:
        """Validate model configuration without creating the model"""
        try:
            from ..config.model_config import get_model_config
            model_config = get_model_config()

            # Handle backward compatibility - if only model_name provided
            if model_name is None and provider is not None:
                # Old style call - provider is actually model_name
                model_name = provider
                provider = None

            if not model_name:
                return False

            # Check if model exists
            if not model_config.get_embedding_model_info(model_name):
                return False

            # Check required config parameters
            api_key = config.get('api_key')
            base_url = config.get('base_url')

            # API key is required for all models
            if not api_key:
                return False

            # Base URL is required for OpenAI models
            detected_provider = model_config.get_embedding_provider(model_name)
            if detected_provider and detected_provider.value == "openai" and not base_url:
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating model config: {e}")
            return False

    @classmethod
    def get_model_requirements(cls, model_name: str) -> Dict[str, Any]:
        """Get requirements for a specific model"""
        try:
            from ..config.model_config import get_model_config
            config = get_model_config()

            model_info = config.get_embedding_model_info(model_name)
            if not model_info:
                return {"error": f"Unknown model: {model_name}"}

            provider = config.get_embedding_provider(model_name)
            provider_name = provider.value if provider else "unknown"

            requirements = {
                "api_key_required": True,
                "base_url_required": provider_name == "openai",
                "minimum_memory": "512MB",
                "required_packages": []
            }

            if provider_name == "openai":
                requirements["required_packages"] = ["langchain-openai"]
            elif provider_name == "google":
                requirements["required_packages"] = ["langchain-google-genai"]

            return requirements
        except Exception as e:
            logger.error(f"Error getting model requirements: {e}")
            return {"error": str(e)}


class EmbeddingService:
    """Service for managing Modern embeddings and similarity search"""

    def __init__(self, provider: str = None, model_name: str = None):
        # Models handle their own configuration
        from ..config.model_config import get_model_config
        config = get_model_config()

        # Use defaults if not specified
        if model_name is None:
            model_name = config.get_default_embedding_model()
        if provider is None:
            # Derive provider from default embedding model
            provider_enum = config.get_embedding_provider(model_name)
            if provider_enum:
                provider = provider_enum.value
            else:
                provider = 'openai'  # fallback default

        # Validate that we have valid provider and model
        # If provider was specified but model wasn't, we still use the default embedding model
        # If model was specified but provider wasn't, derive provider from model
        if provider and not model_name:
            model_name = config.get_default_embedding_model()
        elif model_name and not provider:
            # Derive provider from model name
            provider_enum = config.get_embedding_provider(model_name)
            if provider_enum:
                provider = provider_enum.value
            else:
                provider = 'openai'  # fallback default

        self.provider = provider
        self.model_name = model_name
        self.model = EmbeddingModelFactory.create_model(provider=provider, model_name=model_name)

    async def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts using Modern embeddings"""
        if not texts:
            return []

        try:
            # Modern handles batching internally, so we can pass all texts at once
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
        """Find most similar texts to query using Modern embeddings"""
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
        """Get information about the current Modern model"""
        return {
            "model_name": self.model.get_model_name(),
            "dimension": self.model.get_dimension(),
            "framework": "langchain",
            "model_info": self._get_model_info() or
                        {"description": "Modern embedding model", "size": "Unknown"}
        }

    def change_model(self, new_model_name: str, api_key: str = None, base_url: str = None) -> bool:
        """Change the Modern embedding model"""
        try:
            old_model = self.model_name
            self.model = EmbeddingModelFactory.create_model(model_name=new_model_name)
            self.model_name = new_model_name
            self.api_key = api_key
            self.base_url = base_url
            logger.info(f"Changed Modern embedding model from {old_model} to {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to change Modern model to {new_model_name}: {str(e)}")
            return False

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model info from config"""
        try:
            from ..config.model_config import get_model_config
            config = get_model_config()
            return config.get_embedding_model_info(self.model_name)
        except Exception:
            return None