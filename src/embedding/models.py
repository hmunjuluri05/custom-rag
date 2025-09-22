from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Enum for embedding providers"""
    LOCAL = "local"
    OPENAI = "openai"
    GOOGLE = "google"

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

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for SentenceTransformer models"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        try:
            if not texts:
                return np.array([])

            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise Exception(f"Failed to encode texts: {str(e)}")

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        return self.encode([text])[0]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings"""
        try:
            return self.model.similarity(embeddings1, embeddings2).numpy()
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise Exception(f"Failed to calculate similarity: {str(e)}")

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model wrapper"""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        logger.info(f"Initializing OpenAI embedding model: {model_name}")

        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

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

            logger.info(f"Successfully initialized OpenAI model: {model_name}")

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
    """Google embedding model wrapper"""

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        logger.info(f"Initializing Google embedding model: {model_name}")

        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)

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

            logger.info(f"Successfully initialized Google model: {model_name}")

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

    AVAILABLE_MODELS = {
        # External API Models (OpenAI)
        "text-embedding-3-large": {
            "provider": EmbeddingProvider.OPENAI,
            "dimension": 3072,
            "description": "OpenAI's most capable embedding model",
            "size": "API",
            "category": "Premium External",
            "recommended": True,
            "requires_api_key": True,
            "cost": "Paid"
        },
        "text-embedding-3-small": {
            "provider": EmbeddingProvider.OPENAI,
            "dimension": 1536,
            "description": "OpenAI's fast and efficient embedding model",
            "size": "API",
            "category": "External",
            "recommended": True,
            "requires_api_key": True,
            "cost": "Paid"
        },
        "text-embedding-ada-002": {
            "provider": EmbeddingProvider.OPENAI,
            "dimension": 1536,
            "description": "OpenAI's previous generation embedding model (legacy)",
            "size": "API",
            "category": "External Legacy",
            "recommended": False,
            "requires_api_key": True,
            "cost": "Paid"
        },

        # External API Models (Google)
        "models/embedding-001": {
            "provider": EmbeddingProvider.GOOGLE,
            "dimension": 768,
            "description": "Google's general-purpose embedding model",
            "size": "API",
            "category": "External",
            "recommended": True,
            "requires_api_key": True,
            "cost": "Free/Paid"
        },
        "models/text-embedding-004": {
            "provider": EmbeddingProvider.GOOGLE,
            "dimension": 768,
            "description": "Google's latest text embedding model",
            "size": "API",
            "category": "External",
            "recommended": True,
            "requires_api_key": True,
            "cost": "Free/Paid"
        },

        # Local Models (SentenceTransformers)
        "all-MiniLM-L6-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 384,
            "description": "Fast and efficient model, good for general use",
            "size": "Small (~80MB)",
            "category": "General Purpose",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "all-mpnet-base-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "High quality model with better accuracy",
            "size": "Medium (~420MB)",
            "category": "High Quality",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "sentence-transformers/all-MiniLM-L12-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 384,
            "description": "Balanced model between speed and accuracy",
            "size": "Medium (~130MB)",
            "category": "Balanced",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },

        # State-of-the-art Local Models
        "sentence-transformers/all-mpnet-base-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "One of the best performing general-purpose models",
            "size": "Medium (~420MB)",
            "category": "Premium",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "Optimized for question-answering tasks",
            "size": "Medium (~420MB)",
            "category": "Q&A Specialized",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "sentence-transformers/all-roberta-large-v1": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 1024,
            "description": "Large high-performance model (slower but very accurate)",
            "size": "Large (~1.3GB)",
            "category": "High Performance",
            "recommended": False,
            "requires_api_key": False,
            "cost": "Free"
        },

        # Multilingual Local Models
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 384,
            "description": "Multilingual model supporting 50+ languages",
            "size": "Medium (~420MB)",
            "category": "Multilingual",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "High-quality multilingual model",
            "size": "Large (~970MB)",
            "category": "Multilingual Premium",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },

        # Domain-Specific Local Models
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 384,
            "description": "Fast model optimized for question-answering",
            "size": "Small (~80MB)",
            "category": "Q&A Optimized",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },
        "sentence-transformers/msmarco-distilbert-base-v4": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "Optimized for passage retrieval and search",
            "size": "Medium (~250MB)",
            "category": "Search Optimized",
            "recommended": True,
            "requires_api_key": False,
            "cost": "Free"
        },

        # Legacy/Alternative Local Models
        "all-distilroberta-v1": {
            "provider": EmbeddingProvider.LOCAL,
            "dimension": 768,
            "description": "Distilled RoBERTa model with good performance",
            "size": "Medium (~290MB)",
            "category": "Alternative",
            "recommended": False,
            "requires_api_key": False,
            "cost": "Free"
        }
    }

    @classmethod
    def create_model(cls, model_name: str = "all-mpnet-base-v2", api_key: str = None) -> EmbeddingModel:
        """Create an embedding model"""
        if model_name not in cls.AVAILABLE_MODELS:
            logger.warning(f"Model {model_name} not in available models list, attempting to load as local model")
            return SentenceTransformerModel(model_name)

        model_info = cls.AVAILABLE_MODELS[model_name]
        provider = model_info.get("provider", EmbeddingProvider.LOCAL)

        if provider == EmbeddingProvider.LOCAL:
            return SentenceTransformerModel(model_name)
        elif provider == EmbeddingProvider.OPENAI:
            if not api_key:
                raise ValueError(f"API key required for OpenAI model {model_name}")
            return OpenAIEmbeddingModel(model_name, api_key)
        elif provider == EmbeddingProvider.GOOGLE:
            if not api_key:
                raise ValueError(f"API key required for Google model {model_name}")
            return GoogleEmbeddingModel(model_name, api_key)
        else:
            raise ValueError(f"Unknown provider {provider} for model {model_name}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        return cls.AVAILABLE_MODELS

    @classmethod
    def get_recommended_model(cls, use_case: str = "general") -> str:
        """Get recommended model for specific use case"""
        recommendations = {
            "general": "all-mpnet-base-v2",
            "high_quality": "all-mpnet-base-v2",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
            "fast": "all-MiniLM-L6-v2",
            "accurate": "all-mpnet-base-v2"
        }

        return recommendations.get(use_case, "all-mpnet-base-v2")

class EmbeddingService:
    """Service for managing embeddings and similarity search"""

    def __init__(self, model_name: str = "all-mpnet-base-v2", api_key: str = None):
        self.model = EmbeddingModelFactory.create_model(model_name, api_key)
        self.model_name = model_name
        self.api_key = api_key

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

            # Calculate similarities
            similarities = self.model.similarity(
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
            "model_info": EmbeddingModelFactory.AVAILABLE_MODELS.get(
                self.model_name,
                {"description": "Custom model", "size": "Unknown"}
            )
        }

    def change_model(self, new_model_name: str, api_key: str = None) -> bool:
        """Change the embedding model"""
        try:
            old_model = self.model_name
            self.model = EmbeddingModelFactory.create_model(new_model_name, api_key)
            self.model_name = new_model_name
            self.api_key = api_key
            logger.info(f"Changed embedding model from {old_model} to {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to change model to {new_model_name}: {str(e)}")
            return False