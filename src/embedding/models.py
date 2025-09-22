from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod

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

class EmbeddingModelFactory:
    """Factory for creating embedding models"""

    AVAILABLE_MODELS = {
        # Most Popular and Recommended Models
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Fast and efficient model, good for general use",
            "size": "Small (~80MB)",
            "category": "General Purpose",
            "recommended": True
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "description": "High quality model with better accuracy",
            "size": "Medium (~420MB)",
            "category": "High Quality",
            "recommended": True
        },
        "sentence-transformers/all-MiniLM-L12-v2": {
            "dimension": 384,
            "description": "Balanced model between speed and accuracy",
            "size": "Medium (~130MB)",
            "category": "Balanced",
            "recommended": True
        },

        # State-of-the-art Models
        "sentence-transformers/all-mpnet-base-v2": {
            "dimension": 768,
            "description": "One of the best performing general-purpose models",
            "size": "Medium (~420MB)",
            "category": "Premium",
            "recommended": True
        },
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
            "dimension": 768,
            "description": "Optimized for question-answering tasks",
            "size": "Medium (~420MB)",
            "category": "Q&A Specialized",
            "recommended": True
        },
        "sentence-transformers/all-roberta-large-v1": {
            "dimension": 1024,
            "description": "Large high-performance model (slower but very accurate)",
            "size": "Large (~1.3GB)",
            "category": "High Performance",
            "recommended": False
        },

        # Multilingual Models
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dimension": 384,
            "description": "Multilingual model supporting 50+ languages",
            "size": "Medium (~420MB)",
            "category": "Multilingual",
            "recommended": True
        },
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
            "dimension": 768,
            "description": "High-quality multilingual model",
            "size": "Large (~970MB)",
            "category": "Multilingual Premium",
            "recommended": True
        },

        # Domain-Specific Models
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
            "dimension": 384,
            "description": "Fast model optimized for question-answering",
            "size": "Small (~80MB)",
            "category": "Q&A Optimized",
            "recommended": True
        },
        "sentence-transformers/msmarco-distilbert-base-v4": {
            "dimension": 768,
            "description": "Optimized for passage retrieval and search",
            "size": "Medium (~250MB)",
            "category": "Search Optimized",
            "recommended": True
        },

        # Legacy/Alternative Models
        "all-distilroberta-v1": {
            "dimension": 768,
            "description": "Distilled RoBERTa model with good performance",
            "size": "Medium (~290MB)",
            "category": "Alternative",
            "recommended": False
        }
    }

    @classmethod
    def create_model(cls, model_name: str = "all-mpnet-base-v2") -> EmbeddingModel:
        """Create an embedding model"""
        if model_name not in cls.AVAILABLE_MODELS:
            logger.warning(f"Model {model_name} not in available models list, but attempting to load anyway")

        return SentenceTransformerModel(model_name)

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

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = EmbeddingModelFactory.create_model(model_name)
        self.model_name = model_name

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

    def change_model(self, new_model_name: str) -> bool:
        """Change the embedding model"""
        try:
            old_model = self.model_name
            self.model = EmbeddingModelFactory.create_model(new_model_name)
            self.model_name = new_model_name
            logger.info(f"Changed embedding model from {old_model} to {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to change model to {new_model_name}: {str(e)}")
            return False