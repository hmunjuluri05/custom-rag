from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers"""
    NONE = "none"  # No external LLM, simple concatenation
    OPENAI = "openai"
    GOOGLE = "google"

class LLMModel(ABC):
    """Abstract base class for LLM models"""

    @abstractmethod
    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using context and query"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

class NoLLMModel(LLMModel):
    """Simple context concatenation without external LLM"""

    def __init__(self):
        self.model_name = "Simple Context Response"

    async def generate_response(self, context: str, query: str) -> str:
        """Simple response generation without external LLM"""
        if not context.strip():
            return "I don't have any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded some documents first."

        # Simple template-based response
        return f"Based on the uploaded documents:\n\n{context}\n\nThis information relates to your question: '{query}'"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "none",
            "model_name": self.model_name,
            "description": "Simple context-based responses without external LLM",
            "cost": "Free"
        }

class OpenAIModel(LLMModel):
    """OpenAI GPT model integration"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

        if api_key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model: {model_name}")
            except ImportError:
                logger.error("OpenAI library not installed. Run: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using OpenAI GPT"""
        if not self._client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")

        try:
            system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
            Use only the information from the context to answer questions. If the context doesn't contain relevant information,
            say so clearly. Be concise and accurate."""

            user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error generating response with OpenAI: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "description": f"OpenAI {self.model_name} with intelligent response generation",
            "cost": "Paid API"
        }

class GoogleModel(LLMModel):
    """Google Gemini model integration"""

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._model = None

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name)
                logger.info(f"Initialized Google Gemini client with model: {model_name}")
            except ImportError:
                logger.error("Google Generative AI library not installed. Run: pip install google-generativeai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {str(e)}")
                raise

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using Google Gemini"""
        if not self._model:
            raise ValueError("Google Gemini client not initialized. Please provide API key.")

        try:
            prompt = f"""You are a helpful assistant that answers questions based on provided context from documents.
            Use only the information from the context to answer questions. If the context doesn't contain relevant information,
            say so clearly. Be concise and accurate.

Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

            # Google Gemini doesn't have built-in async, so we'll use sync and wrap it
            import asyncio

            def generate_sync():
                response = self._model.generate_content(prompt)
                return response.text

            response_text = await asyncio.get_event_loop().run_in_executor(None, generate_sync)
            return response_text.strip()

        except Exception as e:
            logger.error(f"Google Gemini API error: {str(e)}")
            return f"Error generating response with Google Gemini: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "google",
            "model_name": self.model_name,
            "description": f"Google {self.model_name} with intelligent response generation",
            "cost": "Paid API"
        }

class LLMFactory:
    """Factory for creating LLM models"""

    AVAILABLE_MODELS = {
        LLMProvider.NONE: {
            "models": ["simple-context"],
            "description": "Simple context-based responses without external LLM",
            "requires_api_key": False
        },
        LLMProvider.OPENAI: {
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            "description": "OpenAI GPT models for intelligent response generation",
            "requires_api_key": True
        },
        LLMProvider.GOOGLE: {
            "models": ["gemini-pro", "gemini-pro-vision"],
            "description": "Google Gemini models for intelligent response generation",
            "requires_api_key": True
        }
    }

    @classmethod
    def create_model(cls, provider: LLMProvider, model_name: str = None, api_key: str = None) -> LLMModel:
        """Create an LLM model instance"""

        if provider == LLMProvider.NONE:
            return NoLLMModel()

        elif provider == LLMProvider.OPENAI:
            model_name = model_name or "gpt-3.5-turbo"
            return OpenAIModel(model_name=model_name, api_key=api_key)

        elif provider == LLMProvider.GOOGLE:
            model_name = model_name or "gemini-pro"
            return GoogleModel(model_name=model_name, api_key=api_key)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        return cls.AVAILABLE_MODELS

class LLMService:
    """Service for managing LLM operations"""

    def __init__(self, provider: LLMProvider = LLMProvider.NONE, model_name: str = None, api_key: str = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.llm_model = LLMFactory.create_model(provider, model_name, api_key)

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using the configured LLM"""
        return await self.llm_model.generate_response(context, query)

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return self.llm_model.get_model_info()

    def change_model(self, provider: LLMProvider, model_name: str = None, api_key: str = None) -> bool:
        """Change the LLM model"""
        try:
            new_model = LLMFactory.create_model(provider, model_name, api_key)
            self.llm_model = new_model
            self.provider = provider
            self.model_name = model_name
            self.api_key = api_key
            logger.info(f"Changed LLM model to {provider.value}: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change LLM model: {str(e)}")
            return False