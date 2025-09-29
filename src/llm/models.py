from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import os
import asyncio
from .interfaces.llm_service_interface import ILLMService
from .interfaces.llm_model_interface import ILLMModel, ILLMModelFactory

logger = logging.getLogger(__name__)


class LLMModel(ILLMModel):
    """Abstract base class for Modern LLM models"""

    @abstractmethod
    async def generate_response(self, context: str, query: str, **kwargs) -> str:
        """Generate response using context and query"""
        pass

    @abstractmethod
    async def generate_response_stream(self, context: str, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection"""
        pass

    @abstractmethod
    def get_token_limit(self) -> Optional[int]:
        """Get token limit"""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        pass


class ModernOpenAIModel(LLMModel):
    """OpenAI model integration with Kong API Gateway support"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        try:
            from langchain_openai import ChatOpenAI

            # Initialize ChatOpenAI with Kong API Gateway
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.7,
                max_tokens=1000,
                timeout=30
            )

            logger.info(f"Initialized OpenAI model: {model_name} via API Gateway: {self.base_url}")

        except ImportError:
            logger.error("LangChain OpenAI library not installed. Run: uv add langchain-openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Modern OpenAI client: {str(e)}")
            raise

    async def generate_response(self, context: str, query: str, callbacks: list = None) -> str:
        """Generate response using Modern OpenAI with built-in retry logic"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly. Be concise and accurate."""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Use Modern's async invoke method with callbacks
            config = {"callbacks": callbacks} if callbacks else {}
            response = await self.llm.ainvoke(messages, config=config)
            return response.content.strip()

        except Exception as e:
            error_msg = str(e).lower()

            # Handle specific error types with user-friendly messages
            if "rate limit" in error_msg or "quota" in error_msg:
                logger.error(f"OpenAI rate limit/quota error: {e}")
                return "❌ OpenAI API rate limit exceeded or quota reached. Please check your OpenAI account billing and usage limits."

            elif "api key" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                logger.error(f"OpenAI authentication error: {e}")
                return "❌ OpenAI API authentication failed. Please check your API key in the .env file or environment variables."

            elif "model" in error_msg and "not found" in error_msg:
                logger.error(f"OpenAI model error: {e}")
                return f"❌ OpenAI model '{self.model_name}' not found or not accessible. Please check your model name and permissions."

            elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                logger.error(f"OpenAI connection error: {e}")
                return self._format_connection_error(e)

            else:
                logger.error(f"OpenAI API error: {e}")
                return f"❌ OpenAI API error: {e}"

    async def generate_response_stream(self, context: str, query: str, callbacks: list = None):
        """Generate streaming response using Modern OpenAI"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly. Be concise and accurate."""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Use streaming for real-time responses with callbacks
            config = {"callbacks": callbacks} if callbacks else {}
            async for chunk in self.llm.astream(messages, config=config):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            error_msg = str(e).lower()

            # Handle specific error types with user-friendly messages
            if "rate limit" in error_msg or "quota" in error_msg:
                logger.error(f"OpenAI rate limit/quota error: {e}")
                yield "❌ OpenAI API rate limit exceeded or quota reached. Please check your OpenAI account billing and usage limits."

            elif "api key" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                logger.error(f"OpenAI authentication error: {e}")
                yield "❌ OpenAI API authentication failed. Please check your API key in the .env file or environment variables."

            elif "model" in error_msg and "not found" in error_msg:
                logger.error(f"OpenAI model error: {e}")
                yield f"❌ OpenAI model '{self.model_name}' not found or not accessible. Please check your model name and permissions."

            elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                logger.error(f"OpenAI connection error: {e}")
                error_response = self._format_connection_error(e)
                yield error_response

            else:
                logger.error(f"OpenAI API error: {e}")
                yield f"❌ OpenAI API error: {e}"

    def _format_connection_error(self, error: Exception) -> str:
        """Format connection error with helpful troubleshooting information"""
        error_msg = str(error)
        base_message = "❌ Connection error communicating with OpenAI API."

        if "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
            return f"""{base_message}

🔒 SSL Certificate Issue Detected:
• This often happens on corporate networks
• Try adding BYPASS_SSL_VERIFICATION=true to your .env file
• Or contact your IT team about certificate configuration
• Run: python diagnose_openai_connection.py for detailed help"""

        elif "timeout" in error_msg.lower():
            return f"""{base_message}

⏱️ Request Timeout:
• Check your internet connection
• OpenAI services might be experiencing delays
• Try again in a few moments
• Check status: https://status.openai.com/"""

        else:
            return f"""{base_message}

🔧 Troubleshooting Steps:
• Check your internet connection
• Verify OpenAI API status: https://status.openai.com/
• Ensure API key is correctly configured
• Run: python diagnose_openai_connection.py for detailed diagnosis

Error details: {error_msg}"""

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "description": f"Modern OpenAI {self.model_name} via API Gateway with intelligent response generation",
            "cost": "Enterprise Gateway",
            "gateway_url": self.base_url,
            "framework": "langchain",
            "capabilities": ["response_generation", "streaming", "context_aware"],
            "token_limit": self.get_token_limit()
        }

    def validate_connection(self) -> bool:
        """Validate OpenAI connection"""
        try:
            # Simple validation - check if we have required credentials
            return bool(self.api_key and self.base_url and self.llm)
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_token_limit(self) -> Optional[int]:
        """Get token limit for OpenAI model"""
        # Common OpenAI model token limits
        token_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        return token_limits.get(self.model_name, 8192)  # Default to 8k

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: roughly 4 characters per token for English text
        # This is an approximation; for exact counts, use tiktoken library
        return max(1, len(text) // 4)


class ModernGoogleModel(LLMModel):
    """Google Gemini model integration with Kong API Gateway support"""

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url

        try:
            # Use Google's native LLM implementation
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Initialize Google LLM with proper configuration
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=1000,
                timeout=30
            )

            logger.info(f"Initialized Modern Google model: {model_name}")

        except ImportError:
            logger.error("LangChain Google GenAI library not installed. Run: uv add langchain-google-genai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Modern Google client: {str(e)}")
            raise

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using Modern Google model via Kong Gateway"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly. Be concise and accurate."""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Use Modern's async invoke method
            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Google Gemini API error: {str(e)}")
            return f"❌ Error generating response with Google Gemini via Kong Gateway: {str(e)}"

    async def generate_response_stream(self, context: str, query: str):
        """Generate streaming response using Modern Google model via Kong Gateway"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly. Be concise and accurate."""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Use streaming for real-time responses
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Google Gemini streaming error: {str(e)}")
            yield f"❌ Error generating streaming response with Google Gemini: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "google",
            "model_name": self.model_name,
            "description": f"Google {self.model_name} with intelligent response generation",
            "cost": "Pay per use",
            "framework": "langchain",
            "capabilities": ["response_generation", "streaming", "context_aware"],
            "token_limit": self.get_token_limit()
        }

    def validate_connection(self) -> bool:
        """Validate Google connection"""
        try:
            # Simple validation - check if we have required credentials
            return bool(self.api_key and self.llm)
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_token_limit(self) -> Optional[int]:
        """Get token limit for Google model"""
        # Common Google model token limits
        token_limits = {
            "gemini-pro": 32768,
            "gemini-1.5-pro": 1000000,
            "gemini-1.5-flash": 1000000,
            "gemini-ultra": 32768
        }
        return token_limits.get(self.model_name, 32768)  # Default to 32k

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: roughly 4 characters per token
        # Google models may have different tokenization, but this is a reasonable approximation
        return max(1, len(text) // 4)


class LLMFactory(ILLMModelFactory):
    """Factory for creating Modern LLM models"""

    @classmethod
    def create_model(cls, provider = None, model_name: str = None, api_key: str = None, base_url: str = None) -> LLMModel:
        """Create a Modern LLM model instance with Kong API Gateway support"""
        from ..config.model_config import get_model_config, LLMProvider, get_api_config, derive_llm_url
        config = get_model_config()

        # Use defaults if not specified
        if provider is None:
            provider_str = config.get_default_llm_provider()
            provider = config.get_llm_provider_enum(provider_str)

        if model_name is None:
            if provider.value == "openai":
                provider_info = config.get_llm_provider_info('openai')
                model_name = provider_info.get('default_model', 'gpt-4') if provider_info else 'gpt-4'
            elif provider.value == "google":
                provider_info = config.get_llm_provider_info('google')
                model_name = provider_info.get('default_model', 'gemini-pro') if provider_info else 'gemini-pro'
            else:
                model_name = config.get_default_llm_model()

        # Get defaults for api_key and base_url if not provided
        if api_key is None:
            api_key = get_api_config()

        if base_url is None:
            base_url = derive_llm_url(provider.value)

        # Validate provider and model
        provider_str = provider.value
        if not config.is_valid_llm_provider(provider_str):
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not config.is_valid_llm_model(provider_str, model_name):
            available_models = config.get_llm_provider_info(provider_str).get('models', [])
            raise ValueError(f"Model {model_name} not available for provider {provider}. Available models: {available_models}")

        if provider.value == "openai":
            return ModernOpenAIModel(model_name=model_name, api_key=api_key, base_url=base_url)
        elif provider.value == "google":
            return ModernGoogleModel(model_name=model_name, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        from ..config.model_config import get_model_config
        config = get_model_config()
        return config.get_llm_models()

    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported providers"""
        return ["openai", "google"]

    @classmethod
    def validate_model_config(cls, provider: str, model_name: str, **config) -> bool:
        """Validate model configuration without creating the model"""
        try:
            from ..config.model_config import get_model_config
            model_config = get_model_config()

            # Check if provider is supported
            if provider not in cls.get_supported_providers():
                return False

            # Check if model is valid for provider
            if not model_config.is_valid_llm_model(provider, model_name):
                return False

            # Check required config parameters
            api_key = config.get('api_key')
            base_url = config.get('base_url')

            # API key and base_url are required for both providers
            if not api_key or not base_url:
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating model config: {e}")
            return False


class LLMService(ILLMService):
    """Service for managing Modern LLM operations"""

    def __init__(self, provider = None, model_name: str = None, api_key: str = None, base_url: str = None, enable_callbacks: bool = True):
        from ..config.model_config import LLMProvider
        self.provider = provider or LLMProvider.OPENAI
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = LLMFactory.create_model(provider, model_name, api_key, base_url)

        # Initialize callback system
        if enable_callbacks:
            from .callbacks import CallbackManager
            self.callback_manager = CallbackManager(log_level="INFO", enable_streaming=True)
            logger.info("Modern callbacks enabled for monitoring and logging")
        else:
            self.callback_manager = None

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using the configured Modern LLM"""
        callbacks = self.callback_manager.get_callbacks() if self.callback_manager else []
        return await self.llm_model.generate_response(context, query, callbacks=callbacks)

    async def generate_response_stream(self, context: str, query: str):
        """Generate streaming response using the configured Modern LLM"""
        callbacks = self.callback_manager.get_callbacks(include_streaming=True) if self.callback_manager else []
        async for chunk in self.llm_model.generate_response_stream(context, query, callbacks=callbacks):
            yield chunk

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return self.llm_model.get_model_info()

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from callback manager"""
        if self.callback_manager:
            return self.callback_manager.get_comprehensive_metrics()
        return {"error": "Callbacks not enabled"}

    def reset_metrics(self):
        """Reset all metrics"""
        if self.callback_manager:
            self.callback_manager.reset_all_metrics()
            logger.info("LLM service metrics reset")
        else:
            logger.warning("Cannot reset metrics - callbacks not enabled")

    def change_model(self, provider, model_name: str = None, api_key: str = None, base_url: str = None) -> bool:
        """Change the LLM model"""
        try:
            new_model = LLMFactory.create_model(provider, model_name, api_key, base_url)
            self.llm_model = new_model
            self.provider = provider
            self.model_name = model_name
            self.api_key = api_key
            self.base_url = base_url
            logger.info(f"Changed Modern LLM model to {provider.value}: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change Modern LLM model: {str(e)}")
            return False

    # Interface implementation methods
    async def generate_response_with_sources(self,
                                           context: str,
                                           query: str,
                                           sources: list = None) -> Dict[str, Any]:
        """Generate response with source attribution"""
        try:
            # Generate the response using context and query
            response = await self.generate_response(context, query)

            return {
                'response': response,
                'sources': sources or [],
                'query': query,
                'model_info': self.get_model_info()
            }

        except Exception as e:
            logger.error(f"Error generating response with sources: {str(e)}")
            return {
                'response': f"Error generating response: {str(e)}",
                'sources': sources or [],
                'query': query,
                'error': str(e)
            }

    async def generate_streaming_response(self,
                                        context: str,
                                        query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response"""
        try:
            # Check if streaming is available
            if hasattr(self.llm_model, 'generate_response_stream'):
                async for chunk in self.generate_response_stream(context, query):
                    yield {
                        'type': 'chunk',
                        'content': chunk,
                        'model_info': self.get_model_info()
                    }
            else:
                # Fallback to non-streaming
                response = await self.generate_response(context, query)
                yield {
                    'type': 'complete',
                    'content': response,
                    'model_info': self.get_model_info()
                }

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield {
                'type': 'error',
                'content': f"Streaming error: {str(e)}",
                'error': str(e)
            }