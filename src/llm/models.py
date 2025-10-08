from abc import abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
import logging
from .interfaces.llm_service_interface import ILLMService
from .interfaces.llm_model_interface import ILLMModel, ILLMModelFactory

logger = logging.getLogger(__name__)


class LLMModel(ILLMModel):
    """Abstract base class for LLM models"""

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


class OpenAILLMModel(LLMModel):
    """OpenAI model integration with API Gateway support"""

    def __init__(self, model_name: str = "gpt-4", provider: str = "openai"):
        self.model_name = model_name
        self.provider = provider

        # Get configuration from config system
        from ..config.model_config import get_model_config, get_api_config
        config = get_model_config()

        # Get API key and base URL from configuration
        self.api_key = get_api_config()
        self.base_url = config.get_llm_model_gateway_url(provider, model_name)

        # Validate required configuration
        if not self.api_key:
            raise ValueError("API_KEY is required. Set API_KEY environment variable.")
        if not self.base_url:
            raise ValueError(f"No gateway URL found for provider: {provider}, model: {model_name}")

        try:
            from langchain_openai import ChatOpenAI
            import httpx

            # Create custom HTTP client with API Gateway headers
            # Headers configured via config/models.yaml (e.g., {"api-key": key, "ai-gateway-version": "v2"})
            async_client = httpx.AsyncClient(headers=config.get_gateway_headers(self.api_key))

            # Initialize ChatOpenAI with API Gateway
            # Use "dummy" for api_key since API Gateway handles authentication via custom header
            # Minimal configuration to avoid parameter conflicts
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key="dummy",  # API Gateway handles auth via custom header
                base_url=self.base_url,
                temperature=0.7,
                http_async_client=async_client
            )

            logger.info(f"Initialized OpenAI model: {model_name} via API Gateway: {self.base_url}")

        except ImportError:
            logger.error("LangChain OpenAI library not installed. Run: uv add langchain-openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Modern OpenAI client: {str(e)}")
            raise

    async def generate_response(self, context: str, query: str, callbacks: list = None, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Modern OpenAI with built-in retry logic"""

        default_system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly.

        CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THESE:
        - Use TWO line breaks (\\n\\n) between different paragraphs or topics
        - Use numbered lists (1., 2., 3.) when presenting multiple points
        - Use bullet points (-) for sub-items or features
        - Add line breaks after each main point for readability
        - NEVER write everything as one big paragraph - break it up!
        - Structure your response like this example:

        Topic introduction paragraph.

        1. First point with explanation

        2. Second point with explanation

        Final conclusion paragraph."""

        # Use custom system prompt if provided, otherwise use default
        if system_prompt is None:
            system_prompt = default_system_prompt

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

            # Use Modern's async invoke method with only valid config parameters
            # Filter out any kwargs that could cause parameter conflicts
            valid_config_keys = {'callbacks', 'tags', 'metadata', 'run_name'}
            config = {}

            if callbacks:
                config["callbacks"] = callbacks

            # Add any other valid config parameters from kwargs
            for key, value in kwargs.items():
                if key in valid_config_keys:
                    config[key] = value

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

    async def generate_response_stream(self, context: str, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Modern OpenAI"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly.

        CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THESE:
        - Use TWO line breaks (\\n\\n) between different paragraphs or topics
        - Use numbered lists (1., 2., 3.) when presenting multiple points
        - Use bullet points (-) for sub-items or features
        - Add line breaks after each main point for readability
        - NEVER write everything as one big paragraph - break it up!
        - Structure your response like this example:

        Topic introduction paragraph.

        1. First point with explanation

        2. Second point with explanation

        Final conclusion paragraph."""

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

            # Use streaming for real-time responses with only valid config parameters
            # Filter out any kwargs that could cause parameter conflicts
            valid_config_keys = {'callbacks', 'tags', 'metadata', 'run_name'}
            config = {}

            callbacks = kwargs.get('callbacks', [])
            if callbacks:
                config["callbacks"] = callbacks

            # Add any other valid config parameters from kwargs
            for key, value in kwargs.items():
                if key in valid_config_keys and key != 'callbacks':  # callbacks already handled
                    config[key] = value

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


class GoogleLLMModel(LLMModel):
    """Google Gemini model integration with API Gateway support"""

    def __init__(self, model_name: str = "gemini-pro", provider: str = "google"):
        self.model_name = model_name
        self.provider = provider

        # Get configuration from config system
        from ..config.model_config import get_model_config, get_api_config
        config = get_model_config()

        # Get API key and base URL from configuration
        self.api_key = get_api_config()
        self.base_url = config.get_llm_model_gateway_url(provider, model_name)

        # Validate required configuration
        if not self.api_key:
            raise ValueError("API_KEY is required. Set API_KEY environment variable.")
        if not self.base_url:
            raise ValueError(f"No gateway URL found for provider: {provider}, model: {model_name}")

        try:
            # Use Google's native LLM implementation
            from langchain_google_genai import ChatGoogleGenerativeAI
            import httpx

            # Create custom HTTP client with API Gateway headers
            # Headers configured via config/models.yaml (e.g., {"api-key": key, "ai-gateway-version": "v2"})
            async_client = httpx.AsyncClient(headers=config.get_gateway_headers(self.api_key))


            # Initialize Google LLM with API Gateway
            # Use "dummy" for google_api_key since API Gateway handles authentication via custom header
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key="dummy",  # API Gateway handles auth via custom header
                temperature=0.7,
                max_output_tokens=1000,
                timeout=30,
                # Note: Google LangChain may not support custom HTTP client
                # If it doesn't work, we'll need to use a different approach
                http_async_client=async_client if hasattr(ChatGoogleGenerativeAI, 'http_async_client') else None
            )

            logger.info(f"Initialized Modern Google model: {model_name}")

        except ImportError:
            logger.error("LangChain Google GenAI library not installed. Run: uv add langchain-google-genai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Modern Google client: {str(e)}")
            raise

    async def generate_response(self, context: str, query: str, callbacks: list = None, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Modern Google model via API Gateway"""

        default_system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly.

        CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THESE:
        - Use TWO line breaks (\\n\\n) between different paragraphs or topics
        - Use numbered lists (1., 2., 3.) when presenting multiple points
        - Use bullet points (-) for sub-items or features
        - Add line breaks after each main point for readability
        - NEVER write everything as one big paragraph - break it up!
        - Structure your response like this example:

        Topic introduction paragraph.

        1. First point with explanation

        2. Second point with explanation

        Final conclusion paragraph."""

        # Use custom system prompt if provided, otherwise use default
        if system_prompt is None:
            system_prompt = default_system_prompt

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
            # Note: Google LangChain integration may not support callbacks the same way
            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Google Gemini API error: {str(e)}")
            return f"❌ Error generating response with Google Gemini via API Gateway: {str(e)}"

    async def generate_response_stream(self, context: str, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Modern Google model via API Gateway"""

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly.

        CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THESE:
        - Use TWO line breaks (\\n\\n) between different paragraphs or topics
        - Use numbered lists (1., 2., 3.) when presenting multiple points
        - Use bullet points (-) for sub-items or features
        - Add line breaks after each main point for readability
        - NEVER write everything as one big paragraph - break it up!
        - Structure your response like this example:

        Topic introduction paragraph.

        1. First point with explanation

        2. Second point with explanation

        Final conclusion paragraph."""

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
    """Factory for creating LLM models"""

    @classmethod
    def create_model(cls, provider: str, model_name: str, **config) -> LLMModel:
        """Create an LLM model instance - models handle their own configuration"""
        from ..config.model_config import get_model_config
        model_config = get_model_config()

        # Convert string provider to enum if needed
        if isinstance(provider, str):
            provider_enum = model_config.get_llm_provider_enum(provider)
        else:
            provider_enum = provider

        # Validate provider and model
        if not model_config.is_valid_llm_provider(provider):
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not model_config.is_valid_llm_model(provider, model_name):
            available_models = model_config.get_llm_provider_info(provider).get('models', [])
            raise ValueError(f"Model {model_name} not available for provider {provider}. Available models: {available_models}")

        # Create model based on provider - models handle their own config
        if provider_enum.value == "openai":
            return OpenAILLMModel(model_name=model_name, provider=provider)
        elif provider_enum.value == "google":
            return GoogleLLMModel(model_name=model_name, provider=provider)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # Backward compatibility method for old signature
    @classmethod
    def create_model_legacy(cls, provider = None, model_name: str = None) -> LLMModel:
        """Legacy method for backward compatibility"""
        from ..config.model_config import get_model_config
        config = get_model_config()

        # Use defaults if not specified and handle string/enum provider input
        if provider is None:
            provider_str = config.get_default_llm_provider()
            from ..config.model_config import ModelConfigLoader
            provider = ModelConfigLoader.get_llm_provider_enum(provider_str)
        elif isinstance(provider, str):
            # Convert string provider to enum
            from ..config.model_config import ModelConfigLoader
            provider = ModelConfigLoader.get_llm_provider_enum(provider)
        # If provider is already an enum, use it as-is

        if model_name is None:
            if provider.value == "openai":
                provider_info = config.get_llm_provider_info('openai')
                model_name = provider_info.get('default_model', 'gpt-4') if provider_info else 'gpt-4'
            elif provider.value == "google":
                provider_info = config.get_llm_provider_info('google')
                model_name = provider_info.get('default_model', 'gemini-pro') if provider_info else 'gemini-pro'
            else:
                model_name = config.get_default_llm_model()


        # Validate provider and model
        provider_str = provider.value
        if not config.is_valid_llm_provider(provider_str):
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not config.is_valid_llm_model(provider_str, model_name):
            available_models = config.get_llm_provider_info(provider_str).get('models', [])
            raise ValueError(f"Model {model_name} not available for provider {provider}. Available models: {available_models}")

        # Create model based on provider - models handle their own config
        if provider.value == "openai":
            return OpenAILLMModel(model_name=model_name, provider=provider_str)
        elif provider.value == "google":
            return GoogleLLMModel(model_name=model_name, provider=provider_str)
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
    """Service for managing LLM operations"""

    def __init__(self, provider: str = None, model_name: str = None, enable_callbacks: bool = True):
        from ..config.model_config import get_model_config
        config = get_model_config()

        # Use defaults if not specified
        if provider is None:
            provider = config.get_default_llm_provider()
        if model_name is None:
            provider_info = config.get_llm_provider_info(provider)
            model_name = provider_info.get('default_model', 'gpt-4') if provider_info else 'gpt-4'

        self.provider = provider
        self.model_name = model_name

        # Create model using factory - models handle their own configuration
        self.llm_model = LLMFactory.create_model(provider=provider, model_name=model_name)

        # Initialize callback system
        if enable_callbacks:
            from .callbacks import CallbackManager
            self.callback_manager = CallbackManager(log_level="INFO", enable_streaming=True)
            logger.info("Modern callbacks enabled for monitoring and logging")
        else:
            self.callback_manager = None

    async def generate_response(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using the configured Modern LLM"""
        callbacks = self.callback_manager.get_callbacks() if self.callback_manager else []
        return await self.llm_model.generate_response(context, query, callbacks=callbacks, system_prompt=system_prompt)

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
            # Handle both string and enum provider inputs
            provider_str = provider.value if hasattr(provider, 'value') else provider

            new_model = LLMFactory.create_model(provider_str, model_name)
            self.llm_model = new_model
            self.provider = provider_str
            self.model_name = model_name
            logger.info(f"Changed LLM model to {provider_str}: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change LLM model: {str(e)}")
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