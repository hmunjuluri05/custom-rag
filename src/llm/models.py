"""
DEPRECATED: This module contains legacy LLM implementations.
Use src.llm.langchain_models instead for new development.
"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import os
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "src.llm.models is deprecated. Use src.llm.langchain_models for new development.",
    DeprecationWarning,
    stacklevel=2
)


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


class OpenAIModel(LLMModel):
    """OpenAI GPT model integration with optional API Gateway support"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url
        self._client = None

        try:
            from openai import AsyncOpenAI
            client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
            logger.info(f"Initialized OpenAI client with model: {model_name} via API Gateway: {self.base_url}")

            self._client = AsyncOpenAI(**client_kwargs)

        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using OpenAI GPT with retry logic"""
        if not self._client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")

        system_prompt = """You are a helpful assistant that answers questions based on provided context from documents.
        Use only the information from the context to answer questions. If the context doesn't contain relevant information,
        say so clearly. Be concise and accurate."""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        # Retry logic for connection issues
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    timeout=30  # Add timeout
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e).lower()

                # Handle different types of errors
                if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"OpenAI connection failed after {max_retries} attempts: {e}")
                        return self._format_connection_error(e)

                elif "rate limit" in error_msg or "quota" in error_msg:
                    logger.error(f"OpenAI rate limit/quota error: {e}")
                    return "❌ OpenAI API rate limit exceeded or quota reached. Please check your OpenAI account billing and usage limits."

                elif "api key" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                    logger.error(f"OpenAI authentication error: {e}")
                    return "❌ OpenAI API authentication failed. Please check your API key in the .env file or environment variables."

                elif "model" in error_msg and "not found" in error_msg:
                    logger.error(f"OpenAI model error: {e}")
                    return f"❌ OpenAI model '{self.model_name}' not found or not accessible. Please check your model name and permissions."

                else:
                    # Generic error
                    logger.error(f"OpenAI API error: {e}")
                    return f"❌ OpenAI API error: {e}"

        # This shouldn't be reached, but just in case
        return "❌ Failed to generate response after multiple attempts."

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

        elif "connection refused" in error_msg.lower() or "network" in error_msg.lower():
            return f"""{base_message}

🌐 Network Connection Issue:
• Check your internet connection
• Verify firewall/proxy settings allow HTTPS to api.openai.com
• If on corporate network, contact IT about API access
• Try from a different network (mobile hotspot) to test
• Run: python diagnose_openai_connection.py for detailed diagnosis"""

        else:
            return f"""{base_message}

🔧 Troubleshooting Steps:
• Check your internet connection
• Verify OpenAI API status: https://status.openai.com/
• Ensure API key is correctly configured
• Run: python diagnose_openai_connection.py for detailed diagnosis

Error details: {error_msg}"""

    def get_model_info(self) -> Dict[str, Any]:
        if self.base_url:
            return {
                "provider": "openai",
                "model_name": self.model_name,
                "description": f"OpenAI {self.model_name} via API Gateway with intelligent response generation",
                "cost": "Enterprise Gateway",
                "gateway_url": self.base_url
            }
        else:
            return {
                "provider": "openai",
                "model_name": self.model_name,
                "description": f"OpenAI {self.model_name} with intelligent response generation",
                "cost": "Paid API"
            }

class GoogleModel(LLMModel):
    """Google Gemini model integration with optional API Gateway support"""

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name

        # API key and base_url are mandatory
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")

        self.api_key = api_key
        self.base_url = base_url
        self._model = None
        self._use_gateway = True  # Always using gateway since base_url is mandatory

        # For API Gateway, we'll use aiohttp directly since google-generativeai doesn't support custom base URLs
        logger.info(f"Initialized Google Gemini client with model: {model_name} via API Gateway: {self.base_url}")

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using Google Gemini"""

        try:
            prompt = f"""You are a helpful assistant that answers questions based on provided context from documents.
            Use only the information from the context to answer questions. If the context doesn't contain relevant information,
            say so clearly. Be concise and accurate.

Context from documents:
{context}

Question: {query}

Please answer the question based only on the provided context."""

            if self._use_gateway:
                # Use API Gateway with HTTP requests
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"].strip()
                        else:
                            error_text = await response.text()
                            logger.error(f"API Gateway error {response.status}: {error_text}")
                            return f"Error from API Gateway: {response.status} - {error_text}"
            else:
                # Direct Google API
                if not self._model:
                    raise ValueError("Google Gemini client not initialized.")

                # Google Gemini doesn't have built-in async, so we'll use sync and wrap it
                def generate_sync():
                    response = self._model.generate_content(prompt)
                    return response.text

                response_text = await asyncio.get_event_loop().run_in_executor(None, generate_sync)
                return response_text.strip()

        except asyncio.TimeoutError:
            logger.error("Google Gemini request timed out")
            return "Error: Google Gemini request timed out"
        except Exception as e:
            logger.error(f"Google Gemini API error: {str(e)}")
            return f"Error generating response with Google Gemini: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        if self.base_url:
            return {
                "provider": "google",
                "model_name": self.model_name,
                "description": f"Google {self.model_name} via API Gateway with intelligent response generation",
                "cost": "Enterprise Gateway",
                "gateway_url": self.base_url
            }
        else:
            return {
                "provider": "google",
                "model_name": self.model_name,
                "description": f"Google {self.model_name} with intelligent response generation",
                "cost": "Paid API"
            }

class LLMFactory:
    """Factory for creating LLM models"""


    @classmethod
    def create_model(cls, provider = None, model_name: str = None, api_key: str = None, base_url: str = None) -> LLMModel:
        """Create an LLM model instance with optional API Gateway support via base_url"""
        from ..config.model_config import get_model_config, LLMProvider, get_kong_config, derive_llm_url
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
            api_key = get_kong_config()

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
            return OpenAIModel(model_name=model_name, api_key=api_key, base_url=base_url)
        elif provider.value == "google":
            return GoogleModel(model_name=model_name, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models"""
        from ..config.model_config import get_model_config
        config = get_model_config()
        return config.get_llm_models()

class LLMService:
    """Service for managing LLM operations"""

    def __init__(self, provider = None, model_name: str = None, api_key: str = None, base_url: str = None):
        from ..config.model_config import LLMProvider
        self.provider = provider or LLMProvider.OPENAI
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = LLMFactory.create_model(provider, model_name, api_key, base_url)

    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using the configured LLM"""
        return await self.llm_model.generate_response(context, query)

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return self.llm_model.get_model_info()

    def change_model(self, provider, model_name: str = None, api_key: str = None, base_url: str = None) -> bool:
        """Change the LLM model"""
        try:
            new_model = LLMFactory.create_model(provider, model_name, api_key, base_url)
            self.llm_model = new_model
            self.provider = provider
            self.model_name = model_name
            self.api_key = api_key
            self.base_url = base_url
            logger.info(f"Changed LLM model to {provider.value}: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change LLM model: {str(e)}")
            return False