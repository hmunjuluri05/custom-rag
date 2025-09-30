"""Model configuration loader from YAML files"""
import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

# Define enums here to avoid circular imports
class EmbeddingProvider(Enum):
    """Enum for embedding providers"""
    OPENAI = "openai"
    GOOGLE = "google"

class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    GOOGLE = "google"

logger = logging.getLogger(__name__)

class ModelConfigLoader:
    """Loads and manages model configuration from YAML files"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config loader"""
        if config_path is None:
            # Default to config/models.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "models.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Model config file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)

            logger.info(f"Loaded model configuration from {self.config_path}")
            self._validate_config()

        except Exception as e:
            logger.error(f"Failed to load model config from {self.config_path}: {str(e)}")
            raise

    def _validate_config(self):
        """Validate the loaded configuration"""
        if not self._config:
            raise ValueError("Configuration is empty")

        required_sections = ['embedding_models', 'llm_models', 'defaults']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section: {section}")

        # Validate embedding models
        embedding_models = self._config.get('embedding_models', {})
        for model_name, model_info in embedding_models.items():
            required_fields = ['provider', 'dimension', 'description']
            for field in required_fields:
                if field not in model_info:
                    raise ValueError(f"Embedding model {model_name} missing required field: {field}")

        # Validate LLM models
        llm_models = self._config.get('llm_models', {})
        for provider_name, provider_info in llm_models.items():
            required_fields = ['description', 'models']
            for field in required_fields:
                if field not in provider_info:
                    raise ValueError(f"LLM provider {provider_name} missing required field: {field}")

        logger.info("Model configuration validation passed")

    def get_embedding_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available embedding models"""
        return self._config.get('embedding_models', {})

    def get_embedding_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific embedding model"""
        return self._config.get('embedding_models', {}).get(model_name)

    def get_llm_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available LLM models"""
        return self._config.get('llm_models', {})

    def get_llm_provider_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific LLM provider"""
        return self._config.get('llm_models', {}).get(provider)

    def get_llm_model_gateway_url(self, provider: str, model: str = None) -> str:
        """Get gateway URL for a specific LLM model with BASE_URL + gateway_url concatenation"""
        base_url = os.getenv('BASE_URL')
        provider_info = self.get_llm_provider_info(provider)

        # Define provider-specific fallback URLs
        fallback_urls = {
            'openai': 'https://api.openai.com/v1',
            'google': 'https://generativelanguage.googleapis.com/v1beta'
        }
        fallback_url = fallback_urls.get(provider.lower(), 'https://api.openai.com/v1')

        if not provider_info:
            logger.warning(f"No provider info found for: {provider}, using fallback URL: {fallback_url}")
            return fallback_url

        # Get gateway_url path from config
        gateway_path = None

        # Check if model-specific gateway_url exists
        if model and 'models' in provider_info:
            model_info = provider_info['models'].get(model, {})
            gateway_path = model_info.get('gateway_url')

        # Fall back to provider-level gateway_url
        if not gateway_path:
            gateway_path = provider_info.get('gateway_url')

        # If no gateway path found, use provider-specific fallback
        if not gateway_path:
            logger.warning(f"No gateway_url found for provider: {provider}, model: {model}, using fallback: {fallback_url}")
            return fallback_url

        # If no BASE_URL, use provider-specific fallback
        if not base_url:
            logger.info(f"BASE_URL not set, using provider fallback for {provider}: {fallback_url}")
            return fallback_url

        # Concatenate BASE_URL + gateway_path
        # Ensure proper concatenation (handle trailing/leading slashes)
        base_url = base_url.rstrip('/')
        gateway_path = gateway_path.lstrip('/')
        final_url = f"{base_url}/{gateway_path}"

        logger.debug(f"Built gateway URL for {provider}/{model}: {final_url}")
        return final_url


    def get_embedding_model_gateway_url(self, model_name: str) -> str:
        """Get gateway URL for a specific embedding model with BASE_URL + gateway_url concatenation"""
        base_url = os.getenv('BASE_URL')
        model_info = self.get_embedding_model_info(model_name)

        # Determine provider-specific fallback based on model name
        if "google" in model_name.lower() or "models/" in model_name.lower():
            fallback_url = "https://generativelanguage.googleapis.com/v1beta"
            provider = "google"
        else:
            fallback_url = "https://api.openai.com/v1"
            provider = "openai"

        if not model_info:
            logger.warning(f"No model info found for embedding model: {model_name}, using {provider} fallback: {fallback_url}")
            return fallback_url

        gateway_path = model_info.get('gateway_url')
        if not gateway_path:
            logger.warning(f"No gateway_url found for embedding model: {model_name}, using {provider} fallback: {fallback_url}")
            return fallback_url

        # If no BASE_URL, use provider-specific fallback
        if not base_url:
            logger.info(f"BASE_URL not set, using {provider} fallback for embedding model {model_name}: {fallback_url}")
            return fallback_url

        # Concatenate BASE_URL + gateway_path
        # Ensure proper concatenation (handle trailing/leading slashes)
        base_url = base_url.rstrip('/')
        gateway_path = gateway_path.lstrip('/')
        final_url = f"{base_url}/{gateway_path}"

        logger.debug(f"Built gateway URL for embedding model {model_name}: {final_url}")
        return final_url



    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return self._config.get('defaults', {})

    def get_default_embedding_model(self) -> str:
        """Get the default embedding model"""
        return self._config.get('defaults', {}).get('embedding_model', 'text-embedding-3-small')

    def get_default_llm_provider(self) -> str:
        """Get the default LLM provider"""
        return self._config.get('defaults', {}).get('llm_provider', 'openai')

    def get_default_llm_model(self) -> str:
        """Get the default LLM model"""
        return self._config.get('defaults', {}).get('llm_model', 'gpt-4')

    def is_valid_embedding_model(self, model_name: str) -> bool:
        """Check if an embedding model is valid"""
        return model_name in self._config.get('embedding_models', {})

    def is_valid_llm_provider(self, provider: str) -> bool:
        """Check if an LLM provider is valid"""
        return provider in self._config.get('llm_models', {})

    def is_valid_llm_model(self, provider: str, model: str) -> bool:
        """Check if an LLM model is valid for a provider"""
        provider_info = self.get_llm_provider_info(provider)
        if not provider_info:
            return False
        return model in provider_info.get('models', [])

    def get_embedding_provider(self, model_name: str) -> Optional[EmbeddingProvider]:
        """Get the provider enum for an embedding model"""
        model_info = self.get_embedding_model_info(model_name)
        if not model_info:
            return None

        provider_str = model_info.get('provider', '').lower()
        try:
            return EmbeddingProvider(provider_str)
        except ValueError:
            logger.error(f"Invalid embedding provider: {provider_str}")
            return None

    @staticmethod
    def get_llm_provider_enum(provider: str) -> Optional[LLMProvider]:
        """Get the provider enum for an LLM provider"""
        try:
            return LLMProvider(provider.lower())
        except ValueError:
            logger.error(f"Invalid LLM provider: {provider}")
            return None

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Model configuration reloaded")

    def get_supported_embedding_providers(self) -> List[str]:
        """Get list of supported embedding providers"""
        return self._config.get('validation', {}).get('embedding_providers', ['openai', 'google'])

    def get_supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM providers"""
        return self._config.get('validation', {}).get('llm_providers', ['openai', 'google'])

    def get_gateway_api_key_header(self) -> str:
        """Get the header name for Kong API key authentication"""
        return self._config.get('gateway', {}).get('api_key_header', 'api-key')

    def get_gateway_version_header(self) -> str:
        """Get the header name for Kong gateway version"""
        return self._config.get('gateway', {}).get('gateway_version_header', 'ai-gateway-version')

    def get_gateway_version(self) -> str:
        """Get the Kong gateway version"""
        return self._config.get('gateway', {}).get('gateway_version', 'v2')

    def get_gateway_headers(self, api_key: str) -> Dict[str, str]:
        """Get Kong API Gateway headers in the correct format"""
        if not api_key:
            return {}

        return {
            self.get_gateway_api_key_header(): api_key,
            self.get_gateway_version_header(): self.get_gateway_version()
        }

    def is_gateway_auth_enabled(self) -> bool:
        """Check if gateway authentication is enabled"""
        return self._config.get('gateway', {}).get('use_gateway_auth', True)

    def get_gateway_config(self) -> Dict[str, Any]:
        """Get complete gateway configuration"""
        return self._config.get('gateway', {
            'api_key_header': 'api-key',
            'gateway_version_header': 'ai-gateway-version',
            'gateway_version': 'v2',
            'use_gateway_auth': True
        })

# Global config loader instance
_config_loader = None

def get_model_config() -> ModelConfigLoader:
    """Get the global model configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ModelConfigLoader()
    return _config_loader

def reload_model_config():
    """Reload the global model configuration"""
    global _config_loader
    if _config_loader is not None:
        _config_loader.reload_config()
    else:
        _config_loader = ModelConfigLoader()

def get_default_llm_config() -> tuple:
    """Get default LLM configuration from environment variables and YAML config"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    config = get_model_config()

    # Try environment variables first, then fall back to YAML defaults
    provider_str = os.getenv('DEFAULT_LLM_PROVIDER', config.get_default_llm_provider()).lower()
    model_name = os.getenv('DEFAULT_LLM_MODEL', config.get_default_llm_model())

    # Determine provider
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        provider = LLMProvider.OPENAI

    # Get API key and base URL
    api_key = get_api_config()
    base_url = config.get_llm_model_gateway_url(provider_str, model_name)

    return provider, model_name, api_key, base_url


def get_api_config() -> str:
    """Get API Gateway configuration"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv('API_KEY')

    return api_key

def is_api_configured() -> bool:
    """Check if API Gateway is properly configured"""
    api_key = get_api_config()
    return bool(api_key)