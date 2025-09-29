"""Model configuration loader from YAML files"""
import yaml
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
        """Get gateway URL for a specific LLM model with fallback"""
        provider_info = self.get_llm_provider_info(provider)
        if not provider_info:
            # Fallback to default OpenAI URL
            return "https://api.openai.com/v1"

        # Check if model-specific gateway_url exists
        if model and 'models' in provider_info:
            model_info = provider_info['models'].get(model, {})
            if 'gateway_url' in model_info:
                return model_info['gateway_url']

        # Fall back to provider-level gateway_url
        gateway_url = provider_info.get('gateway_url')
        if gateway_url:
            return gateway_url

        # Final fallback to default OpenAI URL
        return "https://api.openai.com/v1"

    def get_embedding_model_gateway_url(self, model_name: str) -> str:
        """Get gateway URL for a specific embedding model with fallback"""
        model_info = self.get_embedding_model_info(model_name)
        if not model_info:
            # Fallback to default OpenAI URL
            return "https://api.openai.com/v1"

        gateway_url = model_info.get('gateway_url')
        if gateway_url:
            return gateway_url

        # Fallback to default OpenAI URL
        return "https://api.openai.com/v1"



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