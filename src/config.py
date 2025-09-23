"""Configuration management for the RAG system"""
import os
from typing import Optional
from dotenv import load_dotenv
from src.llm.models import LLMProvider

"""Get default LLM configuration from environment variables"""
load_dotenv()

def get_default_llm_config() -> tuple[LLMProvider, str, Optional[str], Optional[str]]:
    provider_str = os.getenv('DEFAULT_LLM_PROVIDER', 'openai').lower()
    model_name = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4')

    # Determine provider
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        provider = LLMProvider.OPENAI

    # Get API keys/URLs based on provider
    api_key = None
    base_url = None

    if provider == LLMProvider.OPENAI:
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_GATEWAY_KEY')
        base_url = os.getenv('LLM_API_GATEWAY_URL')  # Optional for API Gateway
    elif provider == LLMProvider.GOOGLE:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_GATEWAY_KEY')
        base_url = os.getenv('LLM_API_GATEWAY_URL')  # Optional for API Gateway

    return provider, model_name, api_key, base_url