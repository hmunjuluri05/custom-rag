"""
LLM interfaces for loose coupling and dependency injection.
"""

from .llm_model_interface import ILLMModel, ILLMModelFactory
from .llm_service_interface import ILLMService

__all__ = [
    "ILLMModel",
    "ILLMModelFactory",
    "ILLMService"
]