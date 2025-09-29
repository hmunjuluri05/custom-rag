from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator


class ILLMService(ABC):
    """Interface for LLM service operations"""

    @abstractmethod
    async def generate_response(self, context: str, query: str) -> str:
        """Generate a response using context and query"""
        pass

    @abstractmethod
    async def generate_response_with_sources(self,
                                           context: str,
                                           query: str,
                                           sources: list = None) -> Dict[str, Any]:
        """Generate response with source attribution"""
        pass

    @abstractmethod
    async def generate_streaming_response(self,
                                        context: str,
                                        query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass

    @abstractmethod
    def change_model(self, provider, model_name: str = None, **kwargs) -> bool:
        """Change the current model"""
        pass