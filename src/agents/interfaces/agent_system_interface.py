from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IAgentSystem(ABC):
    """Interface for agent system operations"""

    @abstractmethod
    async def query_with_agent(self,
                              query_text: str,
                              agent_type: str = "general",
                              **kwargs) -> Dict[str, Any]:
        """Query using agent with specified type"""
        pass

    @abstractmethod
    def get_available_agents(self) -> list:
        """Get list of available agent types"""
        pass

    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent system"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if agent system is available and functional"""
        pass

    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed information about the agent system"""
        pass