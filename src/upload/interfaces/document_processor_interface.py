from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class IDocumentProcessor(ABC):
    """Interface for document processing operations"""

    @abstractmethod
    async def process_file(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Process a single file and return processed content"""
        pass

    @abstractmethod
    async def process_files(self, file_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple files and return processed contents"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a file"""
        pass