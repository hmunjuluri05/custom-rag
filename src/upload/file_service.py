from fastapi import UploadFile, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import aiofiles
import logging
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class FileUploadService:
    """Service for handling file uploads and processing"""

    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.document_processor = DocumentProcessor()
        self.allowed_extensions = {'.pdf', '.docx', '.xlsx', '.xls', '.txt'}

    async def save_uploaded_file(self, file: UploadFile) -> Path:
        """Save uploaded file to disk"""

        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(self.allowed_extensions)}"
            )

        # Create unique filename to avoid conflicts
        file_path = self.upload_dir / file.filename
        counter = 1
        original_stem = file_path.stem

        while file_path.exists():
            file_path = self.upload_dir / f"{original_stem}_{counter}{file_extension}"
            counter += 1

        # Save file
        try:
            async with aiofiles.open(file_path, 'wb') as buffer:
                content = await file.read()
                await buffer.write(content)

            logger.info(f"Saved file: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    async def process_uploaded_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """Process multiple uploaded files"""
        processed_files = []

        for file in files:
            try:

                # Save file
                file_path = await self.save_uploaded_file(file)

                # Extract text content
                text_content = self.document_processor.extract_text(file_path)

                # Get metadata
                metadata = self.document_processor.get_document_metadata(file_path)

                processed_files.append({
                    "filename": file.filename,
                    "file_path": str(file_path),
                    "text_content": text_content,
                    "metadata": metadata,
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                processed_files.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })

        return processed_files

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.allowed_extensions)

    def validate_file(self, file: UploadFile) -> bool:
        """Validate if file format is supported"""
        file_extension = Path(file.filename).suffix.lower()
        return file_extension in self.allowed_extensions

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file from disk"""
        try:
            path = Path(file_path).resolve()  # Resolve to absolute path
            upload_dir_resolved = self.upload_dir.resolve()

            # Check if path is within upload directory
            if not str(path).startswith(str(upload_dir_resolved)):
                logger.error(f"Attempted path traversal: {file_path}")
                return False

            if path.exists() and path.is_file():
                path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False

    def get_upload_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded files"""
        try:
            files = list(self.upload_dir.glob("*"))
            total_files = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            # Group by extension
            extensions = {}
            for file_path in files:
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1

            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files_by_extension": extensions,
                "upload_directory": str(self.upload_dir)
            }

        except Exception as e:
            logger.error(f"Error getting upload stats: {str(e)}")
            return {"error": str(e)}