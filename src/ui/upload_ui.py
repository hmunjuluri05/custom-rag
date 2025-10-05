from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class UploadUI:
    """UI components for file upload functionality"""

    def __init__(self, templates_dir: str = "src/ui/templates"):
        self.templates = Jinja2Templates(directory=templates_dir)
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup UI routes"""

        @self.router.get("/upload", response_class=HTMLResponse)
        async def upload_page(request: Request):
            """Display file upload page"""
            # Check if LLM chunking is enabled
            from ..config.model_config import get_model_config
            config = get_model_config()
            llm_chunking_enabled = config.is_llm_chunking_enabled()

            return self.templates.TemplateResponse(
                "upload.html",
                {
                    "request": request,
                    "page_title": "Upload Documents",
                    "llm_chunking_enabled": llm_chunking_enabled
                }
            )

        @self.router.get("/upload-status", response_class=HTMLResponse)
        async def upload_status_page(request: Request):
            """Display upload status page"""
            return self.templates.TemplateResponse(
                "upload_status.html",
                {"request": request, "page_title": "Upload Status"}
            )

        @self.router.get("/documents", response_class=HTMLResponse)
        async def documents_page(request: Request):
            """Display documents management page"""
            return self.templates.TemplateResponse(
                "documents.html",
                {"request": request, "page_title": "Document Management"}
            )

    def get_router(self):
        """Get the FastAPI router for UI routes"""
        return self.router

def create_upload_templates():
    """Create HTML templates for upload UI"""

    # Main upload page template
    upload_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }} - RAG System</title>
    <link href="/static/css/style.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">RAG System</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/upload">Upload</a>
                <a class="nav-link" href="/chat">Chat</a>
                <a class="nav-link" href="/documents">Documents</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <h2>Upload Documents</h2>
                <p class="text-muted">Upload PDF, DOCX, XLSX, or TXT files to add to your knowledge base.</p>

                <div class="card">
                    <div class="card-body">
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="files" class="form-label">Select Files</label>
                                <input type="file" class="form-control" id="files" name="files" multiple
                                       accept=".pdf,.docx,.xlsx,.xls,.txt">
                                <div class="form-text">Supported formats: PDF, DOCX, XLSX, XLS, TXT</div>
                            </div>

                            <div class="mb-3">
                                <button type="submit" class="btn btn-primary" id="uploadBtn">
                                    <span class="spinner-border spinner-border-sm me-2 d-none" id="uploadSpinner"></span>
                                    Upload Files
                                </button>
                            </div>
                        </form>

                        <div id="uploadResults" class="mt-3"></div>
                        <div id="uploadProgress" class="mt-3"></div>
                    </div>
                </div>

                <div class="card mt-4">
                    <div class="card-header">
                        <h5>Supported File Types</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-unstyled">
                            <li><strong>PDF:</strong> Portable Document Format files</li>
                            <li><strong>DOCX:</strong> Microsoft Word documents</li>
                            <li><strong>XLSX/XLS:</strong> Microsoft Excel spreadsheets</li>
                            <li><strong>TXT:</strong> Plain text files</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/upload.js"></script>
</body>
</html>
    """

    # Upload status template
    upload_status_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }} - RAG System</title>
    <link href="/static/css/style.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">RAG System</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/upload">Upload</a>
                <a class="nav-link" href="/chat">Chat</a>
                <a class="nav-link" href="/documents">Documents</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h2>Upload Status</h2>
        <div id="statusContainer"></div>
        <a href="/upload" class="btn btn-primary mt-3">Upload More Files</a>
    </div>

    <script>
        // Auto-refresh status every 2 seconds
        setInterval(loadUploadStatus, 2000);
        loadUploadStatus(); // Load initially

        function loadUploadStatus() {
            fetch('/api/upload-status')
                .then(response => response.json())
                .then(data => updateStatusDisplay(data))
                .catch(error => console.error('Error loading status:', error));
        }

        function updateStatusDisplay(data) {
            const container = document.getElementById('statusContainer');
            // Implementation for displaying upload status
        }
    </script>
</body>
</html>
    """

    # Documents management template
    documents_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }} - RAG System</title>
    <link href="/static/css/style.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">RAG System</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/upload">Upload</a>
                <a class="nav-link" href="/chat">Chat</a>
                <a class="nav-link active" href="/documents">Documents</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>Document Management</h2>
            <button class="btn btn-outline-secondary" onclick="refreshDocuments()">
                <i class="bi bi-arrow-clockwise"></i> Refresh
            </button>
        </div>

        <div class="row mb-3">
            <div class="col-md-6">
                <input type="text" class="form-control" id="searchDocuments"
                       placeholder="Search documents...">
            </div>
            <div class="col-md-6">
                <select class="form-select" id="filterByType">
                    <option value="">All file types</option>
                    <option value=".pdf">PDF</option>
                    <option value=".docx">DOCX</option>
                    <option value=".xlsx">XLSX</option>
                    <option value=".txt">TXT</option>
                </select>
            </div>
        </div>

        <div class="card">
            <div class="card-body">
                <div id="documentsTable">
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p>Loading documents...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/documents.js"></script>
</body>
</html>
    """

    return {
        "upload.html": upload_template,
        "upload_status.html": upload_status_template,
        "documents.html": documents_template
    }