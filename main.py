from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging

# Load environment variables first
from dotenv import load_dotenv
import os
load_dotenv()

# Validate required environment variables
def validate_environment():
    """Validate that required environment variables are set"""
    api_key = os.getenv('API_KEY')

    # Validate API key
    if not api_key:
        raise ValueError("API_KEY environment variable is not set. Please check your .env file.")

    if api_key == 'your_api_key_here':
        raise ValueError(
            "API_KEY is still set to the placeholder value. "
            "Please edit your .env file and set API_KEY to your actual API key."
        )

    # Validate optional but recommended variables
    if not os.getenv('DEFAULT_LLM_PROVIDER'):
        logging.warning("DEFAULT_LLM_PROVIDER not set, using default 'openai'")

    if not os.getenv('DEFAULT_EMBEDDING_MODEL'):
        logging.warning("DEFAULT_EMBEDDING_MODEL not set, using default 'text-embedding-3-small'")

    base_url = os.getenv('BASE_URL')
    if base_url:
        logging.info(f"Using custom gateway BASE_URL: {base_url}")
    else:
        logging.info("BASE_URL not set, will use provider-specific URLs (OpenAI/Google direct)")

    logging.info(f"API key loaded successfully (ends with: ...{api_key[-4:]})")

# Validate environment before proceeding
validate_environment()

# Import API modules
from src.api.documents import create_documents_router
from src.api.upload import create_upload_router
from src.api.query import create_query_router
from src.api.system import create_system_router
from src.api.chat import ChatService

# Import service modules
from src.upload.file_service import FileUploadService
from src.rag_factory import create_rag_system
from src.ui.upload_ui import UploadUI
from src.ui.chat_ui import ChatUI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Knowledge Base Search System", version="2.0.0")

# Initialize services (dependency injection)
file_service = FileUploadService()
rag_system = create_rag_system()
chat_service = ChatService(rag_system)

# Initialize UI components
upload_ui = UploadUI()
chat_ui = ChatUI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
templates = Jinja2Templates(directory="src/ui/templates")

# Create API routers with dependencies
documents_router = create_documents_router(rag_system)
upload_router = create_upload_router(file_service, rag_system)
query_router = create_query_router(rag_system)
system_router = create_system_router(rag_system, file_service)

# Root route - Chat Interface as main page (must be first)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "page_title": "Knowledge Base Search"}
    )

# Admin Panel route (handles both upload and document management)
@app.get("/upload", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel for document upload and management"""
    import time
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "page_title": "Admin Panel", "timestamp": int(time.time())}
    )

# Document viewing endpoint
@app.get("/document/{document_id}", response_class=HTMLResponse)
async def view_document(request: Request, document_id: str):
    """View original document content for verification"""
    try:
        # Try to get original document content first
        doc_data = await rag_system.get_original_document(document_id)

        if not doc_data:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Document not found", "page_title": "Error"}
            )

        # doc_data is a list of chunks, so directly use chunks view
        chunks = doc_data if isinstance(doc_data, list) else await rag_system.get_document_chunks(document_id)
        doc_metadata = chunks[0]["metadata"] if chunks else {}

        return templates.TemplateResponse(
            "document_view.html",
            {
                "request": request,
                "document_id": document_id,
                "filename": doc_metadata.get("filename", "Unknown"),
                "chunks": chunks,
                "page_title": f"Document: {doc_metadata.get('filename', 'Unknown')}"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e), "page_title": "Error"}
        )

# Include API routers with prefix
app.include_router(documents_router, prefix="/api", tags=["documents"])
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(query_router, prefix="/api", tags=["query"])
app.include_router(system_router, prefix="/api", tags=["system"])

# Include UI routers
# app.include_router(upload_ui.get_router(), tags=["ui"])  # Temporarily commented out
app.include_router(chat_ui.get_router(), tags=["ui"])

# Connect chat UI to chat service
chat_ui.process_query = chat_service.process_query

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)