from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging

# Import API modules
from src.api.documents import create_documents_router
from src.api.upload import create_upload_router
from src.api.query import create_query_router
from src.api.system import create_system_router
from src.api.chat import ChatService

# Import service modules
from src.upload.file_service import FileUploadService
from src.rag_system import RAGSystem
from src.ui.upload_ui import UploadUI
from src.ui.chat_ui import ChatUI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Knowledge Base Search System", version="2.0.0")

# Initialize services (dependency injection)
file_service = FileUploadService()
rag_system = RAGSystem()
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
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "page_title": "Admin Panel"}
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