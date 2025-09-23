from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
import json
from typing import Dict, List, Union, Any
import asyncio

logger = logging.getLogger(__name__)

class ChatUI:
    """UI components for chatbot functionality"""

    def __init__(self, templates_dir: str = "src/ui/templates"):
        self.templates = Jinja2Templates(directory=templates_dir)
        self.router = APIRouter()
        self.active_connections: List[WebSocket] = []
        self._setup_routes()

    def _setup_routes(self):
        """Setup chat UI routes"""

        @self.router.get("/chat", response_class=HTMLResponse)
        async def chat_page(request: Request):
            """Display chat interface"""
            return self.templates.TemplateResponse(
                "chat.html",
                {"request": request, "page_title": "Chat with Documents"}
            )

        @self.router.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time chat"""
            await self.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    await self.handle_message(websocket, message_data)
            except WebSocketDisconnect:
                self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection established")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")

    async def handle_message(self, websocket: WebSocket, message_data: Dict):
        """Handle incoming chat message"""
        try:
            message_type = message_data.get("type", "")
            content = message_data.get("content", "")

            if message_type == "query":
                # This will be connected to the RAG system
                response_data = await self.process_query(content)

                # Handle both string responses (fallback) and dict responses (with sources)
                if isinstance(response_data, dict):
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": response_data.get("response", ""),
                        "sources": response_data.get("sources", []),
                        "timestamp": asyncio.get_event_loop().time()
                    }))
                else:
                    # Fallback for string responses
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": response_data,
                        "sources": [],
                        "timestamp": asyncio.get_event_loop().time()
                    }))

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Error processing message: {str(e)}"
            }))

    async def process_query(self, query: str) -> Union[str, Dict[str, Any]]:
        """Process user query (to be connected with RAG system)"""
        # Placeholder - this will be connected to the actual RAG system
        return f"Echo: {query} (This will be connected to the RAG system)"

    def get_router(self):
        """Get the FastAPI router for chat UI routes"""
        return self.router

def create_chat_templates():
    """Create HTML templates for chat UI"""

    # Main chat interface template
    chat_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }} - RAG System</title>
    <link href="/static/css/style.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        .chat-container {
            height: 60vh;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            background-color: #f8f9fa;
        }

        .message {
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-radius: 0.5rem;
            max-width: 80%;
        }

        .message.user {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }

        .message.assistant {
            background-color: white;
            border: 1px solid #dee2e6;
            margin-right: auto;
        }

        .message-meta {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 0.25rem;
        }

        .typing-indicator {
            display: none;
            padding: 0.75rem;
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
            margin-right: auto;
            max-width: 80%;
        }

        .typing-dots {
            display: inline-block;
        }

        .typing-dots span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #007bff;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }

        .typing-dots span:nth-child(1) { animation-delay: 0.0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-10px); opacity: 1; }
        }

        .chat-input-container {
            margin-top: 1rem;
        }

        .source-refs {
            font-size: 0.85rem;
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #dee2e6;
        }

        .source-ref {
            display: inline-block;
            background-color: #e9ecef;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem 0.25rem 0 0;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">RAG System</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/upload">Upload</a>
                <a class="nav-link active" href="/chat">Chat</a>
                <a class="nav-link" href="/documents">Documents</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-8">
                <h2>Chat with Your Documents</h2>
                <p class="text-muted">Ask questions about your uploaded documents and get intelligent answers.</p>

                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div>👋 Hello! I'm ready to help you find information in your documents. What would you like to know?</div>
                        <div class="message-meta">Assistant</div>
                    </div>

                    <div class="typing-indicator" id="typingIndicator">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <div style="margin-left: 10px; display: inline;">Assistant is typing...</div>
                    </div>
                </div>

                <div class="chat-input-container">
                    <div class="input-group">
                        <input type="text" class="form-control" id="messageInput"
                               placeholder="Type your question here..."
                               onkeypress="handleKeyPress(event)">
                        <button class="btn btn-primary" onclick="sendMessage()" id="sendButton">
                            <i class="bi bi-send"></i> Send
                        </button>
                    </div>
                </div>

                <div class="mt-3">
                    <small class="text-muted">
                        💡 Tip: Be specific in your questions for better results. You can ask about content,
                        summaries, comparisons, or specific data from your documents.
                    </small>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h6>Quick Actions</h6>
                    </div>
                    <div class="card-body">
                        <button class="btn btn-outline-primary btn-sm mb-2 w-100"
                                onclick="sendPredefinedMessage('Summarize the main points from all documents')">
                            📄 Summarize Documents
                        </button>
                        <button class="btn btn-outline-primary btn-sm mb-2 w-100"
                                onclick="sendPredefinedMessage('What are the key findings?')">
                            🔍 Key Findings
                        </button>
                        <button class="btn btn-outline-primary btn-sm mb-2 w-100"
                                onclick="sendPredefinedMessage('List all important dates mentioned')">
                            📅 Important Dates
                        </button>
                        <button class="btn btn-outline-secondary btn-sm w-100"
                                onclick="clearChat()">
                            🗑️ Clear Chat
                        </button>
                    </div>
                </div>

                <div class="card mt-3">
                    <div class="card-header">
                        <h6>Connection Status</h6>
                    </div>
                    <div class="card-body">
                        <div id="connectionStatus" class="text-warning">
                            <i class="bi bi-wifi-off"></i> Connecting...
                        </div>
                    </div>
                </div>

                <div class="card mt-3">
                    <div class="card-header">
                        <h6>Chat Statistics</h6>
                    </div>
                    <div class="card-body">
                        <small>
                            <div>Messages sent: <span id="messageCount">0</span></div>
                            <div>Session started: <span id="sessionStart"></span></div>
                        </small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/chat.js"></script>
</body>
</html>
    """

    return {
        "chat.html": chat_template
    }