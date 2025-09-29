// Chat functionality with WebSocket
let socket = null;
let messageCount = 0;
let sessionStart = new Date();

document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
    updateSessionInfo();
    updateChatModeInfo();
});

function initializeChat() {
    connectWebSocket();

    const messageInput = document.getElementById('messageInput');
    messageInput.addEventListener('keypress', handleKeyPress);
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

    socket = new WebSocket(wsUrl);

    socket.onopen = function(event) {
        updateConnectionStatus('connected');
        console.log('WebSocket connected');
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleIncomingMessage(data);
    };

    socket.onclose = function(event) {
        updateConnectionStatus('disconnected');
        console.log('WebSocket disconnected');

        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = function(error) {
        updateConnectionStatus('error');
        console.error('WebSocket error:', error);
    };
}

function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connectionStatus');

    switch(status) {
        case 'connected':
            statusElement.innerHTML = '<i class="bi bi-wifi"></i> Connected';
            statusElement.className = 'text-success';
            break;
        case 'disconnected':
            statusElement.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
            statusElement.className = 'text-danger';
            break;
        case 'error':
            statusElement.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Error';
            statusElement.className = 'text-warning';
            break;
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();

    if (!message) return;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        showAlert('Not connected to chat server. Please wait for reconnection.', 'warning');
        return;
    }

    // Add user message to chat
    addMessageToChat(message, 'user');

    // Show typing indicator
    showTypingIndicator();

    // Get current query mode (settings are managed in Admin Panel)
    const mode = document.getElementById('chatQueryMode').value;

    // Send message via WebSocket with query mode information
    const payload = {
        type: 'query',
        content: message,
        mode: mode,
        timestamp: Date.now()
    };

    socket.send(JSON.stringify(payload));

    // Clear input
    messageInput.value = '';

    // Update message count
    messageCount++;
    updateSessionInfo();
}

function sendPredefinedMessage(message) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = message;
    sendMessage();
}

function handleIncomingMessage(data) {
    hideTypingIndicator();

    switch(data.type) {
        case 'response':
            addMessageToChat(data.content, 'assistant', data.sources);
            break;
        case 'error':
            addMessageToChat(`Error: ${data.content}`, 'assistant', null, 'error');
            break;
        default:
            console.warn('Unknown message type:', data.type);
    }
}

function addMessageToChat(content, sender, sources = null, messageType = 'normal') {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    let messageHtml = `<div>${content}</div>`;

    // Add sources if provided
    if (sources && sources.length > 0) {
        messageHtml += '<div class="source-refs">';
        messageHtml += '<small><strong>📚 Sources:</strong></small><br>';
        sources.forEach(source => {
            const relevanceScore = Math.round(source.relevance_score || 0);
            const chunkInfo = source.chunk_count > 1 ? ` (${source.chunk_count} chunks)` : '';
            messageHtml += `<a href="/document/${source.document_id}" target="_blank" class="source-ref" style="text-decoration: none; color: inherit;">`;
            messageHtml += `<i class="bi bi-file-text"></i> ${source.filename} (${relevanceScore}% relevance${chunkInfo})`;
            messageHtml += `</a>`;
        });
        messageHtml += '</div>';
    }

    // Add timestamp
    const timestamp = new Date().toLocaleTimeString();
    messageHtml += `<div class="message-meta">${sender === 'user' ? 'You' : 'Assistant'} • ${timestamp}</div>`;

    if (messageType === 'error') {
        messageDiv.classList.add('border-danger');
    }

    messageDiv.innerHTML = messageHtml;

    // Insert before typing indicator
    const typingIndicator = document.getElementById('typingIndicator');
    chatContainer.insertBefore(messageDiv, typingIndicator);

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    typingIndicator.style.display = 'block';

    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    typingIndicator.style.display = 'none';
}

function clearChat() {
    const chatContainer = document.getElementById('chatContainer');
    const messages = chatContainer.querySelectorAll('.message');

    // Remove all messages except the welcome message
    messages.forEach((message, index) => {
        if (index > 0) { // Keep the first welcome message
            message.remove();
        }
    });

    messageCount = 0;
    updateSessionInfo();
}

function updateSessionInfo() {
    document.getElementById('messageCount').textContent = messageCount;
    document.getElementById('sessionStart').textContent = sessionStart.toLocaleTimeString();
}

function showAlert(message, type = 'info') {
    // You can implement a toast notification system here
    console.log(`${type.toUpperCase()}: ${message}`);

    // For now, just show in chat as a system message
    addMessageToChat(`System: ${message}`, 'assistant', null, type === 'warning' ? 'error' : 'normal');
}

// Auto-reconnection logic
setInterval(function() {
    if (!socket || socket.readyState === WebSocket.CLOSED) {
        connectWebSocket();
    }
}, 10000); // Check every 10 seconds

// ===== QUERY MODE FUNCTIONS =====

function updateChatModeInfo() {
    const mode = document.getElementById('chatQueryMode').value;
    const description = document.getElementById('chatModeDescription');

    const modeDescriptions = {
        'vector_search': 'Fast document retrieval using similarity search without LLM processing',
        'llm_response': 'Standard RAG with intelligent response generation (recommended)',
        'agentic_rag': 'Agentic RAG with multi-step reasoning and specialized tools'
    };

    description.textContent = modeDescriptions[mode] || 'Unknown mode';
}