// Document management functionality
document.addEventListener('DOMContentLoaded', function() {
    loadDocuments();
    setupEventListeners();
});

function setupEventListeners() {
    const searchInput = document.getElementById('searchDocuments');
    const filterSelect = document.getElementById('filterByType');

    searchInput.addEventListener('input', debounce(filterDocuments, 300));
    filterSelect.addEventListener('change', filterDocuments);
}

function loadDocuments() {
    const documentsTable = document.getElementById('documentsTable');

    fetch('/api/documents/')
        .then(response => response.json())
        .then(data => {
            if (data.documents) {
                displayDocuments(data.documents);
            } else {
                showError('Failed to load documents');
            }
        })
        .catch(error => {
            console.error('Error loading documents:', error);
            showError('Failed to load documents due to network error');
        });
}

function displayDocuments(documents) {
    const documentsTable = document.getElementById('documentsTable');

    if (documents.length === 0) {
        documentsTable.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="bi bi-inbox" style="font-size: 3rem;"></i>
                <h5 class="mt-3">No documents found</h5>
                <p>Upload some documents to get started!</p>
                <a href="/upload" class="btn btn-primary">Upload Documents</a>
            </div>
        `;
        return;
    }

    let html = `
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Type</th>
                        <th>Uploaded</th>
                        <th>Chunks</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    `;

    documents.forEach(doc => {
        const uploadDate = new Date(doc.timestamp).toLocaleDateString();
        const fileIcon = getFileIcon(doc.document_type);

        html += `
            <tr data-doc-id="${doc.document_id}" data-filename="${doc.filename}" data-type="${doc.document_type}">
                <td>
                    <i class="bi ${fileIcon} me-2"></i>
                    <span class="document-filename">${doc.filename}</span>
                </td>
                <td>
                    <span class="badge bg-secondary">${doc.document_type.toUpperCase()}</span>
                </td>
                <td>${uploadDate}</td>
                <td>
                    <span class="badge bg-info">${doc.total_chunks || 0} chunks</span>
                </td>
                <td>
                    <div class="btn-group btn-group-sm" role="group">
                        <button class="btn btn-outline-primary" onclick="viewDocument('${doc.document_id}')"
                                title="View document details">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-info" onclick="queryDocument('${doc.document_id}')"
                                title="Query this document">
                            <i class="bi bi-chat-dots"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="deleteDocument('${doc.document_id}', '${doc.filename}')"
                                title="Delete document">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
    `;

    documentsTable.innerHTML = html;
}

function getFileIcon(fileType) {
    const icons = {
        '.pdf': 'bi-file-earmark-pdf-fill text-danger',
        '.docx': 'bi-file-earmark-word-fill text-primary',
        '.xlsx': 'bi-file-earmark-excel-fill text-success',
        '.xls': 'bi-file-earmark-excel-fill text-success',
        '.txt': 'bi-file-earmark-text-fill text-secondary'
    };

    return icons[fileType] || 'bi-file-earmark text-secondary';
}

function filterDocuments() {
    const searchTerm = document.getElementById('searchDocuments').value.toLowerCase();
    const filterType = document.getElementById('filterByType').value;

    const rows = document.querySelectorAll('#documentsTable tbody tr');

    rows.forEach(row => {
        const filename = row.getAttribute('data-filename').toLowerCase();
        const type = row.getAttribute('data-type');

        const matchesSearch = filename.includes(searchTerm);
        const matchesType = !filterType || type === filterType;

        if (matchesSearch && matchesType) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

function refreshDocuments() {
    const documentsTable = document.getElementById('documentsTable');
    documentsTable.innerHTML = `
        <div class="text-center">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Refreshing documents...</p>
        </div>
    `;

    loadDocuments();
}

function viewDocument(documentId) {
    // Open modal or navigate to document details page
    fetch(`/api/documents/${documentId}/chunks`)
        .then(response => response.json())
        .then(data => {
            showDocumentModal(data);
        })
        .catch(error => {
            console.error('Error loading document details:', error);
            showAlert('Failed to load document details', 'danger');
        });
}

function queryDocument(documentId) {
    // Redirect to chat with pre-filled document filter
    window.location.href = `/chat?doc_id=${documentId}`;
}

function deleteDocument(documentId, filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
        return;
    }

    fetch(`/api/documents/${documentId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            showAlert(`Successfully deleted "${filename}"`, 'success');
            loadDocuments(); // Refresh the list
        } else {
            showAlert('Failed to delete document', 'danger');
        }
    })
    .catch(error => {
        console.error('Error deleting document:', error);
        showAlert('Failed to delete document due to network error', 'danger');
    });
}

function showDocumentModal(documentData) {
    // Create and show a modal with document details
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Document Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <h6>Chunks (${documentData.chunks ? documentData.chunks.length : 0})</h6>
                    <div class="accordion" id="chunksAccordion">
                        ${documentData.chunks ? documentData.chunks.map((chunk, index) => `
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="heading${index}">
                                    <button class="accordion-button collapsed" type="button"
                                            data-bs-toggle="collapse" data-bs-target="#collapse${index}">
                                        Chunk ${index + 1}
                                    </button>
                                </h2>
                                <div id="collapse${index}" class="accordion-collapse collapse"
                                     data-bs-parent="#chunksAccordion">
                                    <div class="accordion-body">
                                        <small class="text-muted">
                                            Words: ${chunk.metadata.start_word}-${chunk.metadata.end_word}
                                        </small>
                                        <div class="mt-2">${chunk.content}</div>
                                    </div>
                                </div>
                            </div>
                        `).join('') : '<p>No chunks available</p>'}
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();

    // Clean up modal after it's hidden
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}

function showError(message) {
    const documentsTable = document.getElementById('documentsTable');
    documentsTable.innerHTML = `
        <div class="alert alert-danger text-center">
            <i class="bi bi-exclamation-triangle me-2"></i>
            ${message}
        </div>
    `;
}

function showAlert(message, type = 'info') {
    // Create a toast notification
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

// Debounce utility function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}