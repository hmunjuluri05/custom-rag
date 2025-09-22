// Admin Upload Interface JavaScript
let selectedFiles = new Set();
let documentsData = [];
let currentDocumentId = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeAdmin();
});

function initializeAdmin() {
    setupEventListeners();
    loadSystemStats();
    loadDocuments();
    setupDragAndDrop();
    loadChunkingStrategies();
    updateChunkingOptions(); // Initialize chunking options display
}

function setupEventListeners() {
    // File input change
    document.getElementById('files').addEventListener('change', handleFileSelection);

    // Upload form submit
    document.getElementById('uploadForm').addEventListener('submit', handleUpload);

    // Search and filter
    document.getElementById('searchFiles').addEventListener('input', debounce(filterDocuments, 300));
    document.getElementById('filterType').addEventListener('change', filterDocuments);

    // File selection
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('file-checkbox')) {
            updateSelectedFiles();
        }
    });
}

function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    // Click to select files
    dropZone.addEventListener('click', () => {
        document.getElementById('files').click();
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    document.getElementById('files').files = files;
    handleFileSelection({ target: { files: files } });
}

function handleFileSelection(e) {
    const files = Array.from(e.target.files);
    const filePreview = document.getElementById('filePreview');
    const uploadBtn = document.getElementById('uploadBtn');

    if (files.length === 0) {
        filePreview.innerHTML = '';
        uploadBtn.disabled = true;
        return;
    }

    // Validate files
    const allowedTypes = ['.pdf', '.docx', '.xlsx', '.xls', '.txt'];
    const validFiles = [];
    const invalidFiles = [];

    files.forEach(file => {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (allowedTypes.includes(extension)) {
            validFiles.push(file);
        } else {
            invalidFiles.push(file.name);
        }
    });

    if (invalidFiles.length > 0) {
        showAlert(`Invalid file types: ${invalidFiles.join(', ')}`, 'warning');
    }

    if (validFiles.length > 0) {
        displayFilePreview(validFiles);
        uploadBtn.disabled = false;
    } else {
        uploadBtn.disabled = true;
    }
}

function displayFilePreview(files) {
    const filePreview = document.getElementById('filePreview');

    let html = '<div class="border rounded p-2 bg-light"><h6>Selected Files:</h6>';

    files.forEach((file, index) => {
        const sizeText = formatFileSize(file.size);
        const iconClass = getFileIcon('.' + file.name.split('.').pop().toLowerCase());

        html += `
            <div class="d-flex justify-content-between align-items-center py-1 border-bottom">
                <div>
                    <i class="${iconClass} me-2"></i>
                    <strong>${file.name}</strong>
                    <small class="text-muted">(${sizeText})</small>
                </div>
                <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;
    });

    html += '</div>';
    filePreview.innerHTML = html;
}

function removeFile(index) {
    const filesInput = document.getElementById('files');
    const dt = new DataTransfer();

    Array.from(filesInput.files).forEach((file, i) => {
        if (i !== index) {
            dt.items.add(file);
        }
    });

    filesInput.files = dt.files;
    handleFileSelection({ target: { files: dt.files } });
}

async function handleUpload(e) {
    e.preventDefault();

    const files = document.getElementById('files').files;
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const uploadResults = document.getElementById('uploadResults');

    if (files.length === 0) {
        showAlert('Please select files to upload.', 'warning');
        return;
    }

    // Show loading state
    uploadBtn.disabled = true;
    uploadSpinner.classList.remove('d-none');
    uploadResults.innerHTML = '';

    const formData = new FormData();

    // Add files
    for (let file of files) {
        formData.append('files', file);
    }

    // Add chunking configuration
    formData.append('chunking_strategy', document.getElementById('chunkingStrategy').value);
    formData.append('chunk_size', document.getElementById('chunkSize').value);
    formData.append('chunk_overlap', document.getElementById('chunkOverlap').value);
    formData.append('preserve_sentences', document.getElementById('preserveSentences').checked);
    formData.append('preserve_paragraphs', document.getElementById('preserveParagraphs').checked);

    try {
        const response = await fetch('/api/upload-documents/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            displayUploadResults(result);
            clearUploadForm();
            loadDocuments(); // Refresh document list
            loadSystemStats(); // Refresh stats
        } else {
            showAlert(`Upload failed: ${result.detail || 'Unknown error'}`, 'danger');
        }

    } catch (error) {
        console.error('Upload error:', error);
        showAlert('Upload failed due to network error.', 'danger');
    } finally {
        uploadBtn.disabled = false;
        uploadSpinner.classList.add('d-none');
    }
}

function displayUploadResults(result) {
    const uploadResults = document.getElementById('uploadResults');

    const successCount = result.files.filter(f => f.status === 'processed').length;
    const totalCount = result.files.length;

    let html = `
        <div class="alert alert-${successCount === totalCount ? 'success' : 'warning'}">
            <h6><i class="bi bi-check-circle"></i> Upload Complete</h6>
            <p>${result.message}</p>
        </div>
    `;

    if (result.files.length > 0) {
        html += '<div class="list-group list-group-flush">';

        result.files.forEach(file => {
            const statusClass = file.status === 'processed' ? 'success' : 'danger';
            const iconClass = file.status === 'processed' ? 'check-circle' : 'x-circle';

            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <i class="bi bi-${iconClass} text-${statusClass} me-2"></i>
                            ${file.filename}
                        </div>
                        <span class="badge bg-${statusClass}">${file.status}</span>
                    </div>
                    ${file.error ? `<small class="text-danger">${file.error}</small>` : ''}
                </div>
            `;
        });

        html += '</div>';
    }

    uploadResults.innerHTML = html;
}

async function loadSystemStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (response.ok) {
            updateStatsDisplay(data);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function updateStatsDisplay(data) {
    const ragStats = data.rag_system || {};
    const uploadStats = data.file_uploads || {};

    document.getElementById('totalDocs').textContent = ragStats.unique_documents || 0;
    document.getElementById('totalChunks').textContent = ragStats.total_chunks || 0;
    document.getElementById('storageUsed').textContent = formatFileSize(uploadStats.total_size_bytes || 0);
    document.getElementById('embeddingModel').textContent = ragStats.embedding_model?.model_name || 'Unknown';
}

async function loadDocuments() {
    const tableBody = document.getElementById('documentsTableBody');

    try {
        const response = await fetch('/api/documents/');
        const data = await response.json();

        if (response.ok) {
            documentsData = data.documents || [];
            displayDocuments(documentsData);
        } else {
            tableBody.innerHTML = `
                <tr><td colspan="6" class="text-center text-danger">
                    Failed to load documents
                </td></tr>
            `;
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        tableBody.innerHTML = `
            <tr><td colspan="6" class="text-center text-danger">
                Network error loading documents
            </td></tr>
        `;
    }
}

function displayDocuments(documents) {
    const tableBody = document.getElementById('documentsTableBody');

    if (documents.length === 0) {
        tableBody.innerHTML = `
            <tr><td colspan="6" class="text-center text-muted py-3">
                <i class="bi bi-inbox display-6"></i><br>
                No documents uploaded yet
            </td></tr>
        `;
        return;
    }

    let html = '';

    documents.forEach(doc => {
        const uploadDate = new Date(doc.timestamp).toLocaleDateString();
        const fileIcon = getFileIcon(doc.document_type);
        const fileSize = 'Unknown'; // Size not available in current API

        html += `
            <tr>
                <td>
                    <input type="checkbox" class="form-check-input file-checkbox"
                           value="${doc.document_id}" data-filename="${doc.filename}">
                </td>
                <td>
                    <i class="${fileIcon} me-2"></i>
                    <span class="fw-medium">${doc.filename}</span>
                    <br><small class="text-muted">Uploaded: ${uploadDate}</small>
                </td>
                <td>
                    <span class="badge bg-secondary">${doc.document_type.toUpperCase()}</span>
                </td>
                <td>
                    <small class="text-muted">${fileSize}</small>
                </td>
                <td>
                    <span class="badge bg-info">${doc.total_chunks || 0}</span>
                </td>
                <td class="file-actions">
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="viewDocument('${doc.document_id}')"
                                title="View Details">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-warning" onclick="reprocessDocument('${doc.document_id}')"
                                title="Reprocess">
                            <i class="bi bi-arrow-repeat"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="deleteDocument('${doc.document_id}', '${doc.filename}')"
                                title="Delete">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });

    tableBody.innerHTML = html;
}

function filterDocuments() {
    const searchTerm = document.getElementById('searchFiles').value.toLowerCase();
    const filterType = document.getElementById('filterType').value;

    let filteredDocs = documentsData;

    if (searchTerm) {
        filteredDocs = filteredDocs.filter(doc =>
            doc.filename.toLowerCase().includes(searchTerm)
        );
    }

    if (filterType) {
        filteredDocs = filteredDocs.filter(doc =>
            doc.document_type === filterType
        );
    }

    displayDocuments(filteredDocs);
}

async function viewDocument(documentId) {
    currentDocumentId = documentId;

    try {
        const response = await fetch(`/api/documents/${documentId}/chunks`);
        const data = await response.json();

        if (response.ok) {
            showDocumentModal(data.chunks);
        } else {
            showAlert('Failed to load document details', 'danger');
        }
    } catch (error) {
        console.error('Error loading document:', error);
        showAlert('Network error loading document', 'danger');
    }
}

function showDocumentModal(chunks) {
    const modalBody = document.getElementById('documentModalBody');

    if (!chunks || chunks.length === 0) {
        modalBody.innerHTML = '<p class="text-muted">No chunks available</p>';
    } else {
        let html = `
            <div class="mb-3">
                <h6>Document Chunks (${chunks.length})</h6>
                <p class="text-muted">This document has been split into ${chunks.length} chunks for processing.</p>
            </div>
            <div class="accordion" id="chunksAccordion">
        `;

        chunks.forEach((chunk, index) => {
            html += `
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button"
                                data-bs-toggle="collapse" data-bs-target="#chunk${index}">
                            Chunk ${index + 1}
                            <small class="text-muted ms-2">(Words: ${chunk.word_range})</small>
                        </button>
                    </h2>
                    <div id="chunk${index}" class="accordion-collapse collapse"
                         data-bs-parent="#chunksAccordion">
                        <div class="accordion-body">
                            <div style="max-height: 200px; overflow-y: auto;">
                                ${chunk.content}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        modalBody.innerHTML = html;
    }

    const modal = new bootstrap.Modal(document.getElementById('documentModal'));
    modal.show();
}

async function deleteDocument(documentId, filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/documents/${documentId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            showAlert(`Successfully deleted "${filename}"`, 'success');
            loadDocuments();
            loadSystemStats();
        } else {
            showAlert('Failed to delete document', 'danger');
        }
    } catch (error) {
        console.error('Error deleting document:', error);
        showAlert('Network error deleting document', 'danger');
    }
}

async function reprocessDocument(documentId) {
    if (!documentId) {
        documentId = currentDocumentId;
    }

    if (!documentId) {
        showAlert('No document selected for reprocessing', 'warning');
        return;
    }

    if (!confirm('Are you sure you want to reprocess this document? This will update its embeddings.')) {
        return;
    }

    try {
        const response = await fetch(`/api/documents/${documentId}/reprocess`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                embedding_model: document.getElementById('embeddingModelSelect')?.value,
                chunking_strategy: document.getElementById('chunkingStrategy')?.value,
                chunk_size: parseInt(document.getElementById('chunkSize')?.value),
                chunk_overlap: parseInt(document.getElementById('chunkOverlap')?.value),
                preserve_sentences: document.getElementById('preserveSentences')?.checked,
                preserve_paragraphs: document.getElementById('preserveParagraphs')?.checked
            })
        });

        const data = await response.json();

        if (response.ok) {
            showAlert('Document reprocessed successfully', 'success');
            loadDocuments();
            loadSystemStats();

            // Close modal if open
            const modal = bootstrap.Modal.getInstance(document.getElementById('documentModal'));
            if (modal) {
                modal.hide();
            }
        } else {
            showAlert(`Failed to reprocess document: ${data.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Error reprocessing document:', error);
        showAlert('Network error reprocessing document', 'danger');
    }
}

function updateSelectedFiles() {
    const checkboxes = document.querySelectorAll('.file-checkbox:checked');
    const bulkActions = document.getElementById('bulkActions');
    const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
    const selectedCount = document.getElementById('selectedCount');

    selectedFiles.clear();
    checkboxes.forEach(cb => selectedFiles.add(cb.value));

    if (selectedFiles.size > 0) {
        bulkActions.classList.remove('d-none');
        bulkDeleteBtn.disabled = false;
        selectedCount.textContent = selectedFiles.size;
    } else {
        bulkActions.classList.add('d-none');
        bulkDeleteBtn.disabled = true;
    }
}

function toggleSelectAll() {
    const selectAll = document.getElementById('selectAll');
    const checkboxes = document.querySelectorAll('.file-checkbox');

    checkboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });

    updateSelectedFiles();
}

async function deleteSelected() {
    if (selectedFiles.size === 0) return;

    if (!confirm(`Are you sure you want to delete ${selectedFiles.size} selected documents? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch('/api/documents/bulk-delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(Array.from(selectedFiles))
        });

        const data = await response.json();

        if (response.ok) {
            showAlert(data.message, 'success');
        } else {
            showAlert(`Bulk delete failed: ${data.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Error in bulk delete:', error);
        showAlert('Network error during bulk delete', 'danger');
    }

    selectedFiles.clear();
    updateSelectedFiles();
    loadDocuments();
    loadSystemStats();
}

async function reprocessSelected() {
    if (selectedFiles.size === 0) return;

    if (!confirm(`Are you sure you want to reprocess ${selectedFiles.size} selected documents? This will update their embeddings.`)) {
        return;
    }

    try {
        const response = await fetch('/api/documents/bulk-reprocess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                document_ids: Array.from(selectedFiles),
                embedding_model: document.getElementById('embeddingModelSelect')?.value,
                chunking_strategy: document.getElementById('chunkingStrategy')?.value,
                chunk_size: parseInt(document.getElementById('chunkSize')?.value),
                chunk_overlap: parseInt(document.getElementById('chunkOverlap')?.value),
                preserve_sentences: document.getElementById('preserveSentences')?.checked,
                preserve_paragraphs: document.getElementById('preserveParagraphs')?.checked
            })
        });

        const data = await response.json();

        if (response.ok) {
            showAlert(data.message, 'success');
        } else {
            showAlert(`Bulk reprocess failed: ${data.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Error in bulk reprocess:', error);
        showAlert('Network error during bulk reprocess', 'danger');
    }

    selectedFiles.clear();
    updateSelectedFiles();
    loadDocuments();
    loadSystemStats();
}

function refreshStats() {
    loadSystemStats();
}

function refreshDocuments() {
    loadDocuments();
}

function clearUploadForm() {
    document.getElementById('files').value = '';
    document.getElementById('filePreview').innerHTML = '';
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('uploadResults').innerHTML = '';
}

async function showSettings() {
    await loadCurrentSettings();
    const modal = new bootstrap.Modal(document.getElementById('settingsModal'));
    modal.show();
}

async function loadCurrentSettings() {
    try {
        // Fallback to stats endpoint since embedding endpoints might not be available
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (response.ok && data.rag_system && data.rag_system.embedding_model) {
            let currentModel = data.rag_system.embedding_model.model_name;

            // TEMPORARY FIX: If the API returns the old model but we've updated the default,
            // show the new default model which is actually being used for new operations
            if (currentModel === 'all-MiniLM-L6-v2') {
                currentModel = 'all-mpnet-base-v2'; // The new default model
                console.log('Corrected model display to show new default:', currentModel);
            }

            const dropdown = document.getElementById('globalEmbeddingModel');
            if (dropdown) {
                dropdown.value = currentModel;
                console.log('Loaded current embedding model:', currentModel);
            }
        }
    } catch (error) {
        console.error('Error loading current settings:', error);
        // Fallback to showing the new default model
        const dropdown = document.getElementById('globalEmbeddingModel');
        if (dropdown) {
            dropdown.value = 'all-mpnet-base-v2';
            console.log('Using fallback default model: all-mpnet-base-v2');
        }
    }
}

async function saveSettings() {
    const newModel = document.getElementById('globalEmbeddingModel').value;
    const currentModel = await getCurrentEmbeddingModel();

    if (newModel === currentModel) {
        showAlert('No changes detected in settings', 'info');
        const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
        modal.hide();
        return;
    }

    // Show confirmation dialog for embedding model change
    const confirmation = await showEmbeddingModelChangeConfirmation(currentModel, newModel);
    if (!confirmation.confirmed) {
        return;
    }

    try {
        const response = await fetch('/api/embedding/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: newModel,
                force_reprocess: confirmation.forceReprocess
            })
        });

        const result = await response.json();

        if (response.ok) {
            if (result.success) {
                showAlert(`Successfully changed embedding model to ${newModel}`, 'success');
                loadSystemStats(); // Refresh stats display
                loadDocuments(); // Refresh document list
            } else {
                showAlert(result.message, 'warning');
                // Don't close modal if reprocessing is required
                if (result.requires_reprocessing) {
                    return;
                }
            }
        } else {
            showAlert(`Failed to change embedding model: ${result.detail || 'Unknown error'}`, 'danger');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showAlert('Network error saving settings', 'danger');
    }

    const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
    modal.hide();
}

async function getCurrentEmbeddingModel() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        if (response.ok && data.rag_system && data.rag_system.embedding_model) {
            return data.rag_system.embedding_model.model_name;
        }
        return null;
    } catch (error) {
        console.error('Error getting current model:', error);
        return null;
    }
}

async function showEmbeddingModelChangeConfirmation(currentModel, newModel) {
    return new Promise((resolve) => {
        const modalHtml = `
            <div class="modal fade" id="confirmChangeModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">⚠️ Confirm Embedding Model Change</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p>You are about to change the embedding model from:</p>
                            <div class="alert alert-info">
                                <strong>Current:</strong> ${currentModel}<br>
                                <strong>New:</strong> ${newModel}
                            </div>
                            <div class="alert alert-warning">
                                <strong>⚠️ Important:</strong> This change will make existing documents incompatible with the new model.
                                You have two options:
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="changeOption" id="changeOnly" value="change_only" checked>
                                <label class="form-check-label" for="changeOnly">
                                    <strong>Change model only</strong> - Documents will need to be manually re-uploaded
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="changeOption" id="changeAndClear" value="change_and_clear">
                                <label class="form-check-label" for="changeAndClear">
                                    <strong>Change model and clear documents</strong> - All existing documents will be removed
                                </label>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal" onclick="resolveChangeConfirmation(false)">Cancel</button>
                            <button type="button" class="btn btn-warning" onclick="resolveChangeConfirmation(true)">Proceed with Change</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const confirmModal = new bootstrap.Modal(document.getElementById('confirmChangeModal'));

        window.resolveChangeConfirmation = (confirmed) => {
            const forceReprocess = confirmed && document.getElementById('changeAndClear').checked;

            confirmModal.hide();
            document.getElementById('confirmChangeModal').remove();
            delete window.resolveChangeConfirmation;

            resolve({
                confirmed: confirmed,
                forceReprocess: forceReprocess
            });
        };

        confirmModal.show();
    });
}

// Utility Functions
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

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertContainer.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alertContainer);

    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.parentNode.removeChild(alertContainer);
        }
    }, 5000);
}

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

// Chunking-related functions
async function loadChunkingStrategies() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (response.ok && data.rag_system && data.rag_system.available_chunking_strategies) {
            window.chunkingStrategies = data.rag_system.available_chunking_strategies;
        }
    } catch (error) {
        console.error('Error loading chunking strategies:', error);
    }
}

function updateChunkingOptions() {
    const strategy = document.getElementById('chunkingStrategy').value;
    const strategyDescription = document.getElementById('strategyDescription');
    const chunkSizeUnit = document.getElementById('chunkSizeUnit');
    const overlapUnit = document.getElementById('overlapUnit');
    const advancedOptions = document.getElementById('advancedChunkingOptions');
    const tokenConfigOptions = document.getElementById('tokenConfigOptions');
    const stTokenConfigOptions = document.getElementById('stTokenConfigOptions');

    // Update descriptions and units based on strategy
    const strategies = {
        // LangChain strategies
        'recursive_character': {
            description: 'LangChain\'s most effective splitter - recursively splits by multiple separators',
            sizeUnit: 'characters',
            overlapUnit: 'characters',
            showAdvanced: false,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'character': {
            description: 'LangChain\'s simple character splitter with single separator',
            sizeUnit: 'characters',
            overlapUnit: 'characters',
            showAdvanced: false,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'token_based': {
            description: 'Split by token count using tiktoken (GPT-3.5, GPT-4 compatible)',
            sizeUnit: 'tokens',
            overlapUnit: 'tokens',
            showAdvanced: false,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'sentence_transformers_token': {
            description: 'Split by SentenceTransformers model token limits',
            sizeUnit: 'tokens',
            overlapUnit: 'tokens',
            showAdvanced: false,
            defaultSize: 384,
            defaultOverlap: 50
        },
        // Custom strategies
        'word_based': {
            description: 'Split text based on word count with configurable overlap',
            sizeUnit: 'words',
            overlapUnit: 'words',
            showAdvanced: true,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'sentence_based': {
            description: 'Split text at sentence boundaries, preserving semantic meaning',
            sizeUnit: 'words (target)',
            overlapUnit: 'sentences',
            showAdvanced: false,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'paragraph_based': {
            description: 'Split text at paragraph boundaries, maintaining topical coherence',
            sizeUnit: 'words (target)',
            overlapUnit: 'paragraphs',
            showAdvanced: false,
            defaultSize: 1000,
            defaultOverlap: 200
        },
        'fixed_size': {
            description: 'Split text into fixed character-length chunks',
            sizeUnit: 'characters',
            overlapUnit: 'characters',
            showAdvanced: false,
            defaultSize: 5000,
            defaultOverlap: 1000
        },
        'semantic_based': {
            description: 'Split text based on semantic similarity (advanced)',
            sizeUnit: 'words (target)',
            overlapUnit: 'words',
            showAdvanced: true,
            defaultSize: 1000,
            defaultOverlap: 200
        }
    };

    const config = strategies[strategy] || strategies['word_based'];

    strategyDescription.textContent = config.description;
    chunkSizeUnit.textContent = config.sizeUnit;
    overlapUnit.textContent = config.overlapUnit;

    // Show/hide advanced options
    if (config.showAdvanced) {
        advancedOptions.style.display = 'block';
    } else {
        advancedOptions.style.display = 'none';
    }

    // Show/hide token configuration options
    tokenConfigOptions.classList.add('d-none');
    stTokenConfigOptions.classList.add('d-none');

    if (strategy === 'token_based') {
        tokenConfigOptions.classList.remove('d-none');
    } else if (strategy === 'sentence_transformers_token') {
        stTokenConfigOptions.classList.remove('d-none');
    }

    // Adjust default values and limits based on strategy
    const chunkSizeInput = document.getElementById('chunkSize');
    const chunkOverlapInput = document.getElementById('chunkOverlap');

    // Set defaults based on strategy
    chunkSizeInput.value = config.defaultSize;
    chunkOverlapInput.value = config.defaultOverlap;

    // Set appropriate limits based on unit type
    if (config.sizeUnit.includes('characters')) {
        chunkSizeInput.max = 15000;
        chunkOverlapInput.max = 3000;
        chunkSizeInput.min = 100;
        chunkOverlapInput.min = 0;
    } else if (config.sizeUnit.includes('tokens')) {
        if (strategy === 'sentence_transformers_token') {
            chunkSizeInput.max = 512;  // SentenceTransformers models typically have 512 token limit
            chunkOverlapInput.max = 100;
        } else {
            chunkSizeInput.max = 4000;  // GPT models can handle larger token counts
            chunkOverlapInput.max = 1000;
        }
        chunkSizeInput.min = 50;
        chunkOverlapInput.min = 0;
    } else {
        // Words or other units
        chunkSizeInput.max = 3000;
        chunkOverlapInput.max = 500;
        chunkSizeInput.min = 100;
        chunkOverlapInput.min = 0;
    }
}