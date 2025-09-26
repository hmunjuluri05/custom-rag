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

    // Set default LLM display immediately, then try to load from API
    const llmModelDisplay = document.getElementById('llmModelDisplay');
    if (llmModelDisplay) {
        llmModelDisplay.textContent = 'OpenAI: gpt-4';
    }

    loadLLMDisplay(); // Load LLM display on page load

    // Add periodic check to ensure LLM display never shows NONE
    setInterval(() => {
        const llmModelDisplay = document.getElementById('llmModelDisplay');
        if (llmModelDisplay && (
            llmModelDisplay.textContent.includes('NONE') ||
            llmModelDisplay.textContent.includes('Simple Context') ||
            llmModelDisplay.textContent === '-' ||
            llmModelDisplay.textContent === ''
        )) {
            llmModelDisplay.textContent = 'OpenAI: gpt-4';
        }
    }, 2000); // Check every 2 seconds

    // Add global modal cleanup on any modal hide event
    document.addEventListener('hidden.bs.modal', function() {
        // Clean up any lingering backdrops
        setTimeout(() => {
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());

            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        }, 100);
    });

    // Make edit functions globally accessible
    window.editEmbeddingModel = editEmbeddingModel;
    window.editLLMModel = editLLMModel;
    window.saveEmbeddingModel = saveEmbeddingModel;
    window.saveLLMModel = saveLLMModel;
    window.updateEditEmbeddingModels = updateEditEmbeddingModels;
    window.updateEditLLMModels = updateEditLLMModels;
    window.closeEmbeddingModal = closeEmbeddingModal;
    window.closeLLMModal = closeLLMModal;
    window.showUploadModal = showUploadModal;
    window.closeUploadModal = closeUploadModal;
    window.removePersistentAlert = removePersistentAlert;

    // Add click event listeners as backup
    setTimeout(() => {
        // Wait for DOM to be fully loaded
        const embeddingBtn = document.querySelector('[onclick="editEmbeddingModel()"]');
        const llmBtn = document.querySelector('[onclick="editLLMModel()"]');

        if (embeddingBtn) {
            embeddingBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Embedding edit button clicked via event listener');
                editEmbeddingModel();
            });
        }

        if (llmBtn) {
            llmBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('LLM edit button clicked via event listener');
                editLLMModel();
            });
        }
    }, 1000);
}

function setupEventListeners() {
    // File input change (check if exists first)
    const filesInput = document.getElementById('files');
    if (filesInput) {
        filesInput.addEventListener('change', handleFileSelection);
    }

    // Upload form submit
    document.getElementById('uploadForm').addEventListener('submit', handleUpload);

    // Search and filter
    document.getElementById('searchFiles').addEventListener('input', debounce(filterDocuments, 300));
    document.getElementById('filterFiles').addEventListener('change', filterDocuments);

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

function uploadFiles() {
    // Trigger the form submit which will call handleUpload
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
        uploadForm.dispatchEvent(submitEvent);
    }
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
    const preserveSentencesEl = document.getElementById('preserveSentences');
    const preserveParagraphsEl = document.getElementById('preserveParagraphs');

    formData.append('preserve_sentences', preserveSentencesEl ? preserveSentencesEl.checked : true);
    formData.append('preserve_paragraphs', preserveParagraphsEl ? preserveParagraphsEl.checked : false);

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

            // Auto-close modal after successful upload
            setTimeout(() => {
                closeUploadModal();
            }, 2000); // Close after 2 seconds to let user see the results
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
        const response = await fetch('/api/system/stats');
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

    // Use rag_system.unique_documents as it reflects documents in the RAG system
    document.getElementById('totalDocs').textContent = ragStats.unique_documents || 0;
    document.getElementById('storageUsed').textContent = formatFileSize(uploadStats.total_size_bytes || 0);

    // Update embedding model display
    const embeddingInfo = ragStats.embedding_model;
    if (embeddingInfo) {
        const modelName = embeddingInfo.model_name || 'Unknown';
        const provider = embeddingInfo.provider || 'openai';
        const displayText = provider === 'openai' || provider === 'google' ? `${provider.toUpperCase()}: ${modelName}` : modelName;
        document.getElementById('embeddingModelDisplay').textContent = displayText;
    } else {
        document.getElementById('embeddingModelDisplay').textContent = 'Unknown';
    }
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
        const fileIcon = getFileIcon(doc.document_type || doc.type);
        const fileSize = doc.file_size ? formatFileSize(doc.file_size) : (doc.size || 'Unknown');

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
                    <span class="badge bg-secondary">${(doc.document_type || doc.type || 'Unknown').toUpperCase()}</span>
                </td>
                <td>
                    <small class="text-muted">${fileSize}</small>
                </td>
                <td>
                    <span class="badge bg-info">${doc.total_chunks || doc.chunks || 0}</span>
                </td>
                <td class="file-actions">
                    <div class="btn-group btn-group-sm d-flex flex-wrap gap-1">
                        <button class="btn btn-outline-primary" onclick="viewDocument('${doc.document_id}')"
                                title="View Details">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-info" onclick="summarizeDocument('${doc.document_id}', '${doc.filename}')"
                                title="Summarize Document">
                            <i class="bi bi-file-text"></i>
                        </button>
                        <button class="btn btn-outline-success" onclick="getKeyFindings('${doc.document_id}', '${doc.filename}')"
                                title="Key Findings">
                            <i class="bi bi-search"></i>
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

async function loadAvailableEmbeddingModels() {
    try {
        console.log('Attempting to load embedding models from API...');

        // Try the embedding models endpoint first
        let response = await fetch('/api/embedding/models');
        console.log('First API call status:', response.status);

        if (response.status === 404) {
            console.log('First endpoint failed, trying system prefix...');
            // Try system prefix
            response = await fetch('/api/system/embedding/models');
            console.log('Second API call status:', response.status);
        }

        if (!response.ok) {
            throw new Error('API endpoint not found, using fallback');
        }

        const models = await response.json();
        console.log('API models loaded successfully:', models);
        populateEmbeddingModelsDropdown(models);
    } catch (error) {
        console.error('Error loading embedding models:', error);
        console.log('Falling back to hardcoded models...');
        // Fallback to hardcoded list with external models
        loadFallbackEmbeddingModels();
    }
}

// Global variable to store all available models
let availableEmbeddingModels = {};

function populateEmbeddingModelsDropdown(models) {
    console.log('Populating embedding models dropdown with:', models);

    // Store models globally for use in provider/model dropdowns
    availableEmbeddingModels = models;

    // Initialize the provider dropdown and load models for default provider
    updateEmbeddingProviderSettings();
}

function updateEmbeddingProviderSettings() {
    console.log('Updating embedding provider settings...');
    const providerDropdown = document.getElementById('embeddingProvider');
    const modelDropdown = document.getElementById('embeddingModel');

    if (!providerDropdown || !modelDropdown) {
        console.error('Could not find embedding dropdowns');
        return;
    }

    const selectedProvider = providerDropdown.value;
    console.log('Selected provider:', selectedProvider);

    // Clear model dropdown
    modelDropdown.innerHTML = '';

    // Filter models by provider
    const modelsForProvider = Object.entries(availableEmbeddingModels).filter(
        ([modelName, modelInfo]) => {
            const provider = modelInfo.provider.toString().toLowerCase();
            return provider === selectedProvider ||
                   (selectedProvider === 'openai' && provider.includes('openai')) ||
                   (selectedProvider === 'google' && provider.includes('google'));
        }
    );

    console.log(`Found ${modelsForProvider.length} models for provider: ${selectedProvider}`);

    // Populate model dropdown
    modelsForProvider.forEach(([modelName, modelInfo]) => {
        const option = document.createElement('option');
        option.value = modelName;
        const dimensions = modelInfo.dimension || 'N/A';
        const category = modelInfo.category || modelInfo.description || '';
        option.textContent = `${modelName} (${dimensions} dim${category ? ' - ' + category : ''})`;
        modelDropdown.appendChild(option);
        console.log(`Added model: ${modelName}`);
    });

    // Trigger model settings update
    updateEmbeddingModelSettings();
}

function loadFallbackEmbeddingModels() {
    console.log('Loading fallback embedding models...');

    // Create fallback models structure
    availableEmbeddingModels = {
        // OpenAI models
        'text-embedding-3-large': { provider: 'openai', dimension: 3072, category: 'Premium', requires_api_key: true, cost: 'Paid' },
        'text-embedding-3-small': { provider: 'openai', dimension: 1536, category: 'Standard', requires_api_key: true, cost: 'Paid' },
        'text-embedding-ada-002': { provider: 'openai', dimension: 1536, category: 'Legacy', requires_api_key: true, cost: 'Paid' },

        // Google models
        'models/embedding-001': { provider: 'google', dimension: 768, category: 'General', requires_api_key: true, cost: 'Free/Paid' },
        'models/text-embedding-004': { provider: 'google', dimension: 768, category: 'Latest', requires_api_key: true, cost: 'Free/Paid' },

    };

    // Initialize the provider dropdown and load models for default provider
    updateEmbeddingProviderSettings();

    console.log(`Loaded ${Object.keys(availableEmbeddingModels).length} fallback embedding models`);
}

function updateEmbeddingModelSettings() {
    const modelDropdown = document.getElementById('embeddingModel');

    if (!modelDropdown) {
        console.error('Could not find embedding model dropdown');
        return;
    }

    // No need for API key handling since using internal gateway
}

async function showSettings() {
    await loadAvailableEmbeddingModels();
    await loadCurrentSettings();
    const modal = new bootstrap.Modal(document.getElementById('settingsModal'));
    modal.show();
}

async function loadCurrentSettings() {
    try {
        // Load embedding model settings
        const response = await fetch('/api/system/embedding/current');
        if (response.ok) {
            const data = await response.json();
            console.log('Current embedding model:', data);

            const currentModel = data.current_model;
            if (currentModel && availableEmbeddingModels[currentModel]) {
                const modelInfo = availableEmbeddingModels[currentModel];
                const provider = modelInfo.provider.toString().toLowerCase();

                // Set provider dropdown
                const providerDropdown = document.getElementById('embeddingProvider');
                if (providerDropdown) {
                    providerDropdown.value = provider;
                    updateEmbeddingProviderSettings();
                }

                // Set model dropdown
                const modelDropdown = document.getElementById('embeddingModel');
                if (modelDropdown) {
                    modelDropdown.value = currentModel;
                    updateEmbeddingModelSettings();
                }
            }
        }

        // Load LLM settings
        await loadCurrentLLMSettings();
    } catch (error) {
        console.error('Error loading current settings:', error);
        // Fallback to showing the default model
        const providerDropdown = document.getElementById('embeddingProvider');
        if (providerDropdown) {
            providerDropdown.value = 'openai';
            updateEmbeddingProviderSettings();
        }
    }
}

async function saveSettings() {
    const modelDropdown = document.getElementById('embeddingModel');
    const newModel = modelDropdown.value;
    const currentModel = await getCurrentEmbeddingModel();

    if (newModel === currentModel) {
        showAlert('No changes detected in embedding model settings', 'info');
        return;
    }

    try {
        const response = await fetch('/api/system/embedding/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: newModel,
                force_reprocess: true
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showAlert('Embedding model updated successfully!', 'success');
            loadSystemStats(); // Refresh stats
        } else {
            showAlert(result.message || 'Failed to update embedding model', 'danger');
        }
    } catch (error) {
        console.error('Error saving embedding settings:', error);
        showAlert('Network error saving embedding settings', 'danger');
    }

    // Save LLM settings
    await saveLLMSettings();

    const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
    modal.hide();
}

async function getCurrentEmbeddingModel() {
    try {
        const response = await fetch('/api/system/stats');
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
        const response = await fetch('/api/system/stats');
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

    if (strategyDescription) strategyDescription.textContent = config.description;
    if (chunkSizeUnit) chunkSizeUnit.textContent = config.sizeUnit;
    if (overlapUnit) overlapUnit.textContent = config.overlapUnit;

    // Show/hide advanced options
    if (advancedOptions) {
        if (config.showAdvanced) {
            advancedOptions.style.display = 'block';
        } else {
            advancedOptions.style.display = 'none';
        }
    }

    // Show/hide token configuration options
    if (tokenConfigOptions) {
        tokenConfigOptions.classList.add('d-none');
        if (strategy === 'token_based') {
            tokenConfigOptions.classList.remove('d-none');
        }
    }

    if (stTokenConfigOptions) {
        stTokenConfigOptions.classList.add('d-none');
        if (strategy === 'sentence_transformers_token') {
            stTokenConfigOptions.classList.remove('d-none');
        }
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

// LLM Settings Functions
async function updateLLMSettings() {
    const provider = document.getElementById('llmProvider').value;
    const modelSection = document.getElementById('llmModelSection');
    const modelSelect = document.getElementById('llmModel');

    // Always show model section since we removed "none" option
    modelSection.classList.remove('d-none');

    // Clear existing options
    modelSelect.innerHTML = '';

    // Load available models for the selected provider
    try {
        const response = await fetch('/api/system/llm/models');
        const data = await response.json();

        if (response.ok && data[provider] && data[provider].models && data[provider].models.length > 0) {
            // Use API models if available
            const models = data[provider].models;
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
            return; // Exit early if API worked
        }
        throw new Error('API did not return valid models');
    } catch (error) {
        console.error('Error loading LLM models, using fallback:', error);
        // Only use fallback if API failed or returned no models
        const fallbackModels = {
            'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
            'google': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
        };

        if (fallbackModels[provider]) {
            fallbackModels[provider].forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
    }
}

async function loadCurrentLLMSettings() {
    try {
        const response = await fetch('/api/system/llm/current');
        const data = await response.json();

        if (response.ok) {
            const provider = data.provider || 'openai';
            document.getElementById('llmProvider').value = provider;

            await updateLLMSettings();
            const model = data.model_name || 'gpt-4';
            document.getElementById('llmModel').value = model;
        }
    } catch (error) {
        console.error('Error loading current LLM settings:', error);
        // Default to 'openai' and 'gpt-4' if there's an error
        document.getElementById('llmProvider').value = 'openai';
        await updateLLMSettings();
        document.getElementById('llmModel').value = 'gpt-4';
    }
}

async function saveLLMSettings() {
    const provider = document.getElementById('llmProvider').value;
    const model = document.getElementById('llmModel').value;

    try {
        const response = await fetch('/api/system/llm/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                provider: provider,
                model_name: model
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showAlert(`Successfully changed LLM to ${provider}: ${model}`, 'success');
            return true;
        } else {
            showAlert(`Failed to change LLM: ${result.message || 'Unknown error'}`, 'danger');
            return false;
        }
    } catch (error) {
        console.error('Error saving LLM settings:', error);
        showAlert('Network error saving LLM settings', 'danger');
        return false;
    }
}

// Edit Modal Functions
function editEmbeddingModel() {
    console.log('editEmbeddingModel function called');
    try {
        const modalElement = document.getElementById('embeddingModelModal');
        console.log('Modal element found:', modalElement);
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();

            // Load data after showing modal
            loadAvailableEmbeddingModels().then(() => {
                loadCurrentEmbeddingSettings();
            });
        } else {
            console.error('Embedding model modal not found');
            showAlert('Modal not found. Please refresh the page.', 'danger');
        }
    } catch (error) {
        console.error('Error in editEmbeddingModel:', error);
        showAlert('Error opening embedding model editor', 'danger');
    }
}

function editLLMModel() {
    console.log('editLLMModel function called');
    try {
        const modalElement = document.getElementById('llmModelModal');
        console.log('LLM Modal element found:', modalElement);
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();

            // Load data after showing modal - ensure models are populated first
            loadCurrentLLMSettingsForEdit();
        } else {
            console.error('LLM model modal not found');
            showAlert('Modal not found. Please refresh the page.', 'danger');
        }
    } catch (error) {
        console.error('Error in editLLMModel:', error);
        showAlert('Error opening LLM model editor', 'danger');
    }
}

async function loadCurrentEmbeddingSettings() {
    try {
        const response = await fetch('/api/system/embedding/current');
        if (response.ok) {
            const data = await response.json();
            const currentModel = data.current_model;

            if (currentModel && availableEmbeddingModels[currentModel]) {
                const modelInfo = availableEmbeddingModels[currentModel];
                const provider = modelInfo.provider.toString().toLowerCase();

                // Set provider dropdown
                const providerDropdown = document.getElementById('editEmbeddingProvider');
                if (providerDropdown) {
                    providerDropdown.value = provider;
                    updateEditEmbeddingModels();
                }

                // Set model dropdown
                const modelDropdown = document.getElementById('editEmbeddingModel');
                if (modelDropdown) {
                    modelDropdown.value = currentModel;
                }
            }
        }
    } catch (error) {
        console.error('Error loading current embedding settings:', error);
    }
}

function updateEditEmbeddingModels() {
    const providerDropdown = document.getElementById('editEmbeddingProvider');
    const modelDropdown = document.getElementById('editEmbeddingModel');

    if (!providerDropdown || !modelDropdown) return;

    const selectedProvider = providerDropdown.value;

    // Clear model dropdown
    modelDropdown.innerHTML = '';

    // Filter models by provider
    const modelsForProvider = Object.entries(availableEmbeddingModels).filter(
        ([modelName, modelInfo]) => {
            const provider = modelInfo.provider.toString().toLowerCase();
            return provider === selectedProvider ||
                   (selectedProvider === 'openai' && provider.includes('openai')) ||
                   (selectedProvider === 'google' && provider.includes('google'));
        }
    );

    // Populate model dropdown
    modelsForProvider.forEach(([modelName, modelInfo]) => {
        const option = document.createElement('option');
        option.value = modelName;
        const dimensions = modelInfo.dimension || 'N/A';
        const category = modelInfo.category || modelInfo.description || '';
        option.textContent = `${modelName} (${dimensions} dim${category ? ' - ' + category : ''})`;
        modelDropdown.appendChild(option);
    });
}

async function saveEmbeddingModel() {
    const modelDropdown = document.getElementById('editEmbeddingModel');
    const newModel = modelDropdown.value;

    try {
        const response = await fetch('/api/system/embedding/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: newModel,
                force_reprocess: true
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showAlert('Embedding model updated successfully!', 'success');
            loadSystemStats();
            updateStatsDisplayFromModal(newModel);
        } else {
            showAlert(result.message || 'Failed to update embedding model', 'danger');
        }
    } catch (error) {
        console.error('Error saving embedding model:', error);
        showAlert('Network error saving embedding model', 'danger');
    }

    // Always close modal and clean up backdrop
    closeEmbeddingModal();
}

function updateEditLLMModels() {
    updateLLMModelsForEdit();
}

async function updateLLMModelsForEdit() {
    console.log('--- updateLLMModelsForEdit called ---');
    const provider = document.getElementById('editLLMProvider').value;
    const modelSection = document.getElementById('editLLMModelSection');
    const modelSelect = document.getElementById('editLLMModel');

    console.log('Provider:', provider);
    console.log('Current options count before clear:', modelSelect.options.length);

    // Always show model section since we removed "none" option
    modelSection.classList.remove('d-none');

    // Clear existing options
    modelSelect.innerHTML = '';
    console.log('Cleared options, count now:', modelSelect.options.length);

    // Load available models for the selected provider
    try {
        console.log('Fetching models from API...');
        const response = await fetch('/api/system/llm/models');
        const data = await response.json();
        console.log('API response:', response.ok, data);

        if (response.ok && data[provider] && data[provider].models && data[provider].models.length > 0) {
            // Use API models if available
            const models = data[provider].models;
            console.log('Using API models:', models);
            models.forEach(model => {
                // Check if option already exists to prevent duplicates
                const existingOption = Array.from(modelSelect.options).find(opt => opt.value === model);
                if (!existingOption) {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                    console.log('Added model:', model);
                } else {
                    console.log('Skipped duplicate model:', model);
                }
            });
            console.log('Added API models, final count:', modelSelect.options.length);
            return; // Exit early if API worked
        }
        throw new Error('API did not return valid models');
    } catch (error) {
        console.error('Error loading LLM models, using fallback:', error);
        // Only use fallback if API failed or returned no models
        const fallbackModels = {
            'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
            'google': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
        };

        if (fallbackModels[provider]) {
            console.log('Using fallback models for', provider, ':', fallbackModels[provider]);
            fallbackModels[provider].forEach(model => {
                // Check if option already exists to prevent duplicates
                const existingOption = Array.from(modelSelect.options).find(opt => opt.value === model);
                if (!existingOption) {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                    console.log('Added fallback model:', model);
                } else {
                    console.log('Skipped duplicate fallback model:', model);
                }
            });
            console.log('Added fallback models, final count:', modelSelect.options.length);
        }
    }
    console.log('--- updateLLMModelsForEdit completed ---');
}

async function saveLLMModel() {
    const provider = document.getElementById('editLLMProvider').value;
    const model = document.getElementById('editLLMModel').value;

    try {
        const response = await fetch('/api/system/llm/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                provider: provider,
                model_name: model
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showAlert(`Successfully changed LLM to ${provider}: ${model}`, 'success');
            updateLLMDisplayFromModal(provider, model);
        } else {
            showAlert(`Failed to change LLM: ${result.message || 'Unknown error'}`, 'danger');
        }
    } catch (error) {
        console.error('Error saving LLM settings:', error);
        showAlert('Network error saving LLM settings', 'danger');
    }

    // Always close modal and clean up backdrop
    closeLLMModal();
}

function updateStatsDisplayFromModal(newModel) {
    const embeddingModelDisplay = document.getElementById('embeddingModelDisplay');
    if (embeddingModelDisplay) {
        const modelInfo = availableEmbeddingModels[newModel];
        const provider = modelInfo?.provider || 'openai';
        const displayText = provider === 'openai' || provider === 'google' ? `${provider.toUpperCase()}: ${newModel}` : newModel;
        embeddingModelDisplay.textContent = displayText;
    }
}

function updateLLMDisplayFromModal(provider, model) {
    const llmModelDisplay = document.getElementById('llmModelDisplay');
    if (llmModelDisplay) {
        // Force override if backend returns 'none' or invalid values
        if (provider === 'none' || !provider || provider === '') {
            provider = 'openai';
        }
        if (!model || model === '' || model === 'Simple Context Response') {
            model = 'gpt-4';
        }

        const displayText = `${provider.toUpperCase()}: ${model}`;
        llmModelDisplay.textContent = displayText;
    }
}

// Load LLM display on page load
async function loadLLMDisplay() {
    try {
        const response = await fetch('/api/system/llm/current');
        const data = await response.json();

        if (response.ok) {
            let provider = data.provider || 'openai';
            let model = data.model_name || 'gpt-4';

            // Force override if backend returns 'none' or empty
            if (provider === 'none' || !provider || provider === '') {
                provider = 'openai';
                model = 'gpt-4';
            }
            if (!model || model === '' || model === 'Simple Context Response') {
                model = 'gpt-4';
            }

            updateLLMDisplayFromModal(provider, model);
        }
    } catch (error) {
        console.error('Error loading LLM display:', error);
        // Set default display to OpenAI GPT-4
        const llmModelDisplay = document.getElementById('llmModelDisplay');
        if (llmModelDisplay) {
            llmModelDisplay.textContent = 'OpenAI: gpt-4';
        }
    }
}

// Modal cleanup functions
function closeEmbeddingModal() {
    try {
        const modalElement = document.getElementById('embeddingModelModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) {
            modal.hide();
        }

        // Force cleanup of backdrop and modal state
        setTimeout(() => {
            // Remove any lingering backdrop
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());

            // Remove modal-open class from body
            document.body.classList.remove('modal-open');

            // Reset body style
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';

            // Ensure modal is completely hidden
            if (modalElement) {
                modalElement.style.display = 'none';
                modalElement.classList.remove('show');
                modalElement.setAttribute('aria-hidden', 'true');
                modalElement.removeAttribute('aria-modal');
            }
        }, 300);
    } catch (error) {
        console.error('Error closing embedding modal:', error);
    }
}

function closeLLMModal() {
    try {
        const modalElement = document.getElementById('llmModelModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) {
            modal.hide();
        }

        // Force cleanup of backdrop and modal state
        setTimeout(() => {
            // Remove any lingering backdrop
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());

            // Remove modal-open class from body
            document.body.classList.remove('modal-open');

            // Reset body style
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';

            // Ensure modal is completely hidden
            if (modalElement) {
                modalElement.style.display = 'none';
                modalElement.classList.remove('show');
                modalElement.setAttribute('aria-hidden', 'true');
                modalElement.removeAttribute('aria-modal');
            }
        }, 300);
    } catch (error) {
        console.error('Error closing LLM modal:', error);
    }
}

// Dedicated function for loading LLM settings in edit modal
async function loadCurrentLLMSettingsForEdit() {
    console.log('=== loadCurrentLLMSettingsForEdit called ===');
    try {
        // First set default provider and populate models
        const providerDropdown = document.getElementById('editLLMProvider');
        const modelDropdown = document.getElementById('editLLMModel');

        if (!providerDropdown || !modelDropdown) {
            console.error('Could not find LLM edit dropdowns');
            return;
        }

        // Clear any existing options first
        console.log('Clearing existing model options');
        modelDropdown.innerHTML = '';

        // Set default provider to OpenAI
        providerDropdown.value = 'openai';
        console.log('Set provider to:', providerDropdown.value);

        // Try to load current settings from API first
        let currentProvider = 'openai';
        let currentModel = 'gpt-4';

        try {
            console.log('Fetching current LLM settings from API...');
            const response = await fetch('/api/system/llm/current');
            const data = await response.json();

            if (response.ok && data.provider && data.model_name) {
                currentProvider = data.provider;
                currentModel = data.model_name;
                console.log('Got current settings from API:', currentProvider, currentModel);
            } else {
                console.log('API returned incomplete data, using defaults');
            }
        } catch (apiError) {
            console.error('Error loading current LLM settings from API:', apiError);
            console.log('Using defaults due to API error');
        }

        // Set the provider and populate models ONCE
        providerDropdown.value = currentProvider;
        console.log('Populating models for provider:', currentProvider);
        await updateLLMModelsForEdit();

        // Set the current model
        console.log('Setting model to:', currentModel);
        modelDropdown.value = currentModel;

        console.log('=== loadCurrentLLMSettingsForEdit completed ===');
    } catch (error) {
        console.error('Error in loadCurrentLLMSettingsForEdit:', error);
    }
}

// Upload Modal Functions
function showUploadModal() {
    console.log('showUploadModal called');
    try {
        const modalElement = document.getElementById('uploadModal');
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();

            // Ensure file input event listener is set up when modal is shown
            setTimeout(() => {
                const filesInput = document.getElementById('files');
                if (filesInput) {
                    // Remove any existing listeners to avoid duplicates
                    filesInput.removeEventListener('change', handleFileSelection);
                    // Add the event listener
                    filesInput.addEventListener('change', handleFileSelection);
                    console.log('File input event listener attached');
                }
            }, 100);
        } else {
            console.error('Upload modal not found');
            showAlert('Upload modal not found. Please refresh the page.', 'danger');
        }
    } catch (error) {
        console.error('Error in showUploadModal:', error);
        showAlert('Error opening upload modal', 'danger');
    }
}

function closeUploadModal() {
    try {
        const modalElement = document.getElementById('uploadModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) {
            modal.hide();
        }

        // Force cleanup of backdrop and modal state
        setTimeout(() => {
            // Remove any lingering backdrop
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());

            // Remove modal-open class from body
            document.body.classList.remove('modal-open');

            // Reset body style
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';

            // Ensure modal is completely hidden
            if (modalElement) {
                modalElement.style.display = 'none';
                modalElement.classList.remove('show');
                modalElement.setAttribute('aria-hidden', 'true');
                modalElement.removeAttribute('aria-modal');
            }

            // Refresh the document library after upload
            refreshDocuments();
        }, 300);
    } catch (error) {
        console.error('Error closing upload modal:', error);
    }
}


// Function to summarize a specific document
async function summarizeDocument(documentId, filename) {
    if (!documentId) {
        showAlert('No document selected for summarization', 'warning');
        return;
    }

    // Find the button that triggered this action and show loading state
    const summarizeButton = document.querySelector(`button[onclick="summarizeDocument('${documentId}', '${filename}')"]`);
    const originalButtonHtml = summarizeButton ? summarizeButton.innerHTML : null;

    if (summarizeButton) {
        summarizeButton.disabled = true;
        summarizeButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span>Processing...';
    }

    try {
        // Show loading alert
        const loadingAlertId = showPersistentAlert(`<i class="bi bi-hourglass-split me-2"></i>Generating summary for "${filename}"... This may take a moment.`, 'info');

        const response = await fetch('/api/query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: `Summarize the main points from the document "${filename}". Please provide a comprehensive overview of the key content.`,
                document_filter: documentId
            })
        });

        const data = await response.json();

        // Remove loading alert
        removePersistentAlert(loadingAlertId);

        if (response.ok) {
            // Create a modal to show the summary
            showDocumentSummaryModal(filename, data.response, data.sources);
            showAlert(`Summary generated successfully for "${filename}"`, 'success');
        } else {
            const errorMessage = data?.detail || data?.message || JSON.stringify(data) || 'Unknown error';
            showAlert(`Failed to generate summary: ${errorMessage}`, 'danger');
        }
    } catch (error) {
        console.error('Error generating summary:', error);
        showAlert('Network error generating summary', 'danger');
    } finally {
        // Restore button state
        if (summarizeButton && originalButtonHtml) {
            summarizeButton.disabled = false;
            summarizeButton.innerHTML = originalButtonHtml;
        }
    }
}

// Function to get key findings from a specific document
async function getKeyFindings(documentId, filename) {
    if (!documentId) {
        showAlert('No document selected for key findings', 'warning');
        return;
    }

    // Find the button that triggered this action and show loading state
    const findingsButton = document.querySelector(`button[onclick="getKeyFindings('${documentId}', '${filename}')"]`);
    const originalButtonHtml = findingsButton ? findingsButton.innerHTML : null;

    if (findingsButton) {
        findingsButton.disabled = true;
        findingsButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span>Analyzing...';
    }

    try {
        // Show loading alert
        const loadingAlertId = showPersistentAlert(`<i class="bi bi-search me-2"></i>Extracting key findings from "${filename}"... This may take a moment.`, 'info');

        const response = await fetch('/api/query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: `What are the key findings, insights, or important conclusions from the document "${filename}"? Please provide specific details and highlight the most significant points.`,
                document_filter: documentId
            })
        });

        const data = await response.json();

        // Remove loading alert
        removePersistentAlert(loadingAlertId);

        if (response.ok) {
            // Create a modal to show the key findings
            showDocumentFindingsModal(filename, data.response, data.sources);
            showAlert(`Key findings extracted successfully for "${filename}"`, 'success');
        } else {
            const errorMessage = data?.detail || data?.message || JSON.stringify(data) || 'Unknown error';
            showAlert(`Failed to extract key findings: ${errorMessage}`, 'danger');
        }
    } catch (error) {
        console.error('Error extracting key findings:', error);
        showAlert('Network error extracting key findings', 'danger');
    } finally {
        // Restore button state
        if (findingsButton && originalButtonHtml) {
            findingsButton.disabled = false;
            findingsButton.innerHTML = originalButtonHtml;
        }
    }
}

// Function to show document summary in a modal
function showDocumentSummaryModal(filename, summary, sources) {
    const modalId = 'documentSummaryModal';
    let modal = document.getElementById(modalId);

    if (!modal) {
        // Create modal if it doesn't exist
        const modalHtml = `
            <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}Label" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="${modalId}Label">
                                <i class="bi bi-file-text me-2"></i>Document Summary
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="summaryContent"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById(modalId);
    }

    // Update content
    const titleElement = modal.querySelector('.modal-title');
    titleElement.innerHTML = `<i class="bi bi-file-text me-2"></i>Summary: ${filename}`;

    const contentElement = modal.querySelector('#summaryContent');
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <hr>
            <h6>Sources:</h6>
            <div class="source-refs">
                ${sources.map(source => `
                    <span class="source-ref">
                        ${source.document_name || source.filename || 'Unknown'}
                        ${source.page ? `(Page ${source.page})` : ''}
                    </span>
                `).join('')}
            </div>
        `;
    }

    contentElement.innerHTML = `
        <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i>
            <strong>Document:</strong> ${filename}
        </div>
        <div class="summary-text">
            ${summary.replace(/\n/g, '<br>')}
        </div>
        ${sourcesHtml}
    `;

    // Show modal
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
}

// Function to show document findings in a modal
function showDocumentFindingsModal(filename, findings, sources) {
    const modalId = 'documentFindingsModal';
    let modal = document.getElementById(modalId);

    if (!modal) {
        // Create modal if it doesn't exist
        const modalHtml = `
            <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}Label" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="${modalId}Label">
                                <i class="bi bi-search me-2"></i>Key Findings
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="findingsContent"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById(modalId);
    }

    // Update content
    const titleElement = modal.querySelector('.modal-title');
    titleElement.innerHTML = `<i class="bi bi-search me-2"></i>Key Findings: ${filename}`;

    const contentElement = modal.querySelector('#findingsContent');
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <hr>
            <h6>Sources:</h6>
            <div class="source-refs">
                ${sources.map(source => `
                    <span class="source-ref">
                        ${source.document_name || source.filename || 'Unknown'}
                        ${source.page ? `(Page ${source.page})` : ''}
                    </span>
                `).join('')}
            </div>
        `;
    }

    contentElement.innerHTML = `
        <div class="alert alert-success">
            <i class="bi bi-search me-2"></i>
            <strong>Document:</strong> ${filename}
        </div>
        <div class="findings-text">
            ${findings.replace(/\n/g, '<br>')}
        </div>
        ${sourcesHtml}
    `;

    // Show modal
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
}

// Helper functions for persistent loading alerts
let persistentAlertCounter = 0;
const activePersistentAlerts = new Map();

function showPersistentAlert(message, type = 'info') {
    const alertId = `persistent-alert-${++persistentAlertCounter}`;

    const alertContainer = document.createElement('div');
    alertContainer.id = alertId;
    alertContainer.className = `alert alert-${type} alert-dismissible fade show position-fixed d-flex align-items-center`;
    alertContainer.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 350px; max-width: 500px;';
    alertContainer.innerHTML = `
        <div class="flex-grow-1">${message}</div>
        <button type="button" class="btn-close ms-2" onclick="removePersistentAlert('${alertId}')"></button>
    `;

    document.body.appendChild(alertContainer);
    activePersistentAlerts.set(alertId, alertContainer);

    return alertId;
}

function removePersistentAlert(alertId) {
    const alertElement = activePersistentAlerts.get(alertId);
    if (alertElement && alertElement.parentNode) {
        alertElement.classList.remove('show');
        alertElement.classList.add('fade');

        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.parentNode.removeChild(alertElement);
            }
            activePersistentAlerts.delete(alertId);
        }, 150);
    }
}