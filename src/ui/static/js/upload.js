// File upload functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const uploadResults = document.getElementById('uploadResults');
    const filesInput = document.getElementById('files');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const files = filesInput.files;
        if (files.length === 0) {
            showAlert('Please select at least one file to upload.', 'warning');
            return;
        }

        // Show loading state
        uploadBtn.disabled = true;
        uploadSpinner.classList.remove('d-none');
        uploadResults.innerHTML = '';

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await fetch('/api/upload-documents/', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayUploadResults(result);
                filesInput.value = ''; // Clear file input
            } else {
                showAlert(`Upload failed: ${result.detail || 'Unknown error'}`, 'danger');
            }

        } catch (error) {
            console.error('Upload error:', error);
            showAlert('Upload failed due to network error. Please try again.', 'danger');
        } finally {
            // Hide loading state
            uploadBtn.disabled = false;
            uploadSpinner.classList.add('d-none');
        }
    });

    // File input change handler for validation
    filesInput.addEventListener('change', function() {
        const files = this.files;
        const allowedTypes = ['.pdf', '.docx', '.xlsx', '.xls', '.txt'];
        let invalidFiles = [];

        for (let file of files) {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedTypes.includes(extension)) {
                invalidFiles.push(file.name);
            }
        }

        if (invalidFiles.length > 0) {
            showAlert(`Invalid file types: ${invalidFiles.join(', ')}. Please select only PDF, DOCX, XLSX, XLS, or TXT files.`, 'warning');
            this.value = '';
        }
    });
});

function displayUploadResults(result) {
    const uploadResults = document.getElementById('uploadResults');

    let html = `
        <div class="alert alert-info">
            <h6>Upload Results</h6>
            <p>${result.message}</p>
        </div>
    `;

    if (result.files && result.files.length > 0) {
        html += '<div class="list-group">';

        result.files.forEach(file => {
            const statusClass = file.status === 'processed' ? 'success' : 'danger';
            const statusIcon = file.status === 'processed' ? 'check-circle' : 'x-circle';

            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">
                                <i class="bi bi-${statusIcon} text-${statusClass}"></i>
                                ${file.filename}
                            </h6>
                            ${file.status === 'processed' ?
                                `<p class="mb-1 text-success">Successfully processed</p>` :
                                `<p class="mb-1 text-danger">Error: ${file.error}</p>`
                            }
                        </div>
                        <span class="badge bg-${statusClass}">${file.status}</span>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    }

    uploadResults.innerHTML = html;
}

function showAlert(message, type = 'info') {
    const uploadResults = document.getElementById('uploadResults');
    uploadResults.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
}

// Drag and drop functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const filesInput = document.getElementById('files');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadForm.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadForm.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        uploadForm.classList.add('border-primary');
    }

    function unhighlight(e) {
        uploadForm.classList.remove('border-primary');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        filesInput.files = files;

        // Trigger change event for validation
        const event = new Event('change', { bubbles: true });
        filesInput.dispatchEvent(event);
    }
});