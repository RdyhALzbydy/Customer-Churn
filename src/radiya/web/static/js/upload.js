class FileUploader {
    constructor() {
        this.currentJobId = null;
        this.polling = false;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const uploadForm = document.getElementById('upload-form');

        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
        });

        dropZone.addEventListener('drop', this.handleDrop.bind(this), false);
        fileInput.addEventListener('change', this.handleFileSelect.bind(this), false);
        uploadForm.addEventListener('submit', this.handleUpload.bind(this), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
                    if (files.length > 0) {
                this.selectFile(files[0]);
            }
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.selectFile(e.target.files[0]);
        }
    }

    selectFile(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        // Show selected file info
        const selectedFileDiv = document.getElementById('selected-file');
        const fileNameSpan = document.getElementById('file-name');
        const fileSizeSpan = document.getElementById('file-size');
        const uploadBtn = document.getElementById('upload-btn');
        const dropZone = document.getElementById('drop-zone');

        fileNameSpan.textContent = file.name;
        fileSizeSpan.textContent = this.formatFileSize(file.size);
        
        selectedFileDiv.style.display = 'block';
        dropZone.style.display = 'none';
        uploadBtn.disabled = false;

        this.selectedFile = file;
    }

    validateFile(file) {
        const maxSize = 129 * 1024 * 1024; // 129 MB
        const allowedTypes = ['.json', '.csv'];
        
        // Check file type
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(extension)) {
            showToast('نوع الملف غير مدعوم. يرجى رفع ملف JSON أو CSV', 'error');
            return false;
        }

        // Check file size
        if (file.size > maxSize) {
            showToast('حجم الملف كبير جداً. الحد الأقصى 129 MB', 'error');
            return false;
        }

        return true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async handleUpload(e) {
        e.preventDefault();
        
        if (!this.selectedFile) {
            showToast('يرجى اختيار ملف أولاً', 'error');
            return;
        }

        this.showUploadProgress();
        
        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            const response = await fetch('/api/v1/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'خطأ في رفع الملف');
            }

            this.currentJobId = result.job_id;
            this.startPolling();
            showToast('تم رفع الملف بنجاح! جاري المعالجة...', 'success');

        } catch (error) {
            this.hideUploadProgress();
            showToast(error.message || 'حدث خطأ في رفع الملف', 'error');
        }
    }

    showUploadProgress() {
        const uploadForm = document.getElementById('upload-form');
        const progressDiv = document.getElementById('upload-progress');
        
        uploadForm.style.display = 'none';
        progressDiv.style.display = 'block';
        
        this.updateProgress(0, 'بدء رفع الملف...');
    }

    hideUploadProgress() {
        const uploadForm = document.getElementById('upload-form');
        const progressDiv = document.getElementById('upload-progress');
        
        uploadForm.style.display = 'block';
        progressDiv.style.display = 'none';
    }

    updateProgress(percentage, status) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const progressStatus = document.getElementById('progress-status');

        progressFill.style.width = percentage + '%';
        progressText.textContent = Math.round(percentage) + '%';
        progressStatus.textContent = status;
    }

    async startPolling() {
        if (this.polling) return;
        
        this.polling = true;
        
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/v1/status/${this.currentJobId}`);
                const status = await response.json();

                this.updateProgress(status.progress, this.getStatusText(status.progress));

                if (status.status === 'completed') {
                    clearInterval(pollInterval);
                    this.polling = false;
                    this.showResults(status.results);
                } else if (status.status === 'error') {
                    clearInterval(pollInterval);
                    this.polling = false;
                    this.hideUploadProgress();
                    showToast(`خطأ في المعالجة: ${status.error}`, 'error');
                }

            } catch (error) {
                clearInterval(pollInterval);
                this.polling = false;
                this.hideUploadProgress();
                showToast('حدث خطأ في متابعة حالة المعالجة', 'error');
            }
        }, 2000);
    }

    getStatusText(progress) {
        if (progress < 30) return 'تحميل وتحليل البيانات...';
        if (progress < 50) return 'التحقق من صحة البيانات...';
        if (progress < 70) return 'هندسة الميزات...';
        if (progress < 90) return 'تدريب النماذج...';
        return 'تجهيز النتائج...';
    }

    showResults(results) {
        const progressDiv = document.getElementById('upload-progress');
        const resultsDiv = document.getElementById('results-preview');
        const resultsGrid = document.getElementById('results-grid');
        const viewFullBtn = document.getElementById('view-full-results');

        progressDiv.style.display = 'none';
        resultsDiv.style.display = 'block';

        // Create results summary
        resultsGrid.innerHTML = `
            <div class="result-card">
                <div class="result-icon"><i class="fas fa-database"></i></div>
                <div class="result-info">
                    <div class="result-value">${results.data_summary.total_records.toLocaleString()}</div>
                    <div class="result-label">إجمالي السجلات</div>
                </div>
            </div>
            <div class="result-card">
                <div class="result-icon"><i class="fas fa-users"></i></div>
                <div class="result-info">
                    <div class="result-value">${results.data_summary.unique_users.toLocaleString()}</div>
                    <div class="result-label">مستخدم فريد</div>
                </div>
            </div>
            <div class="result-card">
                <div class="result-icon"><i class="fas fa-chart-line"></i></div>
                <div class="result-info">
                    <div class="result-value">${results.data_summary.churn_rates.combined}%</div>
                    <div class="result-label">معدل الانسحاب</div>
                </div>
            </div>
            <div class="result-card">
                <div class="result-icon"><i class="fas fa-trophy"></i></div>
                <div class="result-info">
                    <div class="result-value">${Math.max(...Object.values(results.best_models).map(m => m.auc_score * 100)).toFixed(1)}%</div>
                    <div class="result-label">أفضل دقة</div>
                </div>
            </div>
        `;

        viewFullBtn.href = `/results/${this.currentJobId}`;
        showToast('تم تحليل البيانات بنجاح!', 'success');
    }
}

function removeFile() {
    const selectedFileDiv = document.getElementById('selected-file');
    const dropZone = document.getElementById('drop-zone');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');

    selectedFileDiv.style.display = 'none';
    dropZone.style.display = 'block';
    uploadBtn.disabled = true;
    fileInput.value = '';
    
    if (window.uploader) {
        window.uploader.selectedFile = null;
    }
}

function resetUpload() {
    removeFile();
    const resultsDiv = document.getElementById('results-preview');
    const uploadForm = document.getElementById('upload-form');
    
    resultsDiv.style.display = 'none';
    uploadForm.style.display = 'block';
    
    if (window.uploader) {
        window.uploader.currentJobId = null;
        window.uploader.polling = false;
    }
}

// Initialize uploader when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.uploader = new FileUploader();
});