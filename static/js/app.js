// PDF Content Moderator - Frontend JavaScript
class ContentModerator {
    constructor() {
        this.apiEndpoint = '/api';
        this.currentUpload = null;
        this.pollInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupProgressTracking();
        this.checkApiStatus();
    }

    // Event Listeners
    setupEventListeners() {
        // File input change
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        }

        // Form submission
        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => this.handleFormSubmit(e));
        }

        // Confidence threshold slider
        const thresholdSlider = document.getElementById('confidenceThreshold');
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', (e) => this.updateThresholdDisplay(e.target.value));
        }

        // Collapse toggles
        document.querySelectorAll('.collapse-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => this.handleCollapseToggle(e));
        });

        // Image modals
        document.querySelectorAll('.image-thumbnail').forEach(img => {
            img.addEventListener('click', (e) => this.showImageModal(e.target));
        });

        // Search functionality
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.filterViolations(e.target.value));
        }

        // Export buttons
        document.querySelectorAll('[data-export]').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleExport(e.target.dataset.export));
        });
    }

    // Drag and Drop Setup
    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        if (!uploadZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => this.highlight(uploadZone), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => this.unhighlight(uploadZone), false);
        });

        uploadZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(element) {
        element.classList.add('dragover');
    }

    unhighlight(element) {
        element.classList.remove('dragover');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }

    // File Handling
    handleFileSelect(file) {
        if (!file) return;

        // Validate file
        const validation = this.validateFile(file);
        if (!validation.valid) {
            this.showAlert('error', validation.message);
            return;
        }

        // Update UI
        this.displayFileInfo(file);
        this.enableSubmitButton();
        this.updateUploadZone(file);

        // Store file reference
        this.currentUpload = file;
    }

    validateFile(file) {
        // Check file type
        if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
            return {
                valid: false,
                message: 'Please select a PDF file only.'
            };
        }

        // Check file size (50MB limit)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            return {
                valid: false,
                message: 'File size exceeds 50MB limit. Please select a smaller file.'
            };
        }

        // Check if file is not empty
        if (file.size === 0) {
            return {
                valid: false,
                message: 'The selected file appears to be empty.'
            };
        }

        return { valid: true };
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');

        if (fileInfo && fileName && fileSize) {
            fileName.textContent = file.name;
            fileSize.textContent = this.formatFileSize(file.size);
            fileInfo.style.display = 'block';
        }
    }

    updateUploadZone(file) {
        const uploadZone = document.getElementById('uploadZone');
        if (uploadZone) {
            uploadZone.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-check-circle text-success" style="font-size: 3rem;"></i>
                    <h5 class="mt-3 text-success">File Selected</h5>
                    <p class="text-muted">${file.name}</p>
                    <p class="text-muted small">Click to select a different file</p>
                </div>
            `;
        }
    }

    enableSubmitButton() {
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.classList.remove('btn-secondary');
            submitBtn.classList.add('btn-success');
        }
    }

    // Form Submission
    async handleFormSubmit(e) {
        e.preventDefault();

        if (!this.currentUpload) {
            this.showAlert('error', 'Please select a file first.');
            return;
        }

        try {
            // Show processing indicator
            this.showProcessingIndicator();

            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.currentUpload);
            formData.append('include_images', this.getOptionValue('analyzeImages'));
            formData.append('include_text', this.getOptionValue('analyzeText'));
            formData.append('confidence_threshold', this.getOptionValue('confidenceThreshold'));

            // Submit via API
            const response = await this.submitToAPI(formData);

            if (response.success) {
                // Redirect to results
                window.location.href = `/results/${response.result_id}`;
            } else {
                throw new Error(response.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('error', `Upload failed: ${error.message}`);
            this.hideProcessingIndicator();
        }
    }

    async submitToAPI(formData) {
        const response = await fetch(`${this.apiEndpoint}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    getOptionValue(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return null;

        if (element.type === 'checkbox') {
            return element.checked;
        } else if (element.type === 'range') {
            return parseFloat(element.value);
        } else {
            return element.value;
        }
    }

    // Progress Tracking
    setupProgressTracking() {
        // Initialize progress bars with animation
        this.animateProgressBars();
        
        // Setup auto-refresh for processing status
        this.setupStatusPolling();
    }

    animateProgressBars() {
        const progressBars = document.querySelectorAll('.confidence-fill, .progress-bar');
        
        progressBars.forEach(bar => {
            const targetWidth = bar.style.width || bar.dataset.width;
            if (targetWidth) {
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 500);
            }
        });
    }

    setupStatusPolling() {
        // Check if we're on a results page that might need updates
        const resultId = this.getResultIdFromURL();
        if (resultId) {
            this.pollForUpdates(resultId);
        }
    }

    getResultIdFromURL() {
        const match = window.location.pathname.match(/\/results\/([^\/]+)/);
        return match ? match[1] : null;
    }

    async pollForUpdates(resultId) {
        try {
            const response = await fetch(`${this.apiEndpoint}/results/${resultId}`);
            const data = await response.json();

            if (data.success && data.summary.overall_risk_level === 'processing') {
                // Still processing, poll again
                setTimeout(() => this.pollForUpdates(resultId), 5000);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }

    showProcessingIndicator() {
        const cardBody = document.querySelector('.card-body');
        const processingIndicator = document.getElementById('processingIndicator');

        if (cardBody && processingIndicator) {
            cardBody.style.display = 'none';
            processingIndicator.style.display = 'block';

            // Animate progress bar
            this.animateProcessingProgress();
        }
    }

    hideProcessingIndicator() {
        const cardBody = document.querySelector('.card-body');
        const processingIndicator = document.getElementById('processingIndicator');

        if (cardBody && processingIndicator) {
            cardBody.style.display = 'block';
            processingIndicator.style.display = 'none';
        }
    }

    animateProcessingProgress() {
        const progressBar = document.querySelector('#processingIndicator .progress-bar');
        if (!progressBar) return;

        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) {
                progress = 95;
                clearInterval(interval);
            }
            progressBar.style.width = progress + '%';
        }, 1000);

        // Store interval for cleanup
        this.progressInterval = interval;
    }

    // UI Interactions
    updateThresholdDisplay(value) {
        const display = document.getElementById('thresholdValue');
        if (display) {
            display.textContent = value;
        }
    }

    handleCollapseToggle(e) {
        const icon = e.target.querySelector('i') || e.target;
        
        setTimeout(() => {
            if (icon.classList.contains('fa-chevron-down')) {
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            } else {
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            }
        }, 200);
    }

    showImageModal(img) {
        // Create or update modal
        let modal = document.getElementById('imageViewerModal');
        
        if (!modal) {
            modal = this.createImageModal();
        }

        // Update modal content
        const modalImg = modal.querySelector('.modal-body img');
        const modalTitle = modal.querySelector('.modal-title');

        if (modalImg && modalTitle) {
            modalImg.src = img.src;
            modalTitle.textContent = img.alt || 'Image Viewer';
        }

        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    createImageModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'imageViewerModal';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Image Viewer</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img class="img-fluid" alt="Full size image">
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        return modal;
    }

    // Search and Filter
    filterViolations(searchTerm) {
        const violationCards = document.querySelectorAll('.violation-card');
        const term = searchTerm.toLowerCase();

        violationCards.forEach(card => {
            const text = card.textContent.toLowerCase();
            const isVisible = text.includes(term);
            
            card.style.display = isVisible ? 'block' : 'none';
            
            // Add highlight effect
            if (isVisible && term) {
                this.highlightSearchTerm(card, term);
            } else {
                this.removeHighlight(card);
            }
        });

        // Update results count
        this.updateSearchResults(violationCards, searchTerm);
    }

    highlightSearchTerm(element, term) {
        // Simple highlighting - can be enhanced
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const textNodes = [];
        let node;

        while (node = walker.nextNode()) {
            textNodes.push(node);
        }

        textNodes.forEach(textNode => {
            const text = textNode.textContent;
            const regex = new RegExp(`(${term})`, 'gi');
            
            if (regex.test(text)) {
                const highlightedText = text.replace(regex, '<mark>$1</mark>');
                const span = document.createElement('span');
                span.innerHTML = highlightedText;
                textNode.parentNode.replaceChild(span, textNode);
            }
        });
    }

    removeHighlight(element) {
        const marks = element.querySelectorAll('mark');
        marks.forEach(mark => {
            mark.outerHTML = mark.innerHTML;
        });
    }

    updateSearchResults(cards, searchTerm) {
        const visibleCount = Array.from(cards).filter(card => 
            card.style.display !== 'none'
        ).length;

        const resultsInfo = document.getElementById('searchResults');
        if (resultsInfo) {
            if (searchTerm) {
                resultsInfo.textContent = `${visibleCount} violations found for "${searchTerm}"`;
                resultsInfo.style.display = 'block';
            } else {
                resultsInfo.style.display = 'none';
            }
        }
    }

    // Export Functionality
    async handleExport(format) {
        const resultId = this.getResultIdFromURL();
        if (!resultId) {
            this.showAlert('error', 'No results to export');
            return;
        }

        try {
            let url;
            let filename;

            switch (format) {
                case 'pdf':
                    url = `/report/${resultId}`;
                    filename = `moderation_report_${resultId}.pdf`;
                    break;
                case 'json':
                    url = `${this.apiEndpoint}/results/${resultId}?include_details=true`;
                    filename = `results_${resultId}.json`;
                    break;
                case 'csv':
                    // Convert violations to CSV
                    await this.exportToCSV(resultId);
                    return;
                default:
                    throw new Error('Unknown export format');
            }

            // Download file
            this.downloadFile(url, filename);

        } catch (error) {
            console.error('Export error:', error);
            this.showAlert('error', `Export failed: ${error.message}`);
        }
    }

    async exportToCSV(resultId) {
        const response = await fetch(`${this.apiEndpoint}/results/${resultId}?include_details=true`);
        const data = await response.json();

        if (!data.success || !data.violations) {
            throw new Error('No violation data available');
        }

        // Convert violations to CSV
        const csvContent = this.convertViolationsToCSV(data.violations);
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        this.downloadFile(url, `violations_${resultId}.csv`);
        URL.revokeObjectURL(url);
    }

    convertViolationsToCSV(violations) {
        const headers = [
            'Page Number',
            'Violation Type',
            'Category', 
            'Severity',
            'Description',
            'Confidence',
            'Timestamp'
        ];

        const rows = violations.map(v => [
            v.page_number,
            v.violation_type,
            v.category,
            v.severity,
            `"${v.description.replace(/"/g, '""')}"`,
            v.confidence,
            v.timestamp
        ]);

        return [headers, ...rows]
            .map(row => row.join(','))
            .join('\n');
    }

    downloadFile(url, filename) {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // API Status Check
    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiEndpoint}/health`);
            const status = await response.json();

            this.updateStatusIndicator(status);

        } catch (error) {
            console.warn('API status check failed:', error);
            this.updateStatusIndicator({ status: 'offline' });
        }
    }

    updateStatusIndicator(status) {
        const indicator = document.getElementById('apiStatus');
        if (!indicator) return;

        const isHealthy = status.status === 'healthy';
        
        indicator.className = `badge ${isHealthy ? 'bg-success' : 'bg-danger'}`;
        indicator.textContent = isHealthy ? 'Online' : 'Offline';
        indicator.title = isHealthy ? 'API is healthy' : 'API is unavailable';
    }

    // Utility Functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showAlert(type, message) {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        alert.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert at top of container
        const container = document.querySelector('.container, .container-fluid');
        if (container) {
            container.insertBefore(alert, container.firstChild);
        }

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    // Cleanup
    cleanup() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
    }
}

// Analytics and Performance Tracking
class AnalyticsTracker {
    constructor() {
        this.startTime = Date.now();
        this.events = [];
    }

    trackEvent(category, action, label, value) {
        const event = {
            category,
            action,
            label,
            value,
            timestamp: Date.now(),
            url: window.location.pathname
        };

        this.events.push(event);
        
        // Send to analytics service (if configured)
        this.sendToAnalytics(event);
    }

    trackPageLoad() {
        const loadTime = Date.now() - this.startTime;
        this.trackEvent('Performance', 'Page Load', window.location.pathname, loadTime);
    }

    trackFileUpload(fileSize, fileType) {
        this.trackEvent('Upload', 'File Selected', fileType, fileSize);
    }

    trackAnalysisComplete(processingTime, violationCount) {
        this.trackEvent('Analysis', 'Complete', 'Processing Time', processingTime);
        this.trackEvent('Analysis', 'Complete', 'Violations Found', violationCount);
    }

    sendToAnalytics(event) {
        // Implement analytics service integration here
        // e.g., Google Analytics, custom analytics endpoint
        console.log('Analytics Event:', event);
    }
}

// Accessibility Enhancements
class AccessibilityManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupKeyboardNavigation();
        this.setupScreenReaderSupport();
        this.setupFocusManagement();
    }

    setupKeyboardNavigation() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // ESC to close modals
            if (e.key === 'Escape') {
                const openModal = document.querySelector('.modal.show');
                if (openModal) {
                    const modal = bootstrap.Modal.getInstance(openModal);
                    modal?.hide();
                }
            }

            // Ctrl+U for upload
            if (e.ctrlKey && e.key === 'u') {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                fileInput?.click();
            }
        });
    }

    setupScreenReaderSupport() {
        // Add aria-live regions for dynamic updates
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.id = 'liveRegion';
        document.body.appendChild(liveRegion);
    }

    setupFocusManagement() {
        // Trap focus in modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                const modal = document.querySelector('.modal.show');
                if (modal) {
                    this.trapFocus(modal, e);
                }
            }
        });
    }

    trapFocus(modal, e) {
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    }

    announceToScreenReader(message) {
        const liveRegion = document.getElementById('liveRegion');
        if (liveRegion) {
            liveRegion.textContent = message;
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize main application
    window.contentModerator = new ContentModerator();
    
    // Initialize analytics
    window.analytics = new AnalyticsTracker();
    window.analytics.trackPageLoad();
    
    // Initialize accessibility
    window.accessibility = new AccessibilityManager();
    
    // Setup error handling
    window.addEventListener('error', function(e) {
        console.error('JavaScript Error:', e.error);
        window.analytics?.trackEvent('Error', 'JavaScript', e.error.message, 1);
    });

    // Setup unload cleanup
    window.addEventListener('beforeunload', function() {
        window.contentModerator?.cleanup();
    });
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ContentModerator, AnalyticsTracker, AccessibilityManager };
}