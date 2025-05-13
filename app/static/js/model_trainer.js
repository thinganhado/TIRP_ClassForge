// model_trainer.js - Handles HuggingFace model training and status

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const trainModelBtn = document.getElementById('train-model');
    const modelStatusContainer = document.getElementById('model-status');
    const trainingProgressContainer = document.getElementById('training-progress');
    
    // Initialize
    checkModelStatus();
    
    // Event Listeners
    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', function() {
            trainModel();
        });
    }
    
    // Functions
    function checkModelStatus() {
        if (!modelStatusContainer) return;
        
        fetch('/api/assistant/huggingface_status')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const statusHtml = `
                        <div class="status-item ${data.huggingface_available ? 'status-success' : 'status-error'}">
                            <span class="status-label">HuggingFace Available:</span>
                            <span class="status-value">${data.huggingface_available ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="status-item ${data.models_status.nlp_model ? 'status-success' : 'status-warning'}">
                            <span class="status-label">NLP Model:</span>
                            <span class="status-value">${data.models_status.nlp_model ? 'Loaded' : 'Not Loaded'}</span>
                        </div>
                        <div class="status-item ${data.models_status.embedder ? 'status-success' : 'status-warning'}">
                            <span class="status-label">Sentence Embedder:</span>
                            <span class="status-value">${data.models_status.embedder ? 'Loaded' : 'Not Loaded'}</span>
                        </div>
                        <div class="status-item ${data.models_status.fine_tuned_model ? 'status-success' : 'status-warning'}">
                            <span class="status-label">Fine-tuned Model:</span>
                            <span class="status-value">${data.models_status.fine_tuned_model ? 'Available' : 'Not Available'}</span>
                        </div>
                        <div class="status-item ${data.api_configured ? 'status-success' : 'status-warning'}">
                            <span class="status-label">API Token:</span>
                            <span class="status-value">${data.api_configured ? 'Configured' : 'Not Configured'}</span>
                        </div>
                    `;
                    
                    modelStatusContainer.innerHTML = statusHtml;
                    
                    // Update train button state
                    if (trainModelBtn) {
                        if (!data.huggingface_available) {
                            trainModelBtn.disabled = true;
                            trainModelBtn.title = "HuggingFace not available";
                        } else {
                            trainModelBtn.disabled = false;
                            trainModelBtn.title = "";
                        }
                    }
                } else {
                    modelStatusContainer.innerHTML = `<div class="status-error">Error checking model status: ${data.message}</div>`;
                }
            })
            .catch(error => {
                console.error('Error checking model status:', error);
                modelStatusContainer.innerHTML = `<div class="status-error">Error connecting to server</div>`;
            });
    }
    
    function trainModel() {
        if (!trainingProgressContainer) return;
        
        // Show progress
        trainingProgressContainer.innerHTML = `
            <div class="progress-message">
                <div class="spinner"></div>
                <p>Training model on teacher comments data... This may take several minutes.</p>
            </div>
        `;
        
        // Disable button during training
        if (trainModelBtn) {
            trainModelBtn.disabled = true;
        }
        
        // Call API to train model
        fetch('/api/assistant/fine_tune', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                trainingProgressContainer.innerHTML = `
                    <div class="success-message">
                        <i class="fas fa-check-circle"></i>
                        <p>${data.message}</p>
                    </div>
                `;
                
                // Update status after successful training
                setTimeout(checkModelStatus, 1000);
            } else {
                trainingProgressContainer.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>${data.message}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error training model:', error);
            trainingProgressContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Error connecting to server</p>
                </div>
            `;
        })
        .finally(() => {
            // Re-enable button
            if (trainModelBtn) {
                trainModelBtn.disabled = false;
            }
        });
    }
}); 