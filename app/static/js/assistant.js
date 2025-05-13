// assistant.js - Handles client-side functionality for the AI assistant

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const promptInput = document.getElementById('ai-prompt');
    const editButton = document.querySelector('.edit-button');
    const chatContainer = document.querySelector('.chat-container');
    const confirmButton = document.querySelector('.confirm-button');
    const messageInput = document.getElementById('chat-message');
    const sendButton = document.querySelector('.send-button');
    
    // Generate a unique session ID if not already set
    let sessionId = localStorage.getItem('chatSessionId');
    if (!sessionId) {
        sessionId = generateSessionId();
        localStorage.setItem('chatSessionId', sessionId);
    }
    
    // Store the current configuration recommendations
    let currentRecommendedConfig = null;
    
    // Event listeners
    if (editButton) {
        editButton.addEventListener('click', function() {
            promptInput.focus();
            promptInput.select();
        });
    }
    
    if (promptInput) {
        promptInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processPrompt(promptInput.value);
            }
        });
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', function() {
            processUserMessage(messageInput.value);
            messageInput.value = '';
        });
    }
    
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processUserMessage(messageInput.value);
                messageInput.value = '';
            }
        });
    }
    
    if (confirmButton) {
        confirmButton.addEventListener('click', function() {
            if (currentRecommendedConfig) {
                confirmChanges(currentRecommendedConfig);
            } else {
                addAssistantMessage("There are no pending changes to confirm.");
            }
        });
    }
    
    // Initialize
    function initialize() {
        // Get recommendations on page load
        fetch('/api/assistant/recommendations')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.recommendations.length > 0) {
                    const recommendations = data.recommendations.map(rec => `<li>${rec}</li>`).join('');
                    const recommendationHtml = `
                        <div class="assistant-message">
                            <div class="assistant-avatar"><img src="/static/images/chatbot_avatar.png" alt="Customisation Agent"></div>
                            <div class="message-content">
                                <p>Welcome! Here are some recommendations based on current data:</p>
                                <ul>${recommendations}</ul>
                                <p>How would you like to customize your class allocation?</p>
                            </div>
                        </div>
                    `;
                    chatContainer.innerHTML = recommendationHtml;
                }
            })
            .catch(error => {
                console.error('Error fetching recommendations:', error);
            });
    }
    
    // Process the main prompt
    function processPrompt(prompt) {
        if (!prompt.trim()) return;
        
        // Add user message to chat
        addUserMessage(prompt);
        
        // Disable input while processing
        if (promptInput) promptInput.disabled = true;
        if (sendButton) sendButton.disabled = true;
        
        // Send to server for analysis
        fetch('/api/assistant/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                input: prompt,
                session_id: sessionId
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add assistant response
                addAssistantMessage(data.message);
                
                // Store recommended config if available
                if (data.is_modified) {
                    currentRecommendedConfig = data.modified_config;
                    confirmButton.style.display = 'block';
                } else {
                    currentRecommendedConfig = null;
                    confirmButton.style.display = 'none';
                }
            } else {
                // Show error message
                addAssistantMessage(`Sorry, I couldn't process that request: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Error processing prompt:', error);
            addAssistantMessage("Sorry, there was an error processing your request. Please try again.");
        })
        .finally(() => {
            // Re-enable input
            if (promptInput) promptInput.disabled = false;
            if (sendButton) sendButton.disabled = false;
        });
    }
    
    // Process follow-up messages
    function processUserMessage(message) {
        if (!message.trim()) return;
        
        // Add user message to chat
        addUserMessage(message);
        
        // Disable input while processing
        messageInput.disabled = true;
        sendButton.disabled = true;
        
        // Process message based on context
        if (currentRecommendedConfig) {
            // User is responding to recommendation
            if (messageHasConfirmation(message)) {
                confirmChanges(currentRecommendedConfig);
                // Re-enable input after handling confirmation
                messageInput.disabled = false;
                sendButton.disabled = false;
                return; // Prevent further processing
            } else if (messageHasRejection(message)) {
                addAssistantMessage("I've discarded the proposed changes. How else can I help you?");
                currentRecommendedConfig = null;
                confirmButton.style.display = 'none';
                // Re-enable input after handling rejection
                messageInput.disabled = false;
                sendButton.disabled = false;
                return; // Prevent further processing
            }
        }
        
        // For all other messages, send directly to API
        fetch('/api/assistant/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                input: message,
                session_id: sessionId 
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add assistant response
                addAssistantMessage(data.message);
                
                // Store recommended config if available
                if (data.is_modified) {
                    currentRecommendedConfig = data.modified_config;
                    confirmButton.style.display = 'block';
                } else {
                    currentRecommendedConfig = null;
                    confirmButton.style.display = 'none';
                }
            } else {
                // Show error message
                addAssistantMessage(`Sorry, I couldn't process that request: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Error processing message:', error);
            addAssistantMessage("Sorry, there was an error processing your request. Please try again.");
        })
        .finally(() => {
            // Re-enable input
            messageInput.disabled = false;
            sendButton.disabled = false;
        });
    }
    
    // Add a user message to the chat
    function addUserMessage(message) {
        const messageHtml = `
            <div class="user-message">
                <div class="message-content">
                    <p>${escapeHtml(message)}</p>
                </div>
                <div class="user-avatar">U</div>
            </div>
        `;
        chatContainer.insertAdjacentHTML('beforeend', messageHtml);
        scrollToBottom();
    }
    
    // Add an assistant message to the chat
    function addAssistantMessage(message) {
        // Format the message - handle lists and paragraphs
        let formattedMessage = message;
        
        // Format lists if found
        if (message.includes('\n')) {
            const lines = message.split('\n');
            let inList = false;
            let formattedLines = [];
            
            for (const line of lines) {
                if (line.trim().startsWith('-') || line.trim().startsWith('•')) {
                    if (!inList) {
                        formattedLines.push('<ul>');
                        inList = true;
                    }
                    formattedLines.push(`<li>${escapeHtml(line.trim().substring(1).trim())}</li>`);
                } else {
                    if (inList) {
                        formattedLines.push('</ul>');
                        inList = false;
                    }
                    if (line.trim()) {
                        formattedLines.push(`<p>${escapeHtml(line)}</p>`);
                    }
                }
            }
            
            if (inList) {
                formattedLines.push('</ul>');
            }
            
            formattedMessage = formattedLines.join('');
        } else {
            formattedMessage = `<p>${escapeHtml(message)}</p>`;
        }
        
        const messageHtml = `
            <div class="assistant-message">
                <div class="assistant-avatar"><img src="/static/images/chatbot_avatar.png" alt="Customisation Agent"></div>
                <div class="message-content">
                    ${formattedMessage}
                </div>
            </div>
        `;
        chatContainer.insertAdjacentHTML('beforeend', messageHtml);
        scrollToBottom();
    }
    
    // Confirm and apply changes
    function confirmChanges(config) {
        // Disable confirm button while processing
        confirmButton.disabled = true;
        
        // Send confirmation to server
        fetch('/api/assistant/confirm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ config: config }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addAssistantMessage("Changes have been applied successfully. The optimization algorithm will use these settings for the next class allocation.");
                
                // Clear current recommendation
                currentRecommendedConfig = null;
                confirmButton.style.display = 'none';
            } else {
                addAssistantMessage(`Sorry, there was an error applying the changes: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Error confirming changes:', error);
            addAssistantMessage("Sorry, there was an error applying the changes. Please try again.");
        })
        .finally(() => {
            confirmButton.disabled = false;
        });
    }
    
    // Check if message contains confirmation language
    function messageHasConfirmation(message) {
        const confirmationWords = ['yes', 'confirm', 'apply', 'accept', 'okay', 'ok', 'sure', 'good', 'great', 'perfect'];
        message = message.toLowerCase();
        return confirmationWords.some(word => message.includes(word));
    }
    
    // Check if message contains rejection language
    function messageHasRejection(message) {
        const rejectionWords = ['no', 'cancel', 'reject', 'discard', 'don\'t', 'dont', 'not', 'stop'];
        message = message.toLowerCase();
        return rejectionWords.some(word => message.includes(word));
    }
    
    // Escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Scroll chat to bottom
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Generate a random session ID
    function generateSessionId() {
        return 'session_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }
    
    // Initialize on page load
    initialize();
}); 