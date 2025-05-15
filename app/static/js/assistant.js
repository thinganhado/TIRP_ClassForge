// assistant.js - Handles client-side functionality for the AI assistant
// Works with both full-page assistant and widget styles

document.addEventListener('DOMContentLoaded', function() {
    // Initialize full-page assistant if present
    initAssistant();
    
    // Initialize widget if present
    initWidget();
    
    /**
     * Initialize the full-page assistant
     */
    function initAssistant() {
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('chat-message');
        const sendButton = document.getElementById('send-button');
        const confirmButton = document.querySelector('.confirm-button');
        const refreshButton = document.getElementById('refresh-button');
        
        if (!chatContainer || !messageInput || !sendButton) return; // Exit if elements don't exist
        
        // Session ID for tracking conversation
        const sessionId = chatContainer.dataset.sessionId || generateSessionId();
    
    // Event listeners
        sendButton.addEventListener('click', function() {
            sendAssistantMessage(messageInput.value, chatContainer, messageInput, sessionId, confirmButton);
        });
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendAssistantMessage(messageInput.value, chatContainer, messageInput, sessionId, confirmButton);
            }
        });
        
        if (refreshButton) {
            refreshButton.addEventListener('click', function() {
                if (confirm('Are you sure you want to reset the conversation? This will clear all chat history.')) {
                    window.location.href = window.location.pathname + "?reset=true";
                }
            });
        }
        
        // Scroll to bottom of chat on page load
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    /**
     * Initialize the widget assistant
     */
    function initWidget() {
        const chatbotWidget = document.getElementById('chatbot-widget');
        if (!chatbotWidget) return; // Exit if widget doesn't exist
        
        // Get widget elements
        const chatbotIcon = document.getElementById('chatbot-icon');
        const chatbotFull = document.getElementById('chatbot-full');
        const chatbotToggle = document.getElementById('chatbot-toggle');
        const chatbotInput = document.getElementById('chatbot-input');
        const chatbotSend = document.getElementById('chatbot-send');
        const chatbotMessages = document.getElementById('chatbot-messages');
        const refreshChat = document.getElementById('refresh-chat');
        
        if (!chatbotMessages || !chatbotInput || !chatbotSend) return;
        
        // Session ID for tracking conversation
        const sessionId = chatbotWidget.dataset.sessionId || generateSessionId();
        
        // Event listeners for showing/hiding widget
        if (chatbotIcon && chatbotFull && chatbotToggle) {
            // Show full chatbot when icon is clicked
            chatbotIcon.addEventListener('click', function() {
                chatbotIcon.style.display = 'none';
                chatbotFull.style.display = 'flex';
                
                // Focus on input field when opened
                setTimeout(() => {
                    chatbotInput.focus();
                }, 300);
            });
            
            // Hide full chatbot when toggle button is clicked
            chatbotToggle.addEventListener('click', function() {
                chatbotFull.style.display = 'none';
                chatbotIcon.style.display = 'flex';
            });
        }
        
        // Event listeners for chat functionality
        chatbotSend.addEventListener('click', function() {
            sendWidgetMessage(chatbotInput.value, chatbotMessages, chatbotInput, sessionId);
        });
        
        chatbotInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendWidgetMessage(chatbotInput.value, chatbotMessages, chatbotInput, sessionId);
            }
        });
        
        // Reset chat history functionality
        if (refreshChat) {
            refreshChat.addEventListener('click', function() {
                if (confirm('Are you sure you want to reset the conversation? This will clear all chat history.')) {
                    // Clear messages except the first one (welcome message)
                    while (chatbotMessages.children.length > 1) {
                        chatbotMessages.removeChild(chatbotMessages.lastChild);
                    }
                    
                    // Reset session in backend
                    fetch('/api/assistant/reset?session_id=' + sessionId)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            addWidgetMessage('Chat history has been reset.', 'assistant', chatbotMessages);
                        }
                    })
                    .catch(error => {
                        console.error('Error resetting chat:', error);
                    });
                }
            });
        }
        
        // Load chat history if available
        if (sessionId) {
            fetch(`/api/assistant/chat_history?session_id=${sessionId}&limit=5`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.history && data.history.length > 0) {
                    // Add notification dot to icon if messages exist
                    if (data.history.length > 0 && chatbotIcon) {
                        addNotificationDot(chatbotIcon);
                    }
                    
                    // Load last 5 messages
                    data.history.reverse().forEach(entry => {
                        if (entry.user_input) {
                            addWidgetMessage(entry.user_input, 'user', chatbotMessages);
                        }
                        if (entry.response) {
                            addWidgetMessage(entry.response, 'assistant', chatbotMessages);
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
            });
        }
    }
    
    /**
     * Send a message in the full-page assistant
     */
    function sendAssistantMessage(message, chatContainer, messageInput, sessionId, confirmButton) {
        message = message.trim();
        if (!message) return;
        
        // Add user message to chat
        addAssistantMessage(message, 'user', chatContainer);
        
        // Clear input
        messageInput.value = '';
        
        // Show typing indicator
        addAssistantMessage('<em>Thinking...</em>', 'assistant', chatContainer);
        const typingIndicator = chatContainer.lastChild;
        
        // Send to backend
        fetch('/api/assistant/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                input: message,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            chatContainer.removeChild(typingIndicator);
            
            // Add response
            if (data.success) {
                addAssistantMessage(data.message, 'assistant', chatContainer);
                
                // If there are config changes, add a confirm button
                if (data.is_modified && confirmButton) {
                    confirmButton.style.display = 'block';
                }
                
                // If the message contains reference to "Set Priorities", add redirection button
                if (data.message.includes("Set Priorities") || data.message.includes("set priorities")) {
                    addRedirectButton(chatContainer);
                }
            } else {
                // Handle specific error cases
                if (data.message && data.message.includes('friendship_score_weight')) {
                    const errorMsg = "Sorry, there was an error applying your changes. The system doesn't recognize 'friendship_score_weight' as a valid parameter. Please try using 'friend_inclusion_weight' or 'friend_balance_weight' instead.";
                    addAssistantMessage(errorMsg, 'assistant', chatContainer);
                    
                    // Suggest going to set priorities page manually
                        setTimeout(() => {
                        addRedirectButton(chatContainer);
                    }, 500);
                } else if (data.message && data.message.includes("Error applying changes")) {
                    addAssistantMessage("Sorry, there was an error applying your changes: " + data.message, 'assistant', chatContainer);
                } else {
                    addAssistantMessage("Sorry, I couldn't process your request. Please try again.", 'assistant', chatContainer);
                }
            }
        })
        .catch(error => {
            // Remove typing indicator
            chatContainer.removeChild(typingIndicator);
            
            console.error('Error:', error);
            addAssistantMessage("Sorry, there was an error processing your request. Please try again.", 'assistant', chatContainer);
        });
    }
    
    /**
     * Send a message in the widget
     */
    function sendWidgetMessage(message, chatbotMessages, chatbotInput, sessionId) {
        message = message.trim();
        if (!message) return;
        
        // Add user message to chat
        addWidgetMessage(message, 'user', chatbotMessages);
        
        // Clear input
        chatbotInput.value = '';
        
        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message assistant-message typing';
        typingIndicator.innerHTML = '<div class="message-content">Thinking...</div>';
        chatbotMessages.appendChild(typingIndicator);
        
        // Scroll to bottom
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        
        // Send to backend
        fetch('/api/assistant/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                input: message,
                session_id: sessionId 
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            chatbotMessages.removeChild(typingIndicator);
            
            // Add response
            if (data.success) {
                addWidgetMessage(data.message, 'assistant', chatbotMessages);
                
                // If there are config changes, add a confirm button
                if (data.is_modified) {
                    addWidgetConfirmButton(chatbotMessages);
                }
                
                // If the message contains reference to "Set Priorities", suggest it
                if (data.message.includes("Set Priorities") || data.message.includes("set priorities")) {
                    addWidgetMessage("Would you like to go to the Set Priorities page to make these changes?", 'assistant', chatbotMessages);
                    addWidgetLinkButton(chatbotMessages, "Go to Set Priorities", "/customisation/set-priorities");
                }
            } else {
                // Handle specific error cases
                if (data.message && data.message.includes('friendship_score_weight')) {
                    const errorMsg = "Sorry, there was an error applying your changes. The system doesn't recognize 'friendship_score_weight' as a valid parameter. Please try using 'friend_inclusion_weight' or 'friend_balance_weight' instead.";
                    addWidgetMessage(errorMsg, 'assistant', chatbotMessages);
                    
                    // Suggest going to set priorities page manually
                        setTimeout(() => {
                        addWidgetMessage("Would you like to adjust settings directly?", 'assistant', chatbotMessages);
                        addWidgetLinkButton(chatbotMessages, "Go to Set Priorities", "/customisation/set-priorities");
                    }, 500);
                } else if (data.message && data.message.includes("Error applying changes")) {
                    addWidgetMessage("Sorry, there was an error applying your changes: " + data.message, 'assistant', chatbotMessages);
                } else {
                    addWidgetMessage("Sorry, I couldn't process your request. Please try again.", 'assistant', chatbotMessages);
                }
            }
        })
        .catch(error => {
            // Remove typing indicator
            chatbotMessages.removeChild(typingIndicator);
            console.error('Error:', error);
            addWidgetMessage("Sorry, there was an error processing your request. Please try again.", 'assistant', chatbotMessages);
        });
    }
    
    /**
     * Add a message to the full-page assistant chat
     */
    function addAssistantMessage(content, sender, chatContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${sender}-message`;
        
        if (sender === 'assistant') {
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'assistant-avatar';
            const avatarImg = document.createElement('img');
            avatarImg.src = "/static/images/chatbot_avatar.png";
            avatarImg.alt = "Assistant";
            avatarDiv.appendChild(avatarImg);
            messageDiv.appendChild(avatarDiv);
        } else {
            const userAvatar = document.createElement('div');
            userAvatar.className = 'user-avatar';
            userAvatar.textContent = 'U';
            messageDiv.appendChild(userAvatar);
        }
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = content.replace(/\n/g, '<br>');
        
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    /**
     * Add a message to the widget chat
     */
    function addWidgetMessage(content, sender, chatbotMessages) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = content.replace(/\n/g, '<br>');
        
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = 'Just now';
        
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(timestamp);
        
        chatbotMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    /**
     * Add a redirect button to the full-page assistant
     */
    function addRedirectButton(chatContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'assistant-message';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'assistant-avatar';
        const avatarImg = document.createElement('img');
        avatarImg.src = "/static/images/chatbot_avatar.png";
        avatarImg.alt = "Assistant";
        avatarDiv.appendChild(avatarImg);
        messageDiv.appendChild(avatarDiv);
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Create button
        const redirectButton = document.createElement('a');
        redirectButton.href = "/customisation/set-priorities";
        redirectButton.className = 'confirm-button';
        redirectButton.style.display = 'inline-block';
        redirectButton.style.marginTop = '10px';
        redirectButton.style.textDecoration = 'none';
        redirectButton.style.textAlign = 'center';
        redirectButton.style.padding = '8px 16px';
        redirectButton.textContent = 'Go to Set Priorities Page';
        
        messageContent.innerHTML = "Would you like to go to the Set Priorities page to make these changes?<br>";
        messageContent.appendChild(redirectButton);
        
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    /**
     * Add a confirmation button to the widget
     */
    function addWidgetConfirmButton(chatbotMessages) {
        const buttonDiv = document.createElement('div');
        buttonDiv.className = 'message assistant-message confirmation';
        
        const buttonContent = document.createElement('div');
        buttonContent.className = 'message-content';
        
        const confirmButton = document.createElement('a');
        confirmButton.href = "/customisation/set-priorities";
        confirmButton.className = 'widget-confirm-button';
        confirmButton.textContent = 'Go to Settings';
        confirmButton.style.display = 'inline-block';
        confirmButton.style.padding = '8px 12px';
        confirmButton.style.background = '#0056b3';
        confirmButton.style.color = 'white';
        confirmButton.style.borderRadius = '4px';
        confirmButton.style.textDecoration = 'none';
        confirmButton.style.margin = '5px 0';
        
        buttonContent.appendChild(confirmButton);
        buttonDiv.appendChild(buttonContent);
        
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = 'Just now';
        buttonDiv.appendChild(timestamp);
        
        chatbotMessages.appendChild(buttonDiv);
        
        // Scroll to bottom
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    /**
     * Add a link button to the widget
     */
    function addWidgetLinkButton(chatbotMessages, text, url) {
        const buttonDiv = document.createElement('div');
        buttonDiv.className = 'message assistant-message link-button';
        
        const buttonContent = document.createElement('div');
        buttonContent.className = 'message-content';
        
        const linkButton = document.createElement('a');
        linkButton.href = url;
        linkButton.className = 'widget-link-button';
        linkButton.textContent = text;
        linkButton.style.display = 'inline-block';
        linkButton.style.padding = '8px 12px';
        linkButton.style.background = '#0056b3';
        linkButton.style.color = 'white';
        linkButton.style.borderRadius = '4px';
        linkButton.style.textDecoration = 'none';
        linkButton.style.margin = '5px 0';
        
        buttonContent.appendChild(linkButton);
        buttonDiv.appendChild(buttonContent);
        
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = 'Just now';
        buttonDiv.appendChild(timestamp);
        
        chatbotMessages.appendChild(buttonDiv);
        
        // Scroll to bottom
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    /**
     * Generate a random session ID
     */
    function generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Add a notification dot to the chatbot icon
     */
    function addNotificationDot(chatbotIcon) {
        // Only add if it doesn't already exist
        if (!chatbotIcon.querySelector('.notification-dot')) {
            const notificationDot = document.createElement('div');
            notificationDot.className = 'notification-dot';
            chatbotIcon.appendChild(notificationDot);
        }
    }
}); 