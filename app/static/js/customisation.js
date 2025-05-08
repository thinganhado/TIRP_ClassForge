// customisation.js - Handles functionality for customisation pages

document.addEventListener('DOMContentLoaded', function() {
    // Checkboxes for criteria
    const checkboxes = document.querySelectorAll('.criteria-checkbox');
    checkboxes.forEach(checkbox => {
        // Set up initial state
        updateCheckboxState(checkbox);
        
        // Add change event listener
        checkbox.addEventListener('change', function() {
            updateCheckboxState(this);
        });
    });
    
    function updateCheckboxState(checkbox) {
        const valueDisplay = checkbox.nextElementSibling;
        // You can add logic here to change the text if needed based on checkbox state
    }
    
    // --- Slider Value Display --- 
    const sliders = document.querySelectorAll('.criteria-slider');
    sliders.forEach(slider => {
        const valueDisplay = slider.parentElement.querySelector('.slider-value');
        if (valueDisplay) {
            const updateSliderValue = () => {
                // Display the slider's current value
                valueDisplay.textContent = slider.value;
            };
            // Set initial value
            updateSliderValue();
            // Add event listener
            slider.addEventListener('input', updateSliderValue);
        }
    });
    
    // --- Toggle Button Groups ---
    const buttonGroups = document.querySelectorAll('.criteria-button-group');
    buttonGroups.forEach(group => {
        const buttons = group.querySelectorAll('.criteria-button');
        const hiddenInput = group.querySelector('input[type="hidden"]');
        
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons in this group
                buttons.forEach(btn => btn.classList.remove('active'));
                // Add active class to clicked button
                this.classList.add('active');
                // Update hidden input value
                if (hiddenInput) {
                    hiddenInput.value = this.dataset.value;
                }
            });
        });
    });
    
    // --- Tab Navigation ---
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Add tab switching logic here if needed
        });
    });
    
    // --- Drag and Drop for Priority Items --- 
    let draggedItem = null;
    const priorityList = document.getElementById('priority-sortable');
    
    if (priorityList) {
        document.querySelectorAll('.priority-item').forEach(item => {
            item.addEventListener('dragstart', function() {
                draggedItem = this;
                setTimeout(() => this.classList.add('dragging'), 0);
            });
            
            item.addEventListener('dragend', function() {
                this.classList.remove('dragging');
                draggedItem = null;
                updatePriorityWeights();
            });
        });

        priorityList.addEventListener('dragover', function(e) {
            e.preventDefault();
            const afterElement = getDragAfterElement(priorityList, e.clientY);
            if (draggedItem) {
                if (afterElement == null) {
                    priorityList.appendChild(draggedItem);
                } else {
                    priorityList.insertBefore(draggedItem, afterElement);
                }
            }
        });
        
        // Initial weight setting
        updatePriorityWeights(); 
    }

    function getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('.priority-item:not(.dragging)')];
        
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }
    
    function updatePriorityWeights() {
        const items = priorityList.querySelectorAll('.priority-item');
        items.forEach((item, index) => {
            const weight = items.length - index; // Higher position = higher weight
            item.dataset.weight = weight;
            const weightInput = item.querySelector('.weight-input');
            if (weightInput) {
                weightInput.value = weight;
            }
        });
    }
    
    // Reset button functionality
    const resetButton = document.querySelector('.reset-button');
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            if (confirm('Reset all settings to default values?')) {
                // Reset Hard Constraints
                document.getElementById('class-size').value = 25;
                document.getElementById('max-classes').value = 4;
                document.getElementById('min-friends').value = 1;
                
                // Reset Criteria Sliders to default values
                document.getElementById('grade-variance-slider').value = 34; 
                document.getElementById('influential-threshold-slider').value = 60;
                document.getElementById('isolated-threshold-slider').value = 60;
                
                // Trigger input event to update display values
                sliders.forEach(slider => {
                    slider.dispatchEvent(new Event('input'));
                }); 

                // Reset Button Groups
                buttonGroups.forEach(group => {
                    const offButton = group.querySelector('.criteria-button[data-value="false"]');
                    const onButton = group.querySelector('.criteria-button[data-value="true"]');
                    const hiddenInput = group.querySelector('input[type="hidden"]');
                    
                    if (offButton && onButton && hiddenInput) {
                        offButton.classList.remove('active');
                        onButton.classList.add('active');
                        hiddenInput.value = 'true';
                    }
                });

                // Reset Dropdowns
                document.getElementById('bully-victim-handling').value = 'separate';
                document.getElementById('disruptive-handling').value = 'distribute';
                document.getElementById('special-needs-handling').value = 'resource-optimize';
                document.getElementById('gifted-handling').value = 'cluster';

                // Reset Priorities Order
                resetPriorityOrder();
                
                console.log('Settings reset to default values.');
            }
        });
    }
    
    function resetPriorityOrder() {
        // Default order: academic, wellbeing, bullying, influence, friendship
        const defaultOrder = ['academic', 'wellbeing', 'bullying', 'influence', 'friendship'];
        const priorityMap = {};
        
        // Create a map of existing items
        document.querySelectorAll('.priority-item').forEach(item => {
            priorityMap[item.dataset.objective] = item;
        });
        
        // Clear the list
        priorityList.innerHTML = '';
        
        // Add items back in default order
        defaultOrder.forEach((objective, index) => {
            if (priorityMap[objective]) {
                priorityList.appendChild(priorityMap[objective]);
            }
        });
        
        // Update weights
        updatePriorityWeights();
    }
    
    // Apply settings button functionality
    const applyButton = document.querySelector('.apply-button');
    if (applyButton) {
        applyButton.addEventListener('click', function() {
            const settings = {
                hardConstraints: {
                    classSize: parseInt(document.getElementById('class-size').value),
                    maxClasses: parseInt(document.getElementById('max-classes').value),
                    minFriends: parseInt(document.getElementById('min-friends').value)
                },
                criteria: {
                    academic: {
                        minimizeVariance: parseInt(document.getElementById('grade-variance-slider').value),
                        ensureDistribution: document.getElementById('ensure-grade-distribution').checked,
                    },
                    wellbeing: {
                        balanceLow: document.getElementById('balance-low-wellbeing').checked,
                        supportivePairing: document.getElementById('supportive-pairing-wellbeing').checked,
                        distributePairs: document.getElementById('distribute-wellbeing-pairs').checked,
                    },
                    bullying: {
                        handling: document.getElementById('bully-victim-handling').value,
                        protectVulnerable: document.getElementById('protect-vulnerable').checked,
                    },
                    influence: {
                        influentialThreshold: parseInt(document.getElementById('influential-threshold-slider').value),
                        distributeInfluential: document.getElementById('distribute-influential').checked,
                        isolatedThreshold: parseInt(document.getElementById('isolated-threshold-slider').value),
                        distributeIsolated: document.getElementById('distribute-isolated').checked,
                    },
                    friendship: {
                        maximizeConnections: document.getElementById('maximize-friendships').checked,
                        balanceDensity: document.getElementById('balance-friendship-density').checked,
                    }
                },
                priorities: [],
                specialConsiderations: {
                    disruptive: document.getElementById('disruptive-handling').value,
                    specialNeeds: document.getElementById('special-needs-handling').value,
                    gifted: document.getElementById('gifted-handling').value,
                }
            };
            
            // Get priorities based on current order and weights
            document.querySelectorAll('#priority-sortable .priority-item').forEach(item => {
                settings.priorities.push({
                    objective: item.dataset.objective,
                    weight: parseInt(item.querySelector('.weight-input').value)
                });
            });
            
            // Send to server (placeholder)
            console.log('Settings to be applied:', JSON.stringify(settings, null, 2));
            
            // Show success feedback
            const successMsg = document.createElement('div');
            successMsg.className = 'success-message';
            successMsg.textContent = 'Settings applied successfully!';
            document.querySelector('.customisation-container').appendChild(successMsg);
            
            // Remove after 3 seconds
            setTimeout(() => {
                successMsg.remove();
            }, 3000);
        });
    }
}); 