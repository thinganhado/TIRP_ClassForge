# Classroom Optimization Assistant Models

This folder contains the implementation of both rule-based and machine learning models designed to help optimize classroom settings based on various educational priorities.

## Overview

The `assistant.py` module implements:

1. A rule-based chatbot that uses scikit-learn's TF-IDF vectorizer for natural language processing
2. A machine learning-based recommendation system using LogisticRegression with TF-IDF features

The assistant can:
- Understand and respond to queries about different educational priorities
- Make recommendations for configuration settings based on user input
- Detect and respond appropriately to greetings and inappropriate language
- Provide context-aware responses using ML prediction and similarity matching with teacher comments

## Core Components

- **RuleBasedChatbot**: The main chatbot class with both rule-based and ML functionality
- **AssistantModel**: A wrapper class for backward compatibility with existing code

## Machine Learning Features

### ML Pipeline
The system implements a scikit-learn pipeline for query classification:
```python
ml_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])
```

This pipeline:
1. Converts text input into TF-IDF features (Term Frequency-Inverse Document Frequency)
2. Uses LogisticRegression to classify the query into the most appropriate recommendation category

### Training Data
The model is trained on teacher comments and recommendations from `cleaned_teacher_comments (1).csv`, which provides paired examples of:
- Student situations/challenges (input)
- Appropriate optimization recommendations (output)

### Advantages Over Rule-Based Approach
- Better handles misspellings and variations in wording
- Can generalize to new queries not explicitly defined in rules
- Improves response relevance through statistical learning
- Adapts to the specific language patterns used in educational contexts

## Rule-Based Features

1. **Priority-Based Configuration**
   - Academic performance
   - Student wellbeing
   - Bullying prevention
   - Social dynamics
   - Friendship connections

2. **Natural Language Understanding**
   - Category detection based on keywords
   - Priority change detection
   - TF-IDF based similarity matching

3. **Conversation Management**
   - Greeting detection
   - Inappropriate language filtering
   - Conversation history tracking

## Data Files

- `cleaned_teacher_comments (1).csv`: Contains teacher comments and corresponding recommendations used for both ML training and similarity matching

## How to Use

### Basic Usage

```python
from app.models.assistant import chatbot

# Get a response using the ML model
response = chatbot.analyze_request("I need help with improving student wellbeing")
print(response["message"])

# Get a direct ML recommendation
recommendation = chatbot.get_recommendation_ml("prioritize friendship and social connections")
print(recommendation)

# Get the current configuration
config = chatbot.get_current_config()

# Save a modified configuration
chatbot.save_config(config)

# Get chat history
history = chatbot.get_chat_history(limit=5)

# Clear chat history
chatbot.clear_chat_history()
```

### Example Queries

- "How can I improve academic performance in my classes?"
- "I need to focus more on student wellbeing"
- "Help me prevent bullying in the classroom"
- "How can I optimize social dynamics in my class?"
- "I want to ensure students have enough friends in class"

## Configuration Parameters

The chatbot manages the following configuration parameters:

| Parameter | Description |
|-----------|-------------|
| `class_size` | Maximum number of students per class |
| `max_classes` | Maximum number of classes |
| `gpa_penalty_weight` | Weight for GPA balance penalty |
| `wellbeing_penalty_weight` | Weight for student wellbeing penalty |
| `bully_penalty_weight` | Weight for bullying prevention |
| `influence_std_weight` | Weight for social influence standard deviation |
| `isolated_std_weight` | Weight for isolated student penalty |
| `min_friends_required` | Minimum number of friends required per student |
| `friend_inclusion_weight` | Weight for friend inclusion penalty |
| `friendship_balance_weight` | Weight for friendship balance |
| `prioritize_academic` | Priority level for academic performance (1-5) |
| `prioritize_wellbeing` | Priority level for student wellbeing (1-5) |
| `prioritize_bullying` | Priority level for bullying prevention (1-5) |
| `prioritize_social_influence` | Priority level for social influence (1-5) |
| `prioritize_friendship` | Priority level for friendship connections (1-5) |

## Dependencies

- pandas
- numpy
- scikit-learn (for TF-IDF vectorization, LogisticRegression, and similarity metrics)
- NetworkX (optional, for social network analysis) 