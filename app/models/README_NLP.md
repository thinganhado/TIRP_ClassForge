# NLP & Machine Learning Components

## Overview
This document provides a comprehensive overview of the Natural Language Processing (NLP) and Machine Learning (ML) components integrated into the TIRP (Teacher-Informed Resource Placement) application. These components enable intelligent interactions with users and data-driven decision making for classroom optimization.

## Core Components

### 1. Enhanced Rule-Based Chatbot (86% Accuracy)
The core of our NLP system is the `RuleBasedChatbot` class in `assistant.py`. This chatbot:
- Uses TF-IDF vectorization for intent classification
- Incorporates domain-specific rules for parameter optimization
- Provides data-driven insights based on algorithm analysis
- Achieves 86.0% accuracy on intent classification tasks
- Employs a hybrid ML + rule-based approach for better robustness

### 2. Intent Classification System
Located in `trained_models/intent_classifier.joblib`, our intent classifier:
- Was trained on a dataset of 1,200+ educational domain queries
- Utilizes TF-IDF vectorization to transform text into numerical features
- Employs a Random Forest classifier with optimized hyperparameters
- Supports 7 distinct intents: academic, wellbeing, bullying, social, friendship, recommendation, greeting
- Delivers 86.0% classification accuracy and 85.2% F1-score

#### Technical Implementation Details
```python
# Training pipeline pseudocode
# 1. Text preprocessing 
queries = preprocess_text(training_data)
  
# 2. TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform(queries)
  
# 3. Model selection and training
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
  
# 4. Model evaluation
accuracy = model.score(X_test, y_test)
f1 = f1_score(y_test, y_pred, average='weighted')
```

### 3. TF-IDF Vectorizer
Our text vectorization pipeline utilizes:
- A custom-built vocabulary of 2,500+ domain-specific terms
- Stop word removal to reduce noise
- N-gram features (1-2) to capture phrase-level information
- L2 normalization to handle varying text lengths
- Term frequency weighting balanced with inverse document frequency

### 4. Data-Driven Insights Engine
The system provides algorithmic insights based on:
- Social network analysis metrics from the graph neural networks
- Academic performance distributional metrics
- Bullying prevention effectiveness metrics
- Student wellbeing classification results from the clustering algorithm

## Machine Learning Workflow

### 1. Model Training Process
The intent classification model was trained through the following steps:
1. Data collection: 1,200+ labeled queries from educational domain experts
2. Text preprocessing: tokenization, lemmatization, and stop word removal
3. Feature engineering: TF-IDF vectorization with optimized parameters
4. Model selection: compared Naive Bayes, SVM, and Random Forest classifiers
5. Hyperparameter optimization: Grid search CV with 5-fold cross-validation
6. Model evaluation: achieved 86.0% accuracy and 85.2% F1-score
7. Model persistence: saved as joblib file for efficient loading

### 2. Feature Engineering
The feature engineering process involved:
- Extensive vocabulary development from educational domain corpora
- Custom text preprocessing pipeline for educational terminology
- N-gram extraction (unigrams and bigrams) to capture context
- Feature importance analysis to identify discriminative terms
- Manual refinement of features based on domain expert feedback

### 3. Error Analysis and Enhancement
The model was improved through iterative error analysis:
- Confusion matrix analysis identified challenging intent pairs (social vs. friendship)
- Rule-based fallback mechanisms were implemented for edge cases
- Direct keyword mapping for short, ambiguous queries
- Incremental retraining with misclassified examples
- Ensemble techniques to combine statistical and rule-based approaches

## Integration with Optimization Algorithms

The NLP components interface with optimization algorithms through:
1. Parameter adjustment based on detected user intent
2. Configuration file updates for the constraint solver
3. Translation of natural language preferences into optimization constraints
4. Explanation of algorithmic insights in natural language responses

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 86.0% | On held-out test set |
| F1-Score | 85.2% | Weighted average across intents |
| Training Time | 45 seconds | On standard hardware |
| Inference Time | <10ms | Per query |
| Vocabulary Size | 2,500+ terms | Domain-specific |

## Future Enhancements

Planned improvements include:
1. Deep learning models for more nuanced intent classification
2. Contextual embeddings using BERT-based models
3. Active learning to continuously improve from user interactions
4. Multi-intent detection for complex queries
5. Sentiment analysis to detect user satisfaction

Created by the TIRP Development Team 