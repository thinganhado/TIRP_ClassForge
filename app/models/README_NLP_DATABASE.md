# TIRP NLP Model and API Integration

## Overview

This document provides information about the Natural Language Processing (NLP) models in the TIRP application and their integration with the API and database.

## NLP Models

The TIRP application uses multiple NLP techniques to understand and respond to user queries:

1. **TF-IDF Vectorization**: Converts text into numerical features using term frequency-inverse document frequency
   - Vocabulary size: ~224 terms
   - Trained on teacher comments and educational domain terminology

2. **ML Classification Pipeline**: Uses a scikit-learn pipeline with LogisticRegression
   - Trained on synthetic teacher comments generated from student data
   - Predicts appropriate recommendations based on user queries

3. **Intent Recognition**: Detects intents related to:
   - Academic performance
   - Student wellbeing
   - Bullying prevention
   - Social dynamics
   - Friendship connections

4. **Advanced Recommendation System**: Provides data-driven recommendations based on:
   - Wellbeing analysis (56% of students have high wellbeing)
   - Bullying detection (13% of students identified as bullies)
   - GPA statistics (average GPA of 51.3)
   - Social network analysis
   - Correlations between various factors (e.g., wellbeing and social connections)

## API and Database Integration

### API-First Architecture

The application follows an API-first architecture:

1. **Primary Access via API**: All data access attempts are made via API endpoints first
2. **Database Module Fallback**: If API is unavailable, uses database query modules as fallback
3. **File-Based Fallback**: If both API and database fail, falls back to local CSV/JSON files

This layered approach ensures robustness while maintaining separation of concerns.

### Key Database Query Modules

The application leverages these existing query modules from the `/app/database` directory:
- `class_queries.py`: Class-related database operations
- `student_queries.py`: Student-related database operations
- `softcons_queries.py`: Soft constraints configuration operations
- `friends_queries.py`: Friendship network operations

### Data Sources

If API and database access fail, the system falls back to CSV files located in:
- `app/ml_models/Clustering/output/cluster_assignments.csv`: Social wellbeing classifications with feature values
- `app/ml_models/ha_outputs/community_bully_assignments.csv`: Bullying community assignments
- `app/ml_models/ha_outputs/gpa_predictions_with_bins.csv`: GPA predictions and bins

## Model Files in Output Directory

The `/app/ml_models/Clustering/output/` directory contains essential files for the NLP models:

### Required for Production:
- `cluster_assignments.csv`: Student wellbeing classifications with features (REQUIRED)
- `rgcn_classification_model.pt`: Trained RGCN model for classification (REQUIRED)
- `model_metadata.json`: Model metadata in JSON format (REQUIRED)
- `model_metadata.pt`: Model metadata in PyTorch format (REQUIRED)
- `cluster_profiles.csv`: Statistical profiles of each cluster (REQUIRED)
- `wellbeing_labels.json`: Mapping of cluster numbers to wellbeing labels (REQUIRED)

### Optional Analytics Files:
- `cluster_centers.csv`: Coordinates of cluster centers
- `k_evaluation_results.json`: Results of optimal k evaluation
- `pca_loadings.csv`: PCA component loadings
- `rgcn_training_losses.json`: Training loss history
- `student_wellbeing_classifications.csv`: Simplified classification results (redundant with cluster_assignments.csv)

## Usage

To use the NLP models with the API client:

```python
from app.models.api_client import api_client
from app.models.assistant import RuleBasedChatbot

# Create chatbot with the API client
chatbot = RuleBasedChatbot(api_client_instance=api_client)

# Use the chatbot
response = chatbot.analyze_request("How can I improve student wellbeing?")
print(response['message'])
```

## Caching

The system implements a caching mechanism to improve performance:

1. API responses are cached in memory during runtime
2. Data is also cached to disk as Parquet files in `app/cache/`
3. Cache is used when API is unavailable or for repeated requests

## Future Improvements

1. Add more comprehensive API endpoint coverage
2. Enhance API client with retry logic for intermittent failures
3. Improve caching strategy with TTL (Time To Live) for cached data
4. Create synchronization mechanism between API, database and CSV files
5. Implement connection pooling for better performance 