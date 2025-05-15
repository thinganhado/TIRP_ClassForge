# TIRP NLP Model and Database Integration

## Overview

This document provides information about the Natural Language Processing (NLP) models in the TIRP application and their integration with the database.

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

## Database Integration

### Direct Database Connection

The application can now connect directly to the Amazon RDS MySQL database:

```
Host: database-tirp.c1gieoo4asys.us-east-1.rds.amazonaws.com
Port: 3306
User: admin
Database: tirp
```

Key tables utilized:
- `participants`: Student demographic and performance data (176 records)
- `mental_wellbeing`: Mental wellbeing percentages (176 records)
- `academic_wellbeing`: Academic performance metrics (176 records)
- `social_wellbeing`: Social wellbeing metrics (176 records)
- `net_friends`: Friendship network connections (1169 records)
- `net_influential`: Influence network connections (276 records)
- `net_disrespect`: Negative interactions for bullying detection
- `soft_constraints`: Configuration parameters for optimization (126 records)

### Connection Methods

The application supports two methods of database connectivity:

1. **Flask-SQLAlchemy Context**: Used when running within the Flask web application
   - Requires Flask application context
   - Managed through the standard Flask-SQLAlchemy session

2. **Direct SQLAlchemy Connection**: Used for standalone scripts or testing
   - Does not require Flask application context
   - Uses SQLAlchemy engine with direct connection string
   - Enables database access outside of the web application

### Fallback Mechanism

If database connection fails, the system falls back to CSV files located in:
- `app/ml_models/Clustering/output/wellbeing_classification_results.csv`
- `app/ml_models/ha_outputs/community_bully_assignments.csv`
- `app/ml_models/ha_outputs/gpa_predictions_with_bins.csv`
- `app/ml_models/Clustering/output/social_wellbeing_predictions.csv`

## Testing

Two test scripts are available to verify NLP and database functionality:

1. **test_nlp_models.py**: Tests the NLP models with database integration
   - Verifies TF-IDF vectorizer training
   - Tests ML model availability and type
   - Validates recommendation system functionality
   - Tests NLP response generation for various queries
   - Supports both database and CSV data sources

2. **direct_db_test.py**: Tests direct database connection
   - Verifies RDS connectivity
   - Queries data from all relevant tables
   - Displays database schema information

## Usage

To use the NLP models with direct database connection:

```python
from app.models.database_loader import DatabaseLoader
from app.models.assistant import RuleBasedChatbot

# Create database loader with direct connection
db_loader = DatabaseLoader(direct_connect=True)

# Load data (password can be provided here)
db_loader.load_data(password='your_password')

# Create chatbot with the database loader
chatbot = RuleBasedChatbot(db_loader=db_loader)

# Use the chatbot
response = chatbot.analyze_request("How can I improve student wellbeing?")
print(response['message'])
```

## Known Issues

1. The Flask-SQLAlchemy connection fails outside of Flask application context with error:
   `No application found. Either work inside a view function or push an application context.`

2. Direct RDS connection requires password authentication, which should be handled securely.

3. When database connection fails, the system falls back to CSV files, which may contain outdated data.

## Future Improvements

1. Implement secure password handling using environment variables or secure storage
2. Add caching layer to reduce database queries
3. Create synchronization mechanism between database and CSV files
4. Enhance error handling for intermittent connection issues
5. Implement connection pooling for better performance 