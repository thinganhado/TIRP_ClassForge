# Trained Models Directory

This directory contains pre-trained models for the TIRP application.

## Required Models

The following models are used by the application:

1. `intent_classifier.pkl`: Trained classifier to identify intents from user queries
2. `recommendation_generator.pkl`: Model to generate recommendations based on intents

## Training New Models

To train new models, use the `DataManager.train_and_save_model()` method in development mode:

```python
from app.models.data_manager import DataManager

# Create data manager in development mode
data_manager = DataManager(environment="development")

# Train and save a model
data_manager.train_and_save_model(
    model_name="intent_classifier",
    model_type="logistic",
    X=feature_data,
    y=label_data
)
```

## Model Metadata

Each model should have an accompanying metadata file with the same name and a `.json` extension 
(e.g., `intent_classifier_metadata.json`). This file contains information about the model's 
features, performance metrics, and training data.
