# TF-IDF Vectorizer for Classroom Optimization Assistant

This module contains the TF-IDF vectorizer used in the classroom optimization chatbot for text similarity matching and natural language processing.

## Overview

The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used to convert text inputs into numerical vectors that can be compared for similarity. This is essential for the chatbot to understand user requests and find the most relevant responses.

## Features

- Converts text to numerical feature vectors
- Removes common English stop words
- Calculates cosine similarity between user queries and teacher comments
- Provides semantic understanding for category detection

## Technical Details

The vectorizer is initialized with the corpus of teacher comments from the `teacher_comments.csv` file. This ensures that the vocabulary is tailored to the educational domain and can effectively match user queries with appropriate responses.

### Configuration

The vectorizer is configured with the following parameters:

- **stop_words**: 'english' (removes common English words like "the", "and", etc.)
- **max_features**: None (uses all features found in the corpus)
- **ngram_range**: (1, 1) (only single words are considered)
- **norm**: 'l2' (applies L2 normalization to the feature vectors)

## Usage

The vectorizer is primarily used in the `get_similar_comment` method of the `RuleBasedChatbot` class to find the most similar teacher comment based on cosine similarity.

## Files

- `config.json`: Configuration file for the vectorizer
- `vectorizer_info.json`: Generated metadata about the vectorizer's state

## Dependencies

- scikit-learn: For TF-IDF implementation and similarity calculations
- numpy: For numerical operations
- pandas: For data handling 