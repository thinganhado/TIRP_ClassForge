# Social Wellbeing Analysis Model

This project implements a comprehensive workflow for analyzing and predicting student social wellbeing using machine learning and graph neural networks.

## Overview

The model workflow follows this sequence:

1. **Data Preprocessing & Cleaning**: Cleans, handles missing values, detects outliers, and standardizes survey data
2. **Clustering Analysis**: Groups students with similar survey response patterns
3. **Principal Component Analysis (PCA)**: Creates a synthetic social wellbeing score
4. **Relational Graph Convolutional Network (RGCN)**: Predicts wellbeing scores using both survey features and social network connections

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Format

The model expects an Excel file with the following sheets:
- `responses`: Survey responses from students
- `participants`: Student demographic information
- `net_0_Friends`: Friendship network connections (source-target pairs)
- `net_5_Disrespect`: Disrespect network connections (optional)

Required survey columns:
- School_support_engage6
- Manbox5_overall
- Masculinity_contrained
- pwi_wellbeing

### Data Cleaning Process

The model performs comprehensive data cleaning:

1. **Distribution Analysis**: Checks for skewness and outliers in the data
2. **Missing Value Handling**: Imputes missing values using median, mean, or KNN methods
3. **Outlier Treatment**: Applies winsorization to cap extreme values instead of removing them
4. **Standardization**: Applies z-score normalization to all features

In our dataset, we found:
- 5.14% missing values in each feature column (imputed with medians)
- Skewed distributions in Manbox5_overall (positive) and pwi_wellbeing (negative)
- A small number of outliers (0.57%-1.71%) that were successfully handled

## Clustering Analysis

K-means clustering identifies groups of students with similar patterns:
1. The optimal number of clusters is determined using silhouette scores
2. Students are assigned to clusters based on their standardized feature values
3. Cluster profiles reveal distinct student types

In our analysis, we found 3 primary clusters:
- **Cluster 1 (High Wellbeing)**: High school support, low manbox score, low masculinity constraints
- **Cluster 0 (Moderate Wellbeing)**: Moderate school support, high manbox score, high masculinity constraints
- **Cluster 2 (Low Wellbeing)**: Low school support, moderate manbox score, moderate masculinity constraints

## PCA for Wellbeing Score

After clustering, a synthetic social wellbeing score is created:
1. Principal Component Analysis is applied to the standardized feature data
2. The first principal component (PC1) explains 45% of total variance
3. If PC1 is negatively correlated with wellbeing, its sign is reversed
4. The score is normalized to a 1-100 scale
5. This synthetic score becomes the target variable for the RGCN model

Average wellbeing scores by cluster in our analysis:
- Cluster 1: 75.36 (high wellbeing)
- Cluster 0: 45.45 (moderate wellbeing)
- Cluster 2: 41.04 (low wellbeing)

## Social Network Analysis

The model incorporates social network data:
- A friendship network (1169 connections in our dataset)
- A disrespect network (76 connections in our dataset)

These networks are converted to graph structures for the RGCN model.

## Usage

Run the model with:

```bash
python run_social_wellbeing_model.py
```

This will execute the workflow in sequence:
1. Load, clean and preprocess the survey data
2. Perform K-means clustering analysis
3. Create a synthetic social wellbeing score using PCA
4. Prepare graph data for the neural network
5. Train an RGCN model using both survey features and network connections
6. Evaluate the model and save predictions

## Output

The model produces:
- Trained model in `output/social_wellbeing_rgcn_model.pt`
- Predictions in `output/social_wellbeing_predictions.csv`

## Predicting for New Students

The model includes functionality for predicting wellbeing scores for new students based on their survey responses and social connections. This can be done by:

1. Providing scaled survey response features
2. Optionally providing friendship/disrespect connections to existing students
3. Using the `predict_new_student()` function

## Model Performance

Our model demonstrates high predictive accuracy:
- R² score: 0.88 (88% of variance explained)
- Mean Absolute Error: 3.37
- Mean Squared Error: 49.03

This strong performance indicates that the combination of survey features and social network analysis effectively predicts student wellbeing. 