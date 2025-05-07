# Social Wellbeing Classification Model

A streamlined machine learning model that classifies students into wellbeing categories based on survey responses and social network data.

## Overview

This model classifies students into wellbeing categories through:
1. Automatic cluster analysis of student survey data
2. Meaningful labeling of clusters as "High Wellbeing" or "Low Wellbeing"
3. Graph Neural Network classification leveraging both student attributes and social connections

## Key Features

- Automatic data processing and cleaning
- Feature-based wellbeing category assignment
- Graph-based classification incorporating peer relationships
- Simple output focused on classification results

## Inputs

- Student survey data with the following key measures:
  - School support engagement
  - Gender norm adherence measures
  - Growth mindset indicators
- Social network connections between students (friendships and negative interactions)

## Outputs

The model produces:
- `wellbeing_classification_results.csv`: Complete classification results with wellbeing labels and probabilities
- `social_wellbeing_rgcn_classifier.pt`: Trained classification model for predicting new student categories

## Usage

1. Prepare your student data in Excel format with the required sheets:
   - 'responses': Survey responses
   - 'participants': Student information
   - 'net_0_Friends': Friendship connections
   - 'net_5_Disrespect': Negative interactions (optional)

2. Run the classification without verbose processing details:
   ```
   python run_social_wellbeing_classification.py
   ```

## Output Format

The classification results file contains:
- Student identifiers
- Wellbeing category labels
- Classification probabilities
- Feature values

## Wellbeing Categories

The model automatically identifies and labels student wellbeing categories:

**High Wellbeing**: Students with higher school support, lower gender norm adherence, and higher growth mindset.

**Low Wellbeing**: Students with lower school support, higher gender norm adherence, and lower growth mindset.

## Technical Details

Built with:
- PyTorch and PyTorch Geometric for Graph Neural Networks
- scikit-learn for clustering and preprocessing
- pandas for data manipulation

## Requirements

```
pip install torch pandas numpy scikit-learn torch-geometric
``` 