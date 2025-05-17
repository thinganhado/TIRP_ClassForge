# Social Wellbeing Classification Model Files

This directory contains the output files from the social wellbeing classification model. These files are used by the TIRP application to classify students into wellbeing categories and make recommendations.

## Files Required for Production/GitHub

These files must be included in your GitHub repository for the application to function correctly:

1. **cluster_assignments.csv**
   - Contains student IDs, cluster labels, wellbeing labels, and feature values
   - Critical for NLP and recommendation system
   - Used as primary data source for social wellbeing classification

2. **rgcn_classification_model.pt**
   - Trained PyTorch RGCN model for classification
   - Required for making new predictions
   - Contains model weights and architecture

3. **model_metadata.json**
   - JSON format metadata about the model
   - Contains wellbeing labels, feature columns, and timestamp
   - Used for model interpretation

4. **model_metadata.pt** 
   - PyTorch format metadata for easier loading
   - Provides same information as JSON but in PyTorch format
   - Used by prediction scripts

5. **cluster_profiles.csv**
   - Statistical profiles of each wellbeing cluster
   - Shows mean feature values for each cluster
   - Used for interpreting cluster characteristics

6. **wellbeing_labels.json**
   - Maps numeric cluster IDs to wellbeing labels (High, Moderate, Low)
   - Essential for meaningful interpretation of results

## Optional Analytics Files

These files are for analysis purposes but not required for core functionality:

1. **cluster_centers.csv**
   - Technical coordinates of cluster centers in PCA space
   - Useful for analysis but not required for prediction

2. **k_evaluation_results.json**
   - Evaluation data for determining optimal number of clusters (k)
   - Contains silhouette scores and inertia values

3. **pca_loadings.csv**
   - PCA component loadings showing feature contributions
   - Useful for understanding feature importance

4. **rgcn_training_losses.json**
   - Training loss history for the RGCN model
   - Useful for debugging model training issues

5. **student_wellbeing_classifications.csv**
   - Simplified version of cluster assignments
   - Redundant with cluster_assignments.csv (which includes more data)

## Usage Notes

When deploying to production or pushing to GitHub:

1. Include all files in the "Required for Production" section
2. The analytics files can be included or excluded based on your needs
3. If space is a concern, the rgcn_classification_model.pt file is the largest (780KB)
4. Make sure to maintain the directory structure for the application to find these files

For more information on how these files are used, refer to the `app/models/README_NLP_DATABASE.md` file. 