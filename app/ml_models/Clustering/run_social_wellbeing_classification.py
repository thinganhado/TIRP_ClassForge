import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
import os
import argparse
import logging
import warnings
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_wellbeing')

# Suppress warnings
warnings.filterwarnings('ignore')

# RGCN Classification Model
class HeteroRGCNClassifier(torch.nn.Module):
    """Heterogeneous Relational Graph Convolutional Network for social wellbeing classification"""
    def __init__(self, node_types, edge_types, num_classes, hidden_channels=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Store node and edge types
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Create embeddings for each node type
        self.embeddings = torch.nn.ModuleDict()
        for node_type in node_types:
            num_features = 3  # Use 3 PCA components as features
            self.embeddings[node_type] = torch.nn.Linear(num_features, hidden_channels)
        
        # Create convolutional layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        conv1 = torch.nn.ModuleDict()
        for edge_type in edge_types:
            conv1[f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"] = RGCNConv(
                hidden_channels, hidden_channels, len(edge_types)
            )
        self.convs.append(conv1)
        
        # Additional layers
        for _ in range(num_layers - 1):
            conv = torch.nn.ModuleDict()
            for edge_type in edge_types:
                conv[f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"] = RGCNConv(
                    hidden_channels, hidden_channels, len(edge_types)
                )
            self.convs.append(conv)
        
        # Output layers for each node type
        self.output = torch.nn.ModuleDict()
        for node_type in node_types:
            self.output[node_type] = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x_dict, edge_index_dict):
        # Transform features for each node type
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.relu(self.embeddings[node_type](x))
        
        # Apply convolution layers
        for i, conv_dict in enumerate(self.convs):
            h_dict_new = {}
            
            # For each node type, aggregate messages from all connected edge types
            for node_type in self.node_types:
                h_dict_new[node_type] = 0
                valid_edge_count = 0
                
                # Aggregate from all edge types where this node type is the target
                for edge_type in self.edge_types:
                    if edge_type[2] == node_type:  # If this node type is the target
                        src_type, edge_name, dst_type = edge_type
                        edge_key = f"{src_type}_{edge_name}_{dst_type}"
                        
                        # Check if this edge type exists in the data
                        if edge_type in edge_index_dict:
                            edge_index = edge_index_dict[edge_type]
                            conv = conv_dict[edge_key]
                            
                            # Apply convolution (using type 0 for all edges of this type)
                            msg = conv(h_dict[src_type], edge_index, edge_type=torch.zeros(edge_index.size(1), 
                                                                              dtype=torch.long,
                                                                              device=edge_index.device))
                            h_dict_new[node_type] += msg
                            valid_edge_count += 1
                
                # Average the aggregated messages 
                if valid_edge_count > 0:
                    h_dict_new[node_type] = h_dict_new[node_type] / valid_edge_count
                else:
                    h_dict_new[node_type] = h_dict[node_type]
                
                # Apply non-linearity and dropout (except for last layer)
                if i < len(self.convs) - 1:
                    h_dict_new[node_type] = F.relu(h_dict_new[node_type])
                    h_dict_new[node_type] = F.dropout(h_dict_new[node_type], p=self.dropout, training=self.training)
            
            # Update node representations
            h_dict = h_dict_new
        
        # Apply output transformation and return logits
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.output[node_type](h)
        
        return out_dict

def load_and_preprocess_data(excel_file):
    """Load and preprocess survey data"""
    logger.info("Loading and preprocessing data")
    
    try:
        # Load data from Excel sheets
        responses_df = pd.read_excel(excel_file, sheet_name='responses')
        participants_df = pd.read_excel(excel_file, sheet_name='participants')
        
        # Merge participant information with responses
        merged_df = pd.merge(responses_df, participants_df, on='Participant-ID')
        logger.info(f"Initial data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Focus on the specified survey columns
        feature_cols = ['School_support_engage6', 'Manbox5_overall', 'Masculinity_contrained', 'GrowthMindset']
        
        # Check if all required columns exist
        for col in feature_cols:
            if col not in merged_df.columns:
                logger.warning(f"Column '{col}' not found in dataset!")
                if col == 'School_support_engage6':
                    # Try alternative column names
                    alt_cols = [c for c in merged_df.columns if 'school' in c.lower() and 'support' in c.lower()]
                    if alt_cols:
                        logger.info(f"Using alternative column: {alt_cols[0]}")
                        merged_df['School_support_engage6'] = merged_df[alt_cols[0]]
                if col == 'Manbox5_overall':
                    alt_cols = [c for c in merged_df.columns if 'manbox' in c.lower()]
                    if alt_cols:
                        logger.info(f"Using alternative column: {alt_cols[0]}")
                        merged_df['Manbox5_overall'] = merged_df[alt_cols[0]]
                if col == 'GrowthMindset':
                    alt_cols = [c for c in merged_df.columns if 'growth' in c.lower() and 'mind' in c.lower()]
                    if alt_cols:
                        logger.info(f"Using alternative column: {alt_cols[0]}")
                        merged_df['GrowthMindset'] = merged_df[alt_cols[0]]
        
        # Check available columns after potential renaming
        available_cols = [col for col in feature_cols if col in merged_df.columns]
        logger.info(f"Using {len(available_cols)} columns for analysis: {', '.join(available_cols)}")
        
        # Handle missing values
        df_imputed = merged_df.copy()
        for col in available_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
                logger.info(f"Filled {missing_count} missing values in {col} with median ({median_val})")
        
        # Scale the features
        X = df_imputed[available_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a DataFrame with scaled values
        scaled_df = pd.DataFrame(X_scaled, columns=available_cols)
        scaled_df['Participant-ID'] = df_imputed['Participant-ID'].values
        
        return df_imputed, scaled_df, available_cols
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_network_data(excel_file):
    """Load student relationship networks"""
    logger.info("Loading network data")
    
    try:
        # Check if the Excel file exists
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names
        
        has_friendship = 'net_0_Friends' in sheets
        has_disrespect = 'net_5_Disrespect' in sheets
        
        # Load participant data
        participants_df = pd.read_excel(excel_file, sheet_name='participants')
        student_ids = participants_df['Participant-ID'].unique()
        id_to_idx = {id_: idx for idx, id_ in enumerate(student_ids)}
        
        # Initialize empty edge lists
        friend_edges = []
        disrespect_edges = []
        
        # Load friendship network if available
        if has_friendship:
            friends_df = pd.read_excel(excel_file, sheet_name='net_0_Friends')
            logger.info(f"Loaded friendship network: {friends_df.shape[0]} connections")
            
            for _, row in friends_df.iterrows():
                source_id = row['Source']
                target_id = row['Target']
                
                if source_id in id_to_idx and target_id in id_to_idx:
                    source_idx = id_to_idx[source_id]
                    target_idx = id_to_idx[target_id]
                    friend_edges.append([source_idx, target_idx])
        else:
            logger.warning("Friendship network sheet not found")
            
        # Load disrespect network if available
        if has_disrespect:
            disrespect_df = pd.read_excel(excel_file, sheet_name='net_5_Disrespect')
            logger.info(f"Loaded disrespect network: {disrespect_df.shape[0]} connections")
            
            for _, row in disrespect_df.iterrows():
                source_id = row['Source']
                target_id = row['Target']
                
                if source_id in id_to_idx and target_id in id_to_idx:
                    source_idx = id_to_idx[source_id]
                    target_idx = id_to_idx[target_id]
                    disrespect_edges.append([source_idx, target_idx])
        else:
            logger.warning("Disrespect network sheet not found")
        
        logger.info(f"Processed {len(friend_edges)} friendship connections and {len(disrespect_edges)} disrespect connections")
        
        return friend_edges, disrespect_edges, id_to_idx
        
    except Exception as e:
        logger.error(f"Error loading network data: {str(e)}")
        logger.info("Using empty network connections for classification")
        return [], [], {}

def apply_pca(scaled_df, feature_cols, n_components=3, output_dir="output"):
    """Apply PCA to reduce feature dimensions"""
    logger.info(f"Applying PCA with {n_components} components")
    
    # Extract scaled features
    X = scaled_df[feature_cols].values
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame with PCA components
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    pca_df['Participant-ID'] = scaled_df['Participant-ID'].values
    
    # Calculate and log variance explained
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    logger.info(f"Variance explained by {n_components} components: {sum(explained_variance)*100:.2f}%")
    for i, var in enumerate(explained_variance):
        logger.info(f"PC{i+1}: {var*100:.2f}%")
    
    # Calculate PCA loadings (component contributions)
    pca_loadings = pd.DataFrame(
        pca.components_.T, 
        columns=pca_cols, 
        index=feature_cols
    )
    
    # Save PCA loadings to CSV
    pca_loadings.to_csv(f"{output_dir}/pca_loadings.csv")
    
    return pca_df, pca, pca_cols, pca_loadings

def find_optimal_k(pca_df, pca_cols, k_range=(3, 7), output_dir="output"):
    """Find optimal number of clusters using silhouette score"""
    logger.info(f"Finding optimal k from {k_range[0]} to {k_range[1]}")
    
    # Extract PCA components
    X = pca_df[pca_cols].values
    
    # Test different k values
    k_values = range(k_range[0], k_range[1] + 1)
    silhouette_scores = []
    inertia_values = []
    
    # Find best k based on silhouette score
    best_score = -1
    best_k = 3
    
    # Store results for each k value
    k_results = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
        
        logger.info(f"k={k}: Silhouette Score={score:.4f}, Inertia={inertia_values[-1]:.4f}")
        
        # Store result
        k_results.append({
            "k": k,
            "silhouette_score": score,
            "inertia": kmeans.inertia_
        })
        
        # Update best k
        if score > best_score:
            best_score = score
            best_k = k
    
    logger.info(f"Optimal k based on silhouette score: {best_k}")
    
    # Save k evaluation results to JSON
    with open(f"{output_dir}/k_evaluation_results.json", "w") as f:
        json.dump(k_results, f, indent=4)
    
    return best_k

def perform_clustering(pca_df, pca_cols, optimal_k, output_dir="output"):
    """Perform clustering with optimal k"""
    logger.info(f"Performing clustering with k={optimal_k}")
    
    # Extract PCA components
    X = pca_df[pca_cols].values
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to DataFrame
    pca_df['cluster_label'] = cluster_labels
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Save cluster centers
    center_df = pd.DataFrame(cluster_centers, columns=pca_cols)
    center_df.index.name = 'cluster'
    center_df.to_csv(f"{output_dir}/cluster_centers.csv")
    
    return pca_df, kmeans

def analyze_clusters(pca_df, original_df, feature_cols, optimal_k, output_dir="output"):
    """Analyze cluster characteristics and assign meaningful wellbeing labels"""
    logger.info("Analyzing clusters and assigning wellbeing labels")
    
    # Add cluster labels to original DataFrame
    original_df = original_df.copy()
    original_df['cluster_label'] = pca_df['cluster_label'].values
    
    # Calculate cluster profiles
    cluster_profiles = original_df.groupby('cluster_label')[feature_cols].mean()
    
    # Calculate cluster sizes
    cluster_sizes = original_df['cluster_label'].value_counts().sort_index()
    for i, size in cluster_sizes.items():
        logger.info(f"Cluster {i}: {size} samples ({100 * size / len(original_df):.1f}%)")
    
    # Assign meaningful wellbeing labels based on cluster profiles
    wellbeing_labels = {}
    wellbeing_scores = {}
    
    for idx, row in cluster_profiles.iterrows():
        # Calculate wellbeing score based on feature values
        # Higher scores for School_support and GrowthMindset are positive
        # Higher scores for Manbox and Masculinity_constrained are negative
        wellbeing_score = row['School_support_engage6'] + row['GrowthMindset'] - row['Manbox5_overall'] - row['Masculinity_contrained']
        wellbeing_scores[idx] = wellbeing_score
    
    # Sort clusters by wellbeing score
    sorted_clusters = sorted(wellbeing_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Assign labels based on ranking
    for i, (cluster_id, _) in enumerate(sorted_clusters):
        if i == 0:
            wellbeing_labels[cluster_id] = 'High Wellbeing'
        elif i == len(sorted_clusters) - 1:
            wellbeing_labels[cluster_id] = 'Low Wellbeing'
        else:
            wellbeing_labels[cluster_id] = 'Moderate Wellbeing'
    
    # Log wellbeing labels
    for cluster_id, label in wellbeing_labels.items():
        feature_means = ', '.join([f"{col}: {cluster_profiles.loc[cluster_id, col]:.2f}" for col in feature_cols])
        logger.info(f"Cluster {cluster_id} ({label}): {feature_means}")
    
    # Add wellbeing labels to original DataFrame
    original_df['wellbeing_label'] = original_df['cluster_label'].map(wellbeing_labels)
    
    # Save cluster profiles with wellbeing labels
    cluster_profiles_with_labels = cluster_profiles.copy()
    cluster_profiles_with_labels['wellbeing_label'] = cluster_profiles_with_labels.index.map(wellbeing_labels)
    cluster_profiles_with_labels.to_csv(f"{output_dir}/cluster_profiles.csv")
    
    # Save wellbeing labels
    with open(f"{output_dir}/wellbeing_labels.json", "w") as f:
        json.dump(wellbeing_labels, f, indent=4)
    
    # Save student classifications with wellbeing labels
    original_df[['Participant-ID', 'cluster_label', 'wellbeing_label']].to_csv(
        f"{output_dir}/student_wellbeing_classifications.csv", index=False)
    
    # MODIFY: Save cluster assignments with wellbeing labels AND all feature columns
    cluster_assignments_df = pca_df.copy()
    cluster_assignments_df['wellbeing_label'] = cluster_assignments_df['cluster_label'].map(wellbeing_labels)
    
    # Merge with original features
    features_df = original_df[['Participant-ID'] + feature_cols]
    cluster_assignments_df = pd.merge(
        cluster_assignments_df[['Participant-ID', 'cluster_label', 'wellbeing_label']],
        features_df,
        on='Participant-ID'
    )
    
    # Save to CSV
    cluster_assignments_df.to_csv(f"{output_dir}/cluster_assignments.csv", index=False)
    
    return original_df, cluster_profiles, wellbeing_labels

def prepare_graph_data(pca_df, friend_edges, disrespect_edges, pca_cols):
    """Prepare data for graph-based classification"""
    logger.info("Preparing graph data for RGCN classification")
    
    # Create graph data object
    data = HeteroData()
    
    # Get features (X) and cluster labels (y)
    X = pca_df[pca_cols].values
    y = pca_df['cluster_label'].values
    
    # Convert to PyTorch tensors
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    # Add node information
    data['student'].x = x
    data['student'].y = y
    
    # Process friendship edges
    if friend_edges:
        friend_edges_tensor = torch.tensor(friend_edges, dtype=torch.long).t()
        logger.info(f"Added {friend_edges_tensor.size(1)} friendship edges to the graph")
    else:
        friend_edges_tensor = torch.zeros((2, 0), dtype=torch.long)
        logger.info("No friendship edges available")
    
    # Process disrespect edges
    if disrespect_edges:
        disrespect_edges_tensor = torch.tensor(disrespect_edges, dtype=torch.long).t() 
        logger.info(f"Added {disrespect_edges_tensor.size(1)} disrespect edges to the graph")
    else:
        disrespect_edges_tensor = torch.zeros((2, 0), dtype=torch.long)
        logger.info("No disrespect edges available")
    
    # Add edge information
    data['student', 'friend', 'student'].edge_index = friend_edges_tensor
    data['student', 'disrespect', 'student'].edge_index = disrespect_edges_tensor
    
    # Log graph structure
    logger.info(f"Graph structure: {len(data.node_types)} node types, {len(data.edge_types)} edge types")
    logger.info(f"Number of students: {data['student'].num_nodes}")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        logger.info(f"Edge type '{src} -> {rel} -> {dst}': {data[edge_type].edge_index.size(1)} edges")
    
    # Log class distribution
    class_counts = torch.bincount(y)
    for i, count in enumerate(class_counts):
        logger.info(f"Class {i}: {count.item()} samples")
    
    return data

def train_classification_model(data, num_classes, epochs=100, lr=0.01, weight_decay=5e-4, output_dir="output"):
    """Train RGCN classification model"""
    logger.info("Training RGCN classification model")
    
    # Set training flag for all nodes
    data['student'].train_mask = torch.ones(data['student'].num_nodes, dtype=torch.bool)
    
    # Get node and edge types
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    
    # Log model configuration details
    logger.info(f"RGCN model configuration:")
    logger.info(f"  - Node types: {node_types}")
    logger.info(f"  - Edge types: {edge_types}")
    logger.info(f"  - Number of classes: {num_classes}")
    logger.info(f"  - Hidden channels: 128")
    logger.info(f"  - Number of layers: 2")
    logger.info(f"  - Dropout rate: 0.3")
    logger.info(f"  - Learning rate: {lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Max epochs: {epochs}")
    
    # Create model
    model = HeteroRGCNClassifier(
        node_types=node_types,
        edge_types=edge_types,
        num_classes=num_classes,
        hidden_channels=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Log model structure
    logger.info(f"Model structure:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - Total parameters: {total_params}")
    logger.info(f"  - Trainable parameters: {trainable_params}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    # Training loop
    best_loss = float('inf')
    patience = 20
    counter = 0
    best_model_state = None
    
    # Track losses for analysis
    losses = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
        # Calculate loss (cross entropy for classification)
        loss = F.cross_entropy(out['student'], data['student'].y)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")
        
        # Update learning rate scheduler
        scheduler.step(loss)
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Log final training statistics
    logger.info(f"Training completed:")
    logger.info(f"  - Best loss: {best_loss:.6f}")
    logger.info(f"  - Initial loss: {losses[0]:.6f}")
    logger.info(f"  - Final loss: {losses[-1]:.6f}")
    logger.info(f"  - Loss improvement: {(1 - losses[-1]/losses[0])*100:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate model on all data
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        predictions = torch.argmax(out['student'], dim=1)
        accuracy = (predictions == data['student'].y).float().mean()
        logger.info(f"Model accuracy on all data: {accuracy.item():.4f}")
        
        # Confusion matrix as counts
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(data['student'].y, predictions):
            confusion[t.long(), p.long()] += 1
            
        logger.info("Confusion matrix:")
        confusion_str = []
        for i in range(num_classes):
            row = " ".join([f"{confusion[i, j].item():4d}" for j in range(num_classes)])
            confusion_str.append(f"  Class {i}: [{row}]")
        logger.info("\n".join(confusion_str))
    
    # Save training loss curve to JSON
    with open(f"{output_dir}/rgcn_training_losses.json", "w") as f:
        json.dump(losses, f)
    
    return model

def save_results(model, wellbeing_labels, feature_cols, pca_cols, output_dir="output"):
    """Save model and metadata"""
    logger.info(f"Saving model and metadata to {output_dir}")
    
    # Save trained model
    torch.save(model.state_dict(), f"{output_dir}/rgcn_classification_model.pt")
    
    # Save metadata
    metadata = {
        'wellbeing_labels': wellbeing_labels,
        'feature_cols': feature_cols,
        'pca_cols': pca_cols,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save as JSON file
    with open(f"{output_dir}/model_metadata.json", "w") as f:
        # Convert integer keys to strings for JSON serialization
        wellbeing_labels_str = {str(k): v for k, v in wellbeing_labels.items()}
        metadata['wellbeing_labels'] = wellbeing_labels_str
        json.dump(metadata, f, indent=4)
    
    # Save as PyTorch file
    torch.save(metadata, f"{output_dir}/model_metadata.pt")
    
    logger.info(f"Results saved to {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PCA-based Social Wellbeing Classification')
    parser.add_argument('--data', type=str, default="Student Survey - Jan.xlsx", 
                        help='Path to Excel file containing survey data')
    parser.add_argument('--output', type=str, default="output", 
                        help='Output directory for results and visualizations')
    parser.add_argument('--components', type=int, default=3, 
                        help='Number of PCA components to use')
    parser.add_argument('--min-k', type=int, default=3, 
                        help='Minimum number of clusters to consider')
    parser.add_argument('--max-k', type=int, default=7, 
                        help='Maximum number of clusters to consider')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    
    try:
        logger.info("Starting PCA-based Social Wellbeing Classification")
        
        # Step 1: Load and preprocess data
        original_df, scaled_df, feature_cols = load_and_preprocess_data(args.data)
        
        # Step 2: Load network data
        friend_edges, disrespect_edges, id_to_idx = load_network_data(args.data)
        
        # Step 3: Apply PCA
        pca_df, pca, pca_cols, pca_loadings = apply_pca(
            scaled_df, feature_cols, 
            n_components=args.components, 
            output_dir=args.output
        )
        
        # Step 4: Find optimal k
        optimal_k = find_optimal_k(
            pca_df, pca_cols, 
            k_range=(args.min_k, args.max_k), 
            output_dir=args.output
        )
        
        # Step 5: Perform clustering with optimal k
        pca_df, kmeans = perform_clustering(
            pca_df, pca_cols, optimal_k, 
            output_dir=args.output
        )
        
        # Step 6: Analyze clusters and assign wellbeing labels
        labeled_df, cluster_profiles, wellbeing_labels = analyze_clusters(
            pca_df, original_df, feature_cols, optimal_k,
            output_dir=args.output
        )
        
        # Step 7: Prepare graph data for RGCN classification
        graph_data = prepare_graph_data(pca_df, friend_edges, disrespect_edges, pca_cols)
        
        # Step 8: Train RGCN classification model
        model = train_classification_model(
            graph_data, num_classes=optimal_k,
            epochs=args.epochs,
            output_dir=args.output
        )
        
        # Step 9: Save model and metadata
        save_results(
            model, wellbeing_labels, feature_cols, pca_cols,
            output_dir=args.output
        )
        
        logger.info(f"PCA-based Social Wellbeing Classification completed successfully")
        logger.info(f"Optimal number of clusters: {optimal_k}")
        logger.info(f"Results saved to {args.output}/")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 