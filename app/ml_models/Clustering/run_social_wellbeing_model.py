import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.transforms import RandomNodeSplit
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs("output", exist_ok=True)

class HeteroRGCN(torch.nn.Module):
    """Heterogeneous Relational Graph Convolutional Network for social wellbeing prediction"""
    def __init__(self, node_types, edge_types, hidden_channels=128, out_channels=1, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Store node and edge types
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        
        # Create embeddings for each node type
        self.embeddings = torch.nn.ModuleDict()
        for node_type in node_types:
            num_features = 4  # Default for student nodes with our features
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
            self.output[node_type] = torch.nn.Linear(hidden_channels, out_channels)
    
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
        
        # Apply output transformation
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.output[node_type](h)
        
        return out_dict

def check_data_distribution(df, feature_cols):
    """Analyze data distribution for skewness and outliers"""
    print("\nAnalyzing data distribution:")
    
    # Calculate summary statistics
    summary = df[feature_cols].describe()
    print("\nSummary statistics:")
    print(summary)
    
    # Check for skewness
    skewness = df[feature_cols].skew()
    print("\nSkewness analysis:")
    for col, skew_val in skewness.items():
        print(f"  {col}: Skewness = {skew_val:.4f} ", end="")
        if abs(skew_val) < 0.5:
            print("(approximately symmetric)")
        elif abs(skew_val) < 1.0:
            print("(moderately skewed)")
        else:
            print("(highly skewed)")
    
    # Check for outliers using IQR method
    print("\nOutlier analysis (using IQR method):")
    outliers_summary = {}
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_count = len(outliers)
        outliers_pct = 100 * outliers_count / len(df)
        
        print(f"  {col}: {outliers_count} outliers ({outliers_pct:.2f}%)")
        outliers_summary[col] = {
            'count': outliers_count,
            'percentage': outliers_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return summary, skewness, outliers_summary

def handle_missing_values(df, feature_cols, method='median'):
    """Handle missing values in the dataset"""
    print("\nHandling missing values:")
    
    # Count missing values
    missing = df[feature_cols].isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    print("Missing values before imputation:")
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} missing values ({missing_pct[col]:.2f}%)")
        else:
            print(f"  {col}: No missing values")
    
    # Make a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Apply imputation method
    if method == 'median':
        for col in feature_cols:
            if missing[col] > 0:
                median_val = df[col].median()
                df_imputed[col] = df[col].fillna(median_val)
                print(f"  Imputed {missing[col]} values in {col} with median: {median_val:.4f}")
    
    elif method == 'mean':
        for col in feature_cols:
            if missing[col] > 0:
                mean_val = df[col].mean()
                df_imputed[col] = df[col].fillna(mean_val)
                print(f"  Imputed {missing[col]} values in {col} with mean: {mean_val:.4f}")
    
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df_imputed[feature_cols] = pd.DataFrame(
            imputer.fit_transform(df[feature_cols]), 
            columns=feature_cols
        )
        print("  Used KNN imputation for missing values")
    
    # Check if any missing values remain
    still_missing = df_imputed[feature_cols].isnull().sum().sum()
    if still_missing > 0:
        print(f"Warning: {still_missing} missing values remain after imputation")
    else:
        print("  All missing values have been imputed")
    
    return df_imputed

def handle_outliers(df, feature_cols, outliers_summary, method='winsorize'):
    """Handle outliers in the dataset"""
    print("\nHandling outliers:")
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    if method == 'winsorize':
        for col in feature_cols:
            if outliers_summary[col]['count'] > 0:
                lower = outliers_summary[col]['lower_bound']
                upper = outliers_summary[col]['upper_bound']
                
                # Replace outliers with bounds
                df_cleaned[col] = np.where(
                    df[col] < lower,
                    lower,
                    np.where(df[col] > upper, upper, df[col])
                )
                
                print(f"  {col}: Winsorized {outliers_summary[col]['count']} outliers " +
                      f"({outliers_summary[col]['percentage']:.2f}%)")
    
    elif method == 'trim':
        # Create a mask for non-outlier rows
        mask = pd.Series(True, index=df.index)
        
        for col in feature_cols:
            if outliers_summary[col]['count'] > 0:
                lower = outliers_summary[col]['lower_bound']
                upper = outliers_summary[col]['upper_bound']
                
                # Update mask to exclude outliers
                col_mask = (df[col] >= lower) & (df[col] <= upper)
                mask = mask & col_mask
                
        # Apply mask to get trimmed dataframe
        df_cleaned = df[mask].copy()
        removed = len(df) - len(df_cleaned)
        print(f"  Removed {removed} rows ({100*removed/len(df):.2f}%) containing outliers")
    
    return df_cleaned

def load_and_preprocess_data(excel_file):
    """Load and preprocess survey data and relationship data"""
    print("Step 1: Loading and preprocessing data...")
    
    try:
        # Load data from Excel sheets
        responses_df = pd.read_excel(excel_file, sheet_name='responses')
        participants_df = pd.read_excel(excel_file, sheet_name='participants')
        
        # Load relationship data
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names
        
        has_friendship = 'net_0_Friends' in sheets
        has_disrespect = 'net_5_Disrespect' in sheets
        
        if has_friendship:
            friends_df = pd.read_excel(excel_file, sheet_name='net_0_Friends')
            print(f"Loaded friendship network: {friends_df.shape[0]} connections")
        else:
            print("Warning: Friendship network sheet not found")
            friends_df = pd.DataFrame(columns=['Source', 'Target'])
            
        if has_disrespect:
            disrespect_df = pd.read_excel(excel_file, sheet_name='net_5_Disrespect')
            print(f"Loaded disrespect network: {disrespect_df.shape[0]} connections")
        else:
            print("Warning: Disrespect network sheet not found")
            disrespect_df = pd.DataFrame(columns=['Source', 'Target'])
        
        # Merge participant information with responses
        merged_df = pd.merge(responses_df, participants_df, on='Participant-ID')
        print(f"Initial data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Focus on the specified survey columns
        feature_cols = ['School_support_engage6', 'Manbox5_overall', 'Masculinity_contrained', 'pwi_wellbeing']
        
        # Check if all required columns exist
        for col in feature_cols:
            if col not in merged_df.columns:
                print(f"Warning: Column '{col}' not found in dataset!")
                if col == 'School_support_engage6':
                    # Try alternative column names
                    alt_cols = [c for c in merged_df.columns if 'school' in c.lower() and 'support' in c.lower()]
                    if alt_cols:
                        print(f"Using alternative column: {alt_cols[0]}")
                        merged_df['School_support_engage6'] = merged_df[alt_cols[0]]
                if col == 'Manbox5_overall':
                    alt_cols = [c for c in merged_df.columns if 'manbox' in c.lower()]
                    if alt_cols:
                        print(f"Using alternative column: {alt_cols[0]}")
                        merged_df['Manbox5_overall'] = merged_df[alt_cols[0]]
        
        # Check available columns after potential renaming
        available_cols = [col for col in feature_cols if col in merged_df.columns]
        print(f"\nUsing {len(available_cols)} columns for analysis:")
        for col in available_cols:
            print(f"  - {col}")
        
        # Analyze data distribution before cleaning
        _, skewness, outliers_summary = check_data_distribution(merged_df, available_cols)
        
        # Handle missing values
        print("\nStep 1a: Handling missing values...")
        df_imputed = handle_missing_values(merged_df, available_cols, method='median')
        
        # Handle outliers
        print("\nStep 1b: Handling outliers...")
        df_cleaned = handle_outliers(df_imputed, available_cols, outliers_summary, method='winsorize')
        
        # Extract features for clustering and PCA
        X = df_cleaned[available_cols].values
        
        # Save ID mapping for later use
        student_ids = df_cleaned['Participant-ID'].unique()
        id_to_idx = {id_: idx for idx, id_ in enumerate(student_ids)}
        
        # Create edge lists
        friend_edges = []
        if has_friendship:
            for _, row in friends_df.iterrows():
                source_id = row['Source']
                target_id = row['Target']
                
                if source_id in id_to_idx and target_id in id_to_idx:
                    source_idx = id_to_idx[source_id]
                    target_idx = id_to_idx[target_id]
                    friend_edges.append([source_idx, target_idx])
        
        disrespect_edges = []
        if has_disrespect:
            for _, row in disrespect_df.iterrows():
                source_id = row['Source']
                target_id = row['Target']
                
                if source_id in id_to_idx and target_id in id_to_idx:
                    source_idx = id_to_idx[source_id]
                    target_idx = id_to_idx[target_id]
                    disrespect_edges.append([source_idx, target_idx])
        
        print("\nProcessed network connections:")
        print(f"  - {len(friend_edges)} friendship connections")
        print(f"  - {len(disrespect_edges)} disrespect connections")
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a DataFrame with scaled values
        scaled_df = pd.DataFrame(X_scaled, columns=available_cols)
        scaled_df['Participant-ID'] = df_cleaned['Participant-ID'].values
        
        # Add back the scaled features to the original DataFrame
        for i, col in enumerate(available_cols):
            df_cleaned[f"{col}_scaled"] = X_scaled[:, i]
        
        return df_cleaned, scaled_df, available_cols, friend_edges, disrespect_edges, id_to_idx
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def perform_clustering(scaled_df, feature_cols, max_clusters=10):
    """Apply K-Means clustering to the scaled survey data"""
    print("\nStep 2: Performing K-Means clustering...")
    
    # Extract scaled features for clustering
    X = scaled_df[feature_cols].values
    
    # Determine optimal k using silhouette score
    silhouette_scores = []
    for k in range(2, min(max_clusters + 1, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Skip if only one cluster is found
        if len(np.unique(cluster_labels)) < 2:
            silhouette_scores.append(-1)
            continue
            
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"  K={k}: Silhouette Score = {score:.4f}")
    
    # Find optimal k 
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Apply K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to the DataFrame
    scaled_df['cluster'] = cluster_labels
    
    # Create and save cluster profiles
    cluster_profiles = scaled_df.groupby('cluster')[feature_cols].mean()
    print("Cluster profiles (mean values of features for each cluster):")
    print(cluster_profiles)
    
    return scaled_df, optimal_k, cluster_profiles

def apply_pca_for_wellbeing_score(scaled_df, feature_cols):
    """Apply PCA to create a synthetic social wellbeing score"""
    print("\nStep 3: Applying PCA to create synthetic social wellbeing score...")
    
    # Extract scaled features for PCA
    X = scaled_df[feature_cols].values
    
    # Apply PCA
    pca = PCA(n_components=len(feature_cols))
    principal_components = pca.fit_transform(X)
    
    # Extract first principal component
    pc1 = principal_components[:, 0]
    
    # Check variance explained
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by each component: {explained_variance}")
    print(f"Variance explained by first component: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
    
    # Check correlation with pwi_wellbeing if available
    if 'pwi_wellbeing' in scaled_df.columns:
        corr = np.corrcoef(pc1, scaled_df['pwi_wellbeing'])[0, 1]
        print(f"Correlation between PC1 and pwi_wellbeing: {corr:.4f}")
        
        # If negatively correlated, reverse the PC1 scores
        if corr < 0:
            pc1 = -pc1
            print("PC1 was negatively correlated with wellbeing. Scale reversed.")
    
    # Normalize to 1-100 scale
    social_wellbeing = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min())
    
    # Add to DataFrame
    scaled_df['social_wellbeing'] = social_wellbeing
    
    print(f"Social wellbeing score range: {social_wellbeing.min():.2f} to {social_wellbeing.max():.2f}")
    
    # Calculate average wellbeing score by cluster
    cluster_wellbeing = scaled_df.groupby('cluster')['social_wellbeing'].mean()
    print("Average wellbeing score by cluster:")
    print(cluster_wellbeing)
    
    return scaled_df, social_wellbeing, pca

def prepare_graph_data(scaled_df, friend_edges, disrespect_edges, feature_cols, social_wellbeing):
    """Prepare graph data for GNN"""
    print("\nStep 4: Preparing graph data for GNN...")
    
    # Create graph data object
    data = HeteroData()
    
    # Add student nodes and their features
    X = scaled_df[feature_cols].values
    y = social_wellbeing.reshape(-1, 1)  # target: social wellbeing score
    
    # Convert to PyTorch tensors
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    
    # Add node information
    data['student'].x = x
    data['student'].y = y
    
    # Process friendship edges
    if friend_edges:
        friend_edges_tensor = torch.tensor(friend_edges, dtype=torch.long).t()
    else:
        friend_edges_tensor = torch.zeros((2, 0), dtype=torch.long)
    
    # Process disrespect edges
    if disrespect_edges:
        disrespect_edges_tensor = torch.tensor(disrespect_edges, dtype=torch.long).t() 
    else:
        disrespect_edges_tensor = torch.zeros((2, 0), dtype=torch.long)
    
    # Add edge information
    data['student', 'friend', 'student'].edge_index = friend_edges_tensor
    data['student', 'disrespect', 'student'].edge_index = disrespect_edges_tensor
    
    # Split the data
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)
    data = transform(data)
    
    print(f"Graph data prepared:")
    print(f"  - {data['student'].x.size(0)} student nodes")
    print(f"  - {data['student', 'friend', 'student'].edge_index.size(1)} friend edges")
    print(f"  - {data['student', 'disrespect', 'student'].edge_index.size(1)} disrespect edges")
    print(f"Data split: {data['student'].train_mask.sum().item()} train, "
          f"{data['student'].val_mask.sum().item()} validation, "
          f"{data['student'].test_mask.sum().item()} test nodes")
    
    return data

def train_rgcn_model(data, epochs=100, lr=0.01, weight_decay=5e-4):
    """Train RGCN for Regression"""
    print("\nStep 5: Training RGCN for regression...")
    
    # Get node and edge types
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    
    # Create model
    model = HeteroRGCN(
        node_types=node_types,
        edge_types=edge_types,
        hidden_channels=128,
        out_channels=1,
        num_layers=2,
        dropout=0.3
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
        # Calculate loss
        loss = F.mse_loss(out['student'][data['student'].train_mask].squeeze(), 
                          data['student'].y[data['student'].train_mask].squeeze())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Forward pass
            out = model(data.x_dict, data.edge_index_dict)
            
            val_loss = F.mse_loss(out['student'][data['student'].val_mask].squeeze(), 
                                 data['student'].y[data['student'].val_mask].squeeze())
            val_losses.append(val_loss.item())
            
            test_loss = F.mse_loss(out['student'][data['student'].test_mask].squeeze(), 
                                  data['student'].y[data['student'].test_mask].squeeze())
            test_losses.append(test_loss.item())
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses, test_losses

def evaluate_model(model, data):
    """Evaluate model and make predictions"""
    print("\nStep 6: Evaluating model and making predictions...")
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
        # Get predictions for all students
        all_preds = out['student'].squeeze().cpu().numpy()
        all_true = data['student'].y.squeeze().cpu().numpy()
        
        # Predictions for test set
        test_mask = data['student'].test_mask.cpu().numpy()
        test_preds = all_preds[test_mask]
        test_true = all_true[test_mask]
        
        # Calculate metrics
        mse = mean_squared_error(test_true, test_preds)
        mae = mean_absolute_error(test_true, test_preds)
        r2 = r2_score(test_true, test_preds)
        
        print("\nTest Set Metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return all_preds, all_true, mse, r2, mae

def predict_new_student(model, data, features, friend_connections=None, disrespect_connections=None):
    """Predict wellbeing for a new student based on features and connections"""
    print("\nPredicting social wellbeing for a new student...")
    
    # Create a copy of the data
    new_data = data.clone()
    
    # Get number of existing students
    num_students = new_data['student'].x.size(0)
    new_student_idx = num_students
    
    # Convert features to tensor and reshape
    features_tensor = torch.tensor([features], dtype=torch.float)
    
    # Add new student to the graph
    new_x = torch.cat([new_data['student'].x, features_tensor], dim=0)
    new_data['student'].x = new_x
    
    # Add dummy target
    dummy_y = torch.zeros(1, 1)
    new_y = torch.cat([new_data['student'].y, dummy_y], dim=0)
    new_data['student'].y = new_y
    
    # Update masks
    train_mask = torch.cat([new_data['student'].train_mask, torch.tensor([False])], dim=0)
    val_mask = torch.cat([new_data['student'].val_mask, torch.tensor([False])], dim=0)
    test_mask = torch.cat([new_data['student'].test_mask, torch.tensor([True])], dim=0)
    new_data['student'].train_mask = train_mask
    new_data['student'].val_mask = val_mask
    new_data['student'].test_mask = test_mask
    
    # Add friend connections if provided
    if friend_connections:
        new_friend_edges = []
        for friend_idx in friend_connections:
            # Add bidirectional connection
            new_friend_edges.append([new_student_idx, friend_idx])
            new_friend_edges.append([friend_idx, new_student_idx])
        
        new_friend_edges = torch.tensor(new_friend_edges, dtype=torch.long).t()
        
        # Add to existing edges
        if new_data['student', 'friend', 'student'].edge_index.size(1) > 0:
            new_edge_index = torch.cat(
                [new_data['student', 'friend', 'student'].edge_index, new_friend_edges], 
                dim=1
            )
        else:
            new_edge_index = new_friend_edges
            
        new_data['student', 'friend', 'student'].edge_index = new_edge_index
    
    # Add disrespect connections if provided
    if disrespect_connections:
        new_disrespect_edges = []
        for disrespect_idx in disrespect_connections:
            # Add directed connection (can be one-way)
            new_disrespect_edges.append([new_student_idx, disrespect_idx])
        
        new_disrespect_edges = torch.tensor(new_disrespect_edges, dtype=torch.long).t()
        
        # Add to existing edges
        if new_data['student', 'disrespect', 'student'].edge_index.size(1) > 0:
            new_edge_index = torch.cat(
                [new_data['student', 'disrespect', 'student'].edge_index, new_disrespect_edges], 
                dim=1
            )
        else:
            new_edge_index = new_disrespect_edges
            
        new_data['student', 'disrespect', 'student'].edge_index = new_edge_index
    
    # Predict wellbeing for the new student
    model.eval()
    with torch.no_grad():
        out = model(new_data.x_dict, new_data.edge_index_dict)
        prediction = out['student'][-1].item()  # Get prediction for the new student
    
    print(f"Predicted social wellbeing: {prediction:.2f}")
    return prediction

def main():
    try:
        # Step 1: Load and preprocess the data
        excel_file = "Student Survey - Jan.xlsx"
        filtered_df, scaled_df, feature_cols, friend_edges, disrespect_edges, id_to_idx = load_and_preprocess_data(excel_file)
        
        # Step 2: Perform clustering
        scaled_df, optimal_k, cluster_profiles = perform_clustering(scaled_df, feature_cols)
        
        # Step 3: Apply PCA to create social wellbeing score
        scaled_df, social_wellbeing, pca = apply_pca_for_wellbeing_score(scaled_df, feature_cols)
        
        # Step 4: Prepare graph data for GNN
        graph_data = prepare_graph_data(scaled_df, friend_edges, disrespect_edges, feature_cols, social_wellbeing)
        
        # Step 5: Train RGCN model
        model, train_losses, val_losses, test_losses = train_rgcn_model(graph_data)
        
        # Step 6: Evaluate model and make predictions
        all_preds, all_true, mse, r2, mae = evaluate_model(model, graph_data)
        
        # Save predictions to CSV
        scaled_df['predicted_wellbeing'] = np.nan
        idx_list = [i for i, mask in enumerate(graph_data['student'].test_mask) if mask]
        
        for idx in idx_list:
            scaled_df.loc[idx, 'predicted_wellbeing'] = all_preds[idx]
        
        scaled_df['actual_wellbeing'] = all_true
        scaled_df.to_csv('output/social_wellbeing_predictions.csv', index=False)
        
        # Example: Predict for a new student
        print("\n--- Example: Predicting for a new student ---")
        example_features = [0.5, -0.2, 1.0, 0.3]  # Scaled features
        example_friends = [0, 5, 10]  # Example friends
        
        new_prediction = predict_new_student(
            model, graph_data, example_features, friend_connections=example_friends
        )
        
        # Save the trained model
        torch.save(model.state_dict(), 'output/social_wellbeing_rgcn_model.pt')
        print("\nModel saved to output/social_wellbeing_rgcn_model.pt")
        print("\nWorkflow completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 