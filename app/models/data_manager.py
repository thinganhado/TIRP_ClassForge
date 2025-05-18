import os
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import json
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_manager')

class DataManager:
    """
    Streamlined data manager that uses pre-trained models and minimal data loading.
    
    This class optimizes app performance by:
    1. Using pre-trained models instead of runtime data processing
    2. Minimizing memory usage by avoiding large dataset storage
    3. Providing a simple interface for model predictions
    4. Only loading essential configuration data at runtime
    """
    
    def __init__(self, environment="production", config_path=None):
        """
        Initialize the data manager.
        
        Args:
            environment: "development" or "production"
            config_path: Path to configuration file
        """
        self.environment = environment
        self.config_path = config_path or "app/models/soft_constraints_config.json"
        self.config = {}
        self.data_cache = {}
        self.insights = {}
        self.models = {}  # For storing loaded ML models
        
        # Essential paths
        self.models_dir = 'app/models/trained_models'
        
        # Load configuration
        self.load_config()
        
        # Load model insights if available
        self.load_model_insights()
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load all trained models
        self._load_trained_models()
        
        logger.info(f"DataManager initialized in {environment} mode")
        
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Set default configuration
            self.config = {
                "bully_penalty_weight": 75,
                "class_size": 30,
                "friend_inclusion_weight": 75,
                "friendship_balance_weight": 50,
                "gpa_penalty_weight": 75,
                "influence_std_weight": 75,
                "isolated_std_weight": 75,
                "max_classes": 6,
                "min_friends_required": 3,
                "prioritize_academic": 5,
                "prioritize_bullying": 5,
                "prioritize_friendship": 5,
                "prioritize_social_influence": 5,
                "prioritize_wellbeing": 5,
                "wellbeing_penalty_weight": 75
            }
            logger.info("Using default configuration")
    
    def save_config(self, new_config=None):
        """Save configuration to file"""
        config_to_save = new_config if new_config is not None else self.config
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            # Update the current config
            if new_config is not None:
                self.config = new_config
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def update_config(self, updates):
        """
        Update specific configuration parameters
        
        Args:
            updates (dict): Dictionary of parameters to update
            
        Returns:
            dict: The updated configuration
        """
        # Make a copy of the current config
        new_config = self.config.copy()
        
        # Apply updates
        for key, value in updates.items():
            if key in new_config:
                new_config[key] = value
                logger.info(f"Updated parameter {key}: {self.config.get(key)} -> {value}")
            else:
                logger.warning(f"Parameter {key} not found in config, adding it")
                new_config[key] = value
        
        # Save the updated config
        self.save_config(new_config)
        
        return new_config
    
    def load_model_insights(self):
        """Load model insights from trained models for recommendations"""
        insights_path = "app/models/trained_models/model_insights.json"
        
        try:
            if os.path.exists(insights_path):
                with open(insights_path, 'r') as f:
                    self.insights = json.load(f)
                logger.info(f"Model insights loaded from {insights_path}")
                
                # Log the number of insights loaded per category
                for category, items in self.insights.items():
                    if isinstance(items, list):
                        logger.info(f"Loaded {len(items)} {category} insights")
            else:
                logger.warning(f"Model insights file not found: {insights_path}")
        except Exception as e:
            logger.error(f"Failed to load model insights: {e}")
    
    def get_recommendations(self, domain=None):
        """
        Get recommendations based on model insights tailored for the set_priorities page
        
        Args:
            domain (str, optional): Specific domain to get recommendations for
            
        Returns:
            list: List of recommendation sentences ready for display
        """
        if not self.insights:
            return ["Our AI assistant needs more data to provide personalized recommendations."]
            
        # Initialize recommendations list
        recommendations = []
        
        # Add general recommendation based on data
        if "class_allocation_metrics" in self.insights:
            metrics = self.insights["class_allocation_metrics"]
            
            # Find the lowest performing metric
            lowest_metric = min(metrics.items(), key=lambda x: x[1])
            lowest_name, lowest_value = lowest_metric
            
            if lowest_name == "friendship_coverage":
                recommendations.append(f"Friendship coverage metric is at {int(lowest_value*100)}% - consider increasing the Friend Inclusion Weight slider.")
            elif lowest_name == "social_balance":
                recommendations.append(f"Social balance metric is at {int(lowest_value*100)}% - consider increasing both Influence and Isolation Balance sliders.")
            elif lowest_name == "gpa_fairness":
                recommendations.append(f"GPA fairness metric is at {int(lowest_value*100)}% - consider increasing the GPA Balance Weight slider.")
            elif lowest_name == "wellbeing_distribution":
                recommendations.append(f"Wellbeing distribution is at {int(lowest_value*100)}% - consider increasing the Wellbeing Balance Weight slider.")
            elif lowest_name == "bullying_prevention":
                recommendations.append(f"Bullying prevention metric is at {int(lowest_value*100)}% - consider increasing Bully-Victim Separation Weight slider.")
        
        # Recommendations based on influential students
        if "top_influential_students" in self.insights and len(self.insights["top_influential_students"]) > 0:
            top_student = self.insights["top_influential_students"][0]
            score = int(top_student["influential_score"] * 100)
            recommendations.append(f"Student {top_student['Participant-ID']} has a {score}% influence score - distribute influential students evenly by increasing the Influence Balance Weight.")
        
        # Recommendations based on isolated students
        if "most_isolated_students" in self.insights and len(self.insights["most_isolated_students"]) > 0:
            isolated_student = self.insights["most_isolated_students"][0]
            score = int(isolated_student["isolated_score"] * 100)
            recommendations.append(f"Student {isolated_student['Participant-ID']} has a {score}% isolation score - support isolated students by increasing the Isolation Balance Weight.")
        
        # Recommendations based on friendship pairs
        if "strongest_friendships" in self.insights and len(self.insights["strongest_friendships"]) > 0:
            top_pair = self.insights["strongest_friendships"][0]
            score = int(top_pair["friendship_score"] * 100)
            recommendations.append(f"Students {top_pair['Participant1-ID']} and {top_pair['Participant2-ID']} have a {score}% friendship score - keep close friends together by raising Friend Inclusion Weight.")
        
        # Domain-specific recommendations for priorities section
        if domain == "academic" or domain is None:
            recommendations.append("For GPA optimization, drag Academic Performance higher in your priorities ranking and increase its weight value to 5.")
            
        if domain == "wellbeing" or domain is None:
            recommendations.append("To prioritize student wellness, rank Student Wellbeing higher and ensure its weight value is at least 4.")
            
        if domain == "bullying" or domain is None:
            recommendations.append("For effective bullying prevention, set Bully-Victim Separation Weight above 75 and rank Bullying Prevention among your top 3 priorities.")
            
        if domain == "social" or domain is None:
            recommendations.append("To balance social dynamics, set Influence Balance Weight to at least 70 and ensure Social Influence Balance is ranked appropriately.")
            
        if domain == "friendship" or domain is None:
            recommendations.append("To improve friendship connections, set Minimum In-Class Friends to at least 2 and Friend Inclusion Weight above 60.")
        
        # Add an advanced insight that connects multiple factors
        recommendations.append("Based on our model analysis, balancing social influence and friendship factors together yields optimal classroom dynamics and student satisfaction.")
            
        return recommendations

    def get_tfidf_vocabulary(self):
        """Load TF-IDF vocabulary from file"""
        vocab_path = "app/models/tfidf_vectorizer/vocabulary.json"
        
        if "tfidf_vocabulary" in self.data_cache:
            return self.data_cache["tfidf_vocabulary"]
            
        try:
            with open(vocab_path, 'r') as f:
                vocabulary = json.load(f)
            
            self.data_cache["tfidf_vocabulary"] = vocabulary
            return vocabulary
        except Exception as e:
            logger.error(f"Failed to load TF-IDF vocabulary: {e}")
            return {}
    
    def get_student_data(self, data_type):
        """
        Load various types of student data
        
        Args:
            data_type (str): Type of data to load (wellbeing, gpa, bullying, etc.)
            
        Returns:
            pandas.DataFrame or dict: The requested data
        """
        # Define data file paths
        data_paths = {
            "wellbeing": "app/ml_models/Clustering/output/wellbeing_classification_results.csv",
            "gpa": "app/ml_models/ha_outputs/gpa_predictions_with_bins.csv",
            "bullying": "app/ml_models/ha_outputs/community_bully_assignments.csv",
            "student_scores": "app/ml_models/R-GCN_files/student_scores_with_ids.xlsx",
            "friendship_scores": "app/ml_models/R-GCN_files/friendship_scores_with_ids.xlsx",
            "allocations": "app/ml_models/final_class_allocations_ga.xlsx"
        }
        
        # Check if data is already cached
        if data_type in self.data_cache:
            return self.data_cache[data_type]
        
        # Check if the data type is valid
        if data_type not in data_paths:
            logger.error(f"Unknown data type: {data_type}")
            return None
            
        # Load the data
        try:
            file_path = data_paths[data_type]
            
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                return None
                
            # Load data based on file type
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
                
            # Cache the data
            self.data_cache[data_type] = data
            logger.info(f"Loaded {data_type} data with {len(data)} records")
            
            return data
        except Exception as e:
            logger.error(f"Failed to load {data_type} data: {e}")
            return None
    
    def get_intent_model(self):
        """Load the intent classification model"""
        import joblib
        
        if "intent_model" in self.data_cache:
            return self.data_cache["intent_model"]
            
        model_path = "app/models/trained_models/intent_classifier.joblib"
        
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None
                
            model = joblib.load(model_path)
            self.data_cache["intent_model"] = model
            logger.info("Intent classification model loaded")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            return None
    
    def _load_trained_models(self):
        """Load all trained models from the models directory"""
        try:
            models_path = Path(self.models_dir)
            if not models_path.exists():
                logger.warning(f"Models directory {self.models_dir} not found")
                return
                
            model_files = list(models_path.glob("*.pkl"))
            logger.info(f"Found {len(model_files)} trained models")
            
            for model_file in model_files:
                model_name = model_file.stem
                try:
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    
            logger.info(f"Successfully loaded {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_config(self):
        """Get the current configuration"""
        return self.config
    
    def predict(self, model_name, features):
        """
        Make a prediction using a loaded model
        
        Args:
            model_name: Name of the model to use
            features: Feature dictionary or DataFrame
            
        Returns:
            Prediction result or None if error
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
                
            model = self.models[model_name]
            prediction = model.predict(features)
            return prediction
        except Exception as e:
            logger.error(f"Prediction error with model {model_name}: {e}")
            return None
    
    def get_model_metadata(self, model_name):
        """
        Get metadata for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of model metadata or None if not found
        """
        try:
            metadata_file = Path(self.models_dir) / f"{model_name}_metadata.json"
            if not metadata_file.exists():
                return None
                
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metadata for {model_name}: {e}")
            return None
    
    def has_model(self, model_name):
        """Check if a model is available"""
        return model_name in self.models
    
    def train_and_save_model(self, model_name, model_type, X, y, params=None):
        """
        Train a new model and save it (development only)
        
        This should only be used during development/training phase,
        not in the production application.
        
        Args:
            model_name: Name for the model
            model_type: Type of model to train ('logistic', 'svm', etc.)
            X: Feature data
            y: Target data
            params: Model hyperparameters
        
        Returns:
            True if successful, False otherwise
        """
        if self.environment != "development":
            logger.warning("Model training not allowed in production mode")
            return False
            
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create model based on type
            if model_type == 'logistic':
                model = LogisticRegression(**(params or {'max_iter': 1000}))
            elif model_type == 'svm':
                model = SVC(**(params or {'kernel': 'rbf'}))
            elif model_type == 'random_forest':
                model = RandomForestClassifier(**(params or {'n_estimators': 100}))
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
                
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model
            model_path = Path(self.models_dir) / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'model_type': model_type,
                'params': params,
                'performance': report,
                'features': X.columns.tolist() if hasattr(X, 'columns') else None,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = Path(self.models_dir) / f"{model_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Add to loaded models
            self.models[model_name] = model
                
            logger.info(f"Successfully trained and saved model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return False 