#!/usr/bin/env python
"""
Training script for the RuleBasedChatbot in app/models/assistant.py.
This script trains and saves models for the chatbot with target accuracy of 84%+.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_chatbot')

# Fix import path issue
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Import data manager directly without relying on app package
class DataManager:
    """Simple DataManager implementation for training script"""
    def __init__(self, environment="development"):
        self.environment = environment
        logger.info(f"DataManager initialized in {environment} mode")

def load_training_data():
    """Load and prepare training data from CSV files"""
    logger.info("Loading training data...")
    
    data_files = {
        "cluster_assignments": "app/ml_models/Clustering/output/cluster_assignments.csv",
        "bully_assignments": "app/ml_models/ha_outputs/community_bully_assignments.csv",
        "gpa_predictions": "app/ml_models/ha_outputs/gpa_predictions_with_bins.csv",
        "student_wellbeing": "app/ml_models/Clustering/output/student_wellbeing_classifications.csv",
        "cluster_profiles": "app/ml_models/Clustering/output/cluster_profiles.csv",
        "cluster_centers": "app/ml_models/Clustering/output/cluster_centers.csv",
        "pca_loadings": "app/ml_models/Clustering/output/pca_loadings.csv",
        "student_scores": "app/ml_models/R-GCN_files/student_scores_with_ids.xlsx",
        "friendship_scores": "app/ml_models/R-GCN_files/friendship_scores_with_ids.xlsx",
        "final_allocations": "app/ml_models/final_class_allocations_ga.xlsx"
    }
    
    datasets = {}
    
    for name, file_path in data_files.items():
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                datasets[name] = df
                logger.info(f"Loaded {name} dataset with {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {name} dataset: {e}")
        else:
            logger.warning(f"Dataset file not found: {file_path}")
    
    return datasets

def generate_enhanced_intent_training_data():
    """Generate an enhanced training dataset for intent classification with multiple variations"""
    logger.info("Generating enhanced intent training dataset...")
    
    # Define expanded templates for training examples (significantly increased)
    templates = {
        "academic": [
            # Original templates
            "I want to focus on academic performance",
            "How can I improve student grades?",
            "Academic achievement is my priority",
            "I need to boost GPA across all classes",
            "Students need better test scores",
            "How can I help students perform better academically?",
            "We need to improve our academic outcomes",
            "Can you help optimize for academic success?",
            "The school is focusing on academic excellence",
            "Parents are concerned about grades",
            # Additional variations
            "I need to prioritize academics in my classroom",
            "Help me focus more on grades and learning",
            "My main concern is academic progress",
            "I want to prioritize studies and learning outcomes",
            "Academic achievement should be emphasized",
            "Please adjust settings to favor academic performance",
            "I'd like to give more weight to academic factors",
            "Let's make grades the top priority",
            "How can I make academic success the focus?",
            "School performance is most important",
            "Focus on test scores and academic metrics",
            "My primary goal is academic improvement",
            "Please emphasize student achievement in classes",
            "Learning outcomes should be prioritized",
            "I'm worried about academic standards",
            "Need to increase academic focus",
            "Make sure academics are weighted highly",
            "GPA improvement is critical",
            "Academic metrics need to improve",
            "Let's make sure academic factors are prioritized"
        ],
        "wellbeing": [
            # Original templates
            "Student wellbeing should be the priority",
            "I'm concerned about mental health",
            "How can I reduce student stress?",
            "My students seem anxious about school",
            "Emotional wellbeing is important to me",
            "We need to support student mental health",
            "Some students are experiencing stress",
            "I want to create a positive emotional environment",
            "How can we improve student happiness?",
            "Wellbeing should be our top concern",
            # Additional variations
            "Mental health is my top priority",
            "I care most about student happiness",
            "Emotional health needs to be emphasized",
            "Student wellness should be the focus",
            "Please adjust to prioritize wellbeing",
            "I want to focus on student mental health",
            "Psychological wellbeing is most important",
            "Help me create emotionally healthy classrooms",
            "Student stress levels are too high",
            "I need to address anxiety in my students",
            "Can you help me improve student wellbeing?",
            "Emotional support should be our focus",
            "Please weight wellbeing factors heavily",
            "Student happiness is my main concern",
            "Promote better mental health in classes",
            "Help reduce emotional distress",
            "Need to focus on psychological support",
            "Create emotionally safe environments",
            "Stress reduction is critical",
            "Mental health factors need higher weight"
        ],
        "bullying": [
            # Original templates
            "There's a bullying problem in my class",
            "How can I prevent harassment between students?",
            "I need to create a safer classroom environment",
            "Some students are being excluded by others",
            "Bullying prevention should be a top priority",
            "We need to address aggressive behavior",
            "Some students feel unsafe in class",
            "How can I stop bullying in my classroom?",
            "We need better anti-bullying measures",
            "Students report feeling intimidated",
            # Additional variations
            "I'm worried about bullying issues",
            "Help me reduce victimization in classes",
            "Need to separate bullies from victims",
            "Students are harassing others",
            "Please focus on preventing student conflicts",
            "Safety from bullying is my main concern",
            "Need to address aggressive behaviors",
            "Help me manage the bullying situation",
            "I want to prioritize anti-bullying measures",
            "Some kids are being picked on",
            "Make bullying prevention the top priority",
            "Need to create safer spaces for vulnerable students",
            "Conflict between students is increasing",
            "Help stop intimidation in my classroom",
            "Let's focus on reducing aggressive interactions",
            "Need to address harassment between students",
            "Student safety from bullying is critical",
            "Please give more weight to bullying prevention",
            "Victimization needs to be addressed urgently",
            "I want more emphasis on stopping harmful behaviors"
        ],
        "social": [
            # Original templates
            "I want to improve social dynamics",
            "There are issues with social hierarchies",
            "Some students have too much influence",
            "How can I manage peer pressure?",
            "Social interactions need balancing",
            "There are cliques forming in the classroom",
            "Some students dominate group activities",
            "I need to balance social power in the classroom",
            "How can I improve classroom social dynamics?",
            "We need better peer interaction patterns",
            # Additional variations
            "Social balance is my main concern",
            "Help me address social influence issues",
            "Need to create better social dynamics",
            "Please focus on balancing social interactions",
            "I want to improve peer relationships",
            "Social hierachies need to be addressed",
            "Some students have too much social power",
            "Need to balance social influence in class",
            "Help me create healthier social patterns",
            "I'm concerned about social imbalance",
            "Let's prioritize better social dynamics",
            "Need to address peer pressure problems",
            "Can you help with social status issues?",
            "Improve how students interact socially",
            "Balance social factors in my classroom",
            "Focus on creating social equity",
            "Improve group dynamics and social patterns",
            "Need to address unbalanced social influence",
            "Please give more weight to social factors",
            "I want to create better social environments"
        ],
        "friendship": [
            # Original templates
            "Students need more friends in class",
            "How can I strengthen peer relationships?",
            "Some students are isolated from friend groups",
            "I want to ensure every student has a friend",
            "Friendship connections should be improved",
            "Many students lack social connections",
            "How can I help isolated students make friends?",
            "We need to strengthen student bonds",
            "Some kids don't have any friends",
            "I want to foster better friendships",
            # Additional variations
            "Friendships are my top priority",
            "Help me keep friend groups together",
            "I want to maximize friendship connections",
            "Need to ensure students are with their friends",
            "Please prioritize friendship relationships",
            "Focus on maintaining social bonds",
            "Keep friend circles intact please",
            "Help isolated students find friends",
            "I want to emphasize peer connections",
            "Friendship groups should stay together",
            "Make sure students have their friends in class",
            "Prioritize keeping friends together",
            "Need to maintain existing friendships",
            "Ensure friendship connections remain intact",
            "Give high weight to friendship factors",
            "Prevent student isolation by keeping friends together",
            "Build stronger friendship networks",
            "Need to focus on peer relationships",
            "Help students maintain their friendships",
            "Promote stronger student connections"
        ],
        "recommendation": [
            # Original templates
            "What do you recommend for improving class allocation?",
            "Give me recommendations based on the model outputs",
            "How can I optimize class assignments?",
            "What are your suggestions for class formation?",
            "What insights can you provide from the model data?",
            "Show me recommendations from the analysis",
            "What should I do based on the model results?",
            "Provide insights from the data analysis",
            "Give me actionable recommendations",
            "What does the data suggest I should do?",
            # Additional variations
            "I need recommendations for my classes",
            "Help me understand what the data suggests",
            "What actions should I take based on the analysis?",
            "Please provide guidance on class formation",
            "What can I learn from the model results?",
            "Give me advice on optimizing classrooms",
            "What do the insights tell us to do?",
            "I'd like some recommendations please",
            "Based on the data, what should I change?",
            "Can you offer any suggestions?",
            "What findings are most important?",
            "Help me interpret the model results",
            "Give me your top recommendations",
            "What would you suggest based on the data?",
            "I need advice on classroom optimization",
            "Show me what the models recommend",
            "Provide insights from your analysis",
            "What actions should I consider taking?",
            "Share your recommendations for improvement",
            "What patterns should I be aware of?"
        ]
    }
    
    # Generate training data from templates with domain labels
    training_data = []
    for domain, queries in templates.items():
        for query in queries:
            training_data.append({"query": query, "domain": domain})
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(training_data)
    return df.sample(frac=1).reset_index(drop=True)

def train_improved_model():
    """Train an improved intent classification model with better accuracy"""
    logger.info("Training an improved intent classification model...")
    
    # Generate enhanced training data
    df = generate_enhanced_intent_training_data()
    logger.info(f"Generated training dataset with {len(df)} examples")
    
    # Split the data
    X = df['query'].values
    y = df['domain'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set: {len(X_train)} examples")
    logger.info(f"Test set: {len(X_test)} examples")
    
    # Create a pipeline with TF-IDF and multiple classifiers
    # We'll train several models and pick the best
    pipelines = {
        'lr': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000, C=10.0, class_weight='balanced'))
        ]),
        'rf': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)),
            ('clf', RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=20))
        ]),
        'mlp': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)),
            ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, pipeline in pipelines.items():
        logger.info(f"Training {name} model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
    
    # Print classification report for the best model
    y_pred = best_model.predict(X_test)
    logger.info("\nDetailed classification report for the best model:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save the TF-IDF vocabulary
    tfidf_vectorizer = best_model.named_steps['tfidf']
    vocabulary = tfidf_vectorizer.vocabulary_
    
    os.makedirs("app/models/tfidf_vectorizer", exist_ok=True)
    with open("app/models/tfidf_vectorizer/vocabulary.json", 'w') as f:
        json.dump(vocabulary, f)
    
    # Save the best model
    os.makedirs("app/models/trained_models", exist_ok=True)
    joblib.dump(best_model, "app/models/trained_models/intent_classifier.joblib")
    
    logger.info(f"Model saved. Best accuracy: {best_accuracy:.4f}")
    return best_model, vocabulary, best_accuracy

def improve_contextual_understanding():
    """Add contextual understanding capabilities to improve model responses"""
    # Implement additional logic for the RuleBasedChatbot
    contextual_rules = {
        "academic": {
            "keywords": ["academic", "grades", "gpa", "test scores", "achievement", "performance", "study", "learning"],
            "params_to_modify": ["gpa_penalty_weight", "prioritize_academic"],
            "param_adjustments": {"gpa_penalty_weight": 85, "prioritize_academic": 9}
        },
        "wellbeing": {
            "keywords": ["wellbeing", "mental health", "stress", "anxiety", "emotional", "happiness", "wellness", "support"],
            "params_to_modify": ["wellbeing_penalty_weight", "prioritize_wellbeing"],
            "param_adjustments": {"wellbeing_penalty_weight": 85, "prioritize_wellbeing": 9}
        },
        "bullying": {
            "keywords": ["bullying", "harassment", "safe", "exclude", "intimidate", "aggressive", "victim", "unsafe"],
            "params_to_modify": ["bully_penalty_weight", "prioritize_bullying"],
            "param_adjustments": {"bully_penalty_weight": 85, "prioritize_bullying": 9}
        },
        "social": {
            "keywords": ["social dynamics", "hierarchy", "influence", "peer pressure", "cliques", "dominant", "power", "interaction"],
            "params_to_modify": ["influence_std_weight", "prioritize_social_influence", "isolated_std_weight"],
            "param_adjustments": {"influence_std_weight": 85, "prioritize_social_influence": 9, "isolated_std_weight": 85}
        },
        "friendship": {
            "keywords": ["friends", "friendship", "peer", "isolated", "bond", "connection", "relationships", "social ties"],
            "params_to_modify": ["friend_inclusion_weight", "prioritize_friendship", "friendship_balance_weight", "min_friends_required"],
            "param_adjustments": {"friend_inclusion_weight": 85, "prioritize_friendship": 9, "friendship_balance_weight": 85, "min_friends_required": 5}
        }
    }
    
    # Save the contextual rules
    with open("app/models/trained_models/contextual_rules.json", 'w') as f:
        json.dump(contextual_rules, f)
    
    logger.info("Contextual understanding rules saved")
    return contextual_rules

def main():
    """Main function to execute the training process"""
    logger.info("Starting improved chatbot model training")
    
    # Train the improved model
    best_model, vocabulary, accuracy = train_improved_model()
    
    # Add contextual understanding
    contextual_rules = improve_contextual_understanding()
    
    logger.info("Training completed successfully")
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # For testing: simulate evaluation
    if accuracy >= 0.84:
        logger.info("✅ Model achieved target accuracy of 84% or higher!")
    else:
        logger.warning(f"⚠️ Model accuracy ({accuracy:.4f}) is below target (0.84). Further improvements needed.")

if __name__ == "__main__":
    main()
