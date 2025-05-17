import os
import random
import pandas as pd
import json
import re
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Import API client
try:
    from app.models.api_client import api_client
    HAS_API_CLIENT = True
except Exception as e:
    HAS_API_CLIENT = False
    print(f"[WARN] Could not import API client: {e}")

# Try to import config
try:
    from app.config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("[INFO] App config not found, will use default API settings")

class RuleBasedChatbot:
    def __init__(self, api_client_instance=None):
        # Update file paths to use the provided datasets
        self.wellbeing_classification_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
        self.community_bully_file = "app/ml_models/ha_outputs/community_bully_assignments.csv"
        self.gpa_predictions_file = "app/ml_models/ha_outputs/gpa_predictions_with_bins.csv"
        self.social_wellbeing_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
        self.r_gcn_dir = "app/ml_models/R-GCN_files"
        self.config_file = "soft_constraints_config.json"
        self.default_config = {
            "class_size": 30,
            "max_classes": 6,
            "gpa_penalty_weight": 30,
            "wellbeing_penalty_weight": 50,
            "bully_penalty_weight": 60,
            "influence_std_weight": 60, 
            "isolated_std_weight": 60,
            "min_friends_required": 1,
            "friend_inclusion_weight": 50,
            "friendship_balance_weight": 40,
            "prioritize_academic": 5,
            "prioritize_wellbeing": 4,
            "prioritize_bullying": 3,
            "prioritize_social_influence": 2,
            "prioritize_friendship": 1
        }
        
        # Additional data sources
        self.data_sources = {
            'allocations': 'allocations',
            'participants': 'participants',
            'wellbeing': {
                'db_table': 'mental_wellbeing',
                'csv_file': self.wellbeing_classification_file
            },
            'academic': {
                'db_table': 'academic_wellbeing',
                'csv_file': None
            },
            'social': {
                'db_table': 'social_wellbeing',
                'csv_file': self.social_wellbeing_file
            },
            'disrespect': {
                'db_table': 'net_disrespect',
                'csv_file': None
            },
            'friends': {
                'db_table': 'net_friends',
                'csv_file': None
            },
            'influence': {
                'db_table': 'net_influential',
                'csv_file': None
            },
            'bullying': {
                'db_table': None,
                'csv_file': self.community_bully_file
            },
            'gpa': {
                'db_table': None,
                'csv_file': self.gpa_predictions_file
            },
            'responses': {
                'db_table': 'responses',
                'csv_file': None
            }
        }
        
        # Enhanced with detailed mapping schema
        self.nl_to_param_mapping = {
            # Class size mappings
            "small class size": {"class_size": 20},
            "smaller classes": {"class_size": 25},
            "reduced class size": {"class_size": 25},
            "large class size": {"class_size": 35},
            "larger classes": {"class_size": 35},
            "medium class size": {"class_size": 30},
            
            # Academic performance mappings
            "focus on academics": {"gpa_penalty_weight": 80, "prioritize_academic": 5},
            "academic excellence": {"gpa_penalty_weight": 90, "prioritize_academic": 5},
            "balance academics": {"gpa_penalty_weight": 60, "prioritize_academic": 3},
            "reduce academic pressure": {"gpa_penalty_weight": 40, "prioritize_academic": 2},
            
            # Wellbeing mappings
            "focus on wellbeing": {"wellbeing_penalty_weight": 80, "prioritize_wellbeing": 5},
            "prioritize mental health": {"wellbeing_penalty_weight": 90, "prioritize_wellbeing": 5},
            "support student wellbeing": {"wellbeing_penalty_weight": 70, "prioritize_wellbeing": 4},
            
            # Bullying prevention mappings
            "prevent bullying": {"bully_penalty_weight": 90, "prioritize_bullying": 5},
            "anti-bullying": {"bully_penalty_weight": 95, "prioritize_bullying": 5},
            "address bullying": {"bully_penalty_weight": 85, "prioritize_bullying": 4},
            "reduce bullying": {"bully_penalty_weight": 80, "prioritize_bullying": 4},
            
            # Social influence mappings
            "manage social dynamics": {"influence_std_weight": 70, "prioritize_social_influence": 4},
            "balance influence": {"influence_std_weight": 75, "isolated_std_weight": 75, "prioritize_social_influence": 4},
            "address isolation": {"isolated_std_weight": 80, "friend_inclusion_weight": 70, "prioritize_social_influence": 4},
            
            # Friendship mappings
            "strong peer support": {"friend_inclusion_weight": 80, "prioritize_friendship": 4},
            "friend groups": {"min_friends_required": 2, "friend_inclusion_weight": 70, "prioritize_friendship": 4},
            "friendship balance": {"friendship_balance_weight": 75, "prioritize_friendship": 4},
            "more friends": {"min_friends_required": 3, "friend_inclusion_weight": 80},
            "fewer friends": {"min_friends_required": 1, "friend_inclusion_weight": 40}
        }
        
        # Keywords for intent detection
        self.intent_keywords = {
            "academic": ["academic", "gpa", "grades", "performance", "study", "studies", "learning", "achievement"],
            "wellbeing": ["wellbeing", "well-being", "mental health", "anxiety", "stress", "overwhelm", "happiness", "emotional"],
            "bullying": ["bully", "bullying", "victim", "harass", "safe", "protection", "safe space"],
            "social": ["social", "influence", "leaders", "dynamics", "interaction", "peers", "inclusive", "inclusion"],
            "friendship": ["friend", "friendship", "peers", "support", "connection", "social network", "buddy"]
        }
        
        # Modifiers for intensity detection
        self.intensity_modifiers = {
            "high": 1.2,  # Increase the parameter value
            "very": 1.3,
            "extremely": 1.4,
            "strong": 1.2,
            "maximum": 1.5,
            "highest": 1.5,
            "low": 0.8,  # Decrease the parameter value
            "minimal": 0.7,
            "reduced": 0.8,
            "less": 0.8,
            "least": 0.6,
            "moderate": 1.0,
            "balanced": 1.0
        }
        
        self.sample_responses = {
            "academic_priority": "I recommend prioritizing academic balance with a higher GPA penalty weight (60-70) and setting academic as the top priority (5).",
            "wellbeing_priority": "For student wellbeing focus, I suggest increasing the wellbeing_penalty_weight to 70-80 and setting it as the top priority (5).",
            "bullying_prevention": "To address bullying concerns, set bully_penalty_weight to 80-90 and make bullying prevention the top priority (5).",
            "social_balance": "For better social dynamics, increase influence_std_weight and isolated_std_weight to 70, and prioritize social influence (4-5).",
            "friendship_focus": "To optimize friendship connections, set min_friends_required to 2 and increase friend_inclusion_weight to 70-80."
        }
        
        # Conversation memory
        self.conversation_history = []
        self.data_cache = {}
        
        # Setup API client
        self.api_client = api_client_instance if api_client_instance else None
        
        if not self.api_client and HAS_API_CLIENT:
            # Use the imported global api_client
            self.api_client = api_client
            print("[INFO] Using imported API client")
        
        # Comprehensive data loading from all sources via API
        print("[INFO] Starting comprehensive data loading via API client")
        self._load_data_comprehensive()
        self._initialize_vectorizer()
        self._initialize_ml_model()
        self._extract_parameter_patterns()
        print("[INFO] Model initialization and training complete")
        
    @property
    def processed_features(self):
        """Cache processed features in memory to avoid recomputing"""
        if not hasattr(self, '_processed_features'):
            # Do expensive processing once
            print("[INFO] Computing processed features")
            self._processed_features = self._prepare_integrated_dataset()
        return self._processed_features

    def _load_data_comprehensive(self):
        """
        Load data from all available sources in a systematic way.
        Prioritizes API access, falls back to CSV for each data type.
        """
        # First, try to use API client for all data types
        if self.api_client:
            print("[INFO] Loading all available data from API")
            
            # Load all data from API first
            api_data = self.api_client.load_data_comprehensive()
            if api_data and len(api_data) > 0:
                self.data_cache.update(api_data)
                print(f"[INFO] Successfully loaded {len(api_data)} datasets from API")
                
                # Track which data types we've loaded
                loaded_data_types = set(self.data_cache.keys())
                print(f"[INFO] Data types loaded from API: {', '.join(loaded_data_types)}")
                
                # Make sure we've loaded the essential data
                essential_types = {'students', 'wellbeing', 'gpa', 'social', 'bullying'}
                missing_types = essential_types - loaded_data_types
                
                if missing_types:
                    print(f"[INFO] Missing essential data types: {', '.join(missing_types)}")
                    # Try to load missing types from CSV files
                    self._load_missing_csv_data(missing_types)
                else:
                    print("[INFO] All essential data types loaded from API")
            else:
                print("[WARN] Failed to load data from API, falling back to CSV files")
                self._load_csv_data()
        else:
            # API client not available, load from CSV files
            print("[WARN] API client not available, loading from CSV files")
            self._load_csv_data()
            
        # Generate synthetic teacher comments
        self._generate_synthetic_comments()
        
        # Create integrated dataset from all loaded data
        self._prepare_integrated_dataset()
        
        # Verify we have all needed data
        self._verify_data_integrity()

    def _load_missing_csv_data(self, missing_types):
        """Load missing data types from CSV files"""
        print(f"[INFO] Loading missing data types from CSV files: {', '.join(missing_types)}")
        
        for data_type in missing_types:
            if data_type == 'wellbeing' and os.path.exists(self.wellbeing_classification_file):
                self.data_cache['wellbeing'] = pd.read_csv(self.wellbeing_classification_file)
                print(f"[INFO] Loaded {len(self.data_cache['wellbeing'])} wellbeing classification records from CSV")
            
            elif data_type == 'bullying' and os.path.exists(self.community_bully_file):
                self.data_cache['bullying'] = pd.read_csv(self.community_bully_file)
                print(f"[INFO] Loaded {len(self.data_cache['bullying'])} community bully assignments from CSV")
            
            elif data_type == 'gpa' and os.path.exists(self.gpa_predictions_file):
                self.data_cache['gpa'] = pd.read_csv(self.gpa_predictions_file)
                print(f"[INFO] Loaded {len(self.data_cache['gpa'])} GPA predictions from CSV")
            
            elif data_type == 'social' and os.path.exists(self.social_wellbeing_file):
                # Load social data from cluster_assignments.csv
                df = pd.read_csv(self.social_wellbeing_file)
                
                # Make sure the dataframe has the required columns
                if 'Participant-ID' in df.columns:
                    # Create social_wellbeing column if it doesn't exist
                    if 'social_wellbeing' not in df.columns:
                        # Map wellbeing labels to numeric values
                        wellbeing_map = {
                            'High Wellbeing': 80,
                            'Moderate Wellbeing': 50,
                            'Low Wellbeing': 30
                        }
                        # Use wellbeing_label to create social_wellbeing score
                        if 'wellbeing_label' in df.columns:
                            df['social_wellbeing'] = df['wellbeing_label'].map(wellbeing_map)
                        else:
                            # Default value
                            df['social_wellbeing'] = 50
                    
                    # Make sure there's a cluster column
                    if 'cluster' not in df.columns and 'cluster_label' in df.columns:
                        df['cluster'] = df['cluster_label']
                        
                    self.data_cache['social'] = df
                    print(f"[INFO] Loaded {len(df)} social wellbeing records from CSV")
                else:
                    print(f"[WARN] Invalid format in social wellbeing file: missing required columns")
            
            elif data_type == 'students':
                # Create synthetic student data if needed
                students_data = []
                
                # Try to get student IDs from other datasets
                student_ids = set()
                
                # Extract from wellbeing data
                if 'wellbeing' in self.data_cache and 'Participant-ID' in self.data_cache['wellbeing'].columns:
                    student_ids.update(self.data_cache['wellbeing']['Participant-ID'].unique())
                
                # Extract from social data
                if 'social' in self.data_cache and 'Participant-ID' in self.data_cache['social'].columns:
                    student_ids.update(self.data_cache['social']['Participant-ID'].unique())
                
                # Extract from bullying data
                if 'bullying' in self.data_cache and 'Student_ID' in self.data_cache['bullying'].columns:
                    student_ids.update(self.data_cache['bullying']['Student_ID'].unique())
                
                # Extract from GPA data
                if 'gpa' in self.data_cache and 'Student_ID' in self.data_cache['gpa'].columns:
                    student_ids.update(self.data_cache['gpa']['Student_ID'].unique())
                
                # Create student records
                for student_id in student_ids:
                    students_data.append({
                        'id': student_id,
                        'name': f"Student {student_id}",
                        'email': f"student{student_id}@example.com",
                        'class_id': "C1"  # Default class
                    })
                
                if students_data:
                    self.data_cache['students'] = pd.DataFrame(students_data)
                    print(f"[INFO] Created {len(students_data)} synthetic student records")

    def _load_csv_data(self):
        """Load all required datasets from CSV files"""
        # Load wellbeing classification data
        if os.path.exists(self.wellbeing_classification_file):
            self.data_cache['wellbeing'] = pd.read_csv(self.wellbeing_classification_file)
            print(f"[INFO] Loaded {len(self.data_cache['wellbeing'])} wellbeing classification records")
        else:
            print("[WARN] Wellbeing classification file not found")
            
        # Load community bully assignments
        if os.path.exists(self.community_bully_file):
            self.data_cache['bullying'] = pd.read_csv(self.community_bully_file)
            print(f"[INFO] Loaded {len(self.data_cache['bullying'])} community bully assignments")
        else:
            print("[WARN] Community bully assignments file not found")
            
        # Load GPA predictions
        if os.path.exists(self.gpa_predictions_file):
            self.data_cache['gpa'] = pd.read_csv(self.gpa_predictions_file)
            print(f"[INFO] Loaded {len(self.data_cache['gpa'])} GPA predictions")
        else:
            print("[WARN] GPA predictions file not found")
            
        # Load social wellbeing predictions data
        if os.path.exists(self.social_wellbeing_file):
            self.data_cache['social'] = pd.read_csv(self.social_wellbeing_file)
            print(f"[INFO] Loaded {len(self.data_cache['social'])} social wellbeing records")
        else:
            print("[WARN] Social wellbeing file not found")
        
        # Check R-GCN files
        if os.path.exists(self.r_gcn_dir):
            r_gcn_files = os.listdir(self.r_gcn_dir)
            self.data_cache['r_gcn_files'] = r_gcn_files
            print(f"[INFO] Found {len(r_gcn_files)} R-GCN files")
        else:
            print("[WARN] R-GCN directory not found")

    def _verify_data_integrity(self):
        """Verify that all essential data is loaded"""
        essential_types = {'wellbeing', 'bullying', 'gpa', 'social'}
        loaded_types = set(self.data_cache.keys())
        
        missing_types = essential_types - loaded_types
        if missing_types:
            print(f"[WARN] Missing essential data types: {', '.join(missing_types)}")
            
            # Generate placeholder data if needed
            for missing_type in missing_types:
                if missing_type == 'wellbeing':
                    print("[INFO] Generating placeholder wellbeing data")
                    # Create placeholder data if students are available
                    if 'students' in self.data_cache:
                        students = self.data_cache['students']
                        wellbeing_data = []
                        for student in students:
                            # Create random wellbeing label
                            wellbeing_data.append({
                                'Participant-ID': student['id'],
                                'wellbeing_label': 'High Wellbeing' if random.random() > 0.5 else 'Low Wellbeing',
                                'School_support_engage6': random.uniform(0, 100),
                                'GrowthMindset': random.uniform(0, 100)
                            })
                        self.data_cache['wellbeing'] = pd.DataFrame(wellbeing_data)
                        print(f"[INFO] Generated {len(wellbeing_data)} placeholder wellbeing records")
                
                elif missing_type == 'bullying':
                    print("[INFO] Generating placeholder bullying data")
                    # Create placeholder data if students are available
                    if 'students' in self.data_cache:
                        students = self.data_cache['students']
                        bullying_data = []
                        for student in students:
                            # Create random bullying data (10% are bullies)
                            is_bully = 1 if random.random() < 0.1 else 0
                            bullying_data.append({
                                'Student_ID': student['id'],
                                'Community_ID': random.randint(1, 5),
                                'Is_Bully': is_bully,
                                'Primary_Bully_ID': student['id'] if is_bully else None
                            })
                        self.data_cache['bullying'] = pd.DataFrame(bullying_data)
                        print(f"[INFO] Generated {len(bullying_data)} placeholder bullying records")
                
                elif missing_type == 'gpa':
                    print("[INFO] Generating placeholder GPA data")
                    # Create placeholder data if students are available
                    if 'students' in self.data_cache:
                        students = self.data_cache['students']
                        gpa_data = []
                        for student in students:
                            # Use academic percentage if available, otherwise random
                            gpa = student.get('perc_academic', random.uniform(0, 100))
                            # Determine bin (0-3) based on quartiles
                            gpa_bin = 0
                            if gpa > 75:
                                gpa_bin = 3
                            elif gpa > 50:
                                gpa_bin = 2
                            elif gpa > 25:
                                gpa_bin = 1
                            
                            gpa_data.append({
                                'Student_ID': student['id'],
                                'Predicted_GPA': gpa,
                                'GPA_Bin': gpa_bin
                            })
                        self.data_cache['gpa'] = pd.DataFrame(gpa_data)
                        print(f"[INFO] Generated {len(gpa_data)} placeholder GPA records")
                
                elif missing_type == 'social':
                    print("[INFO] Generating placeholder social data")
                    # Create placeholder data if students are available
                    if 'students' in self.data_cache:
                        students = self.data_cache['students']
                        social_data = []
                        for student in students:
                            # Create random social wellbeing
                            social_score = random.uniform(0, 100)
                            social_data.append({
                                'Participant-ID': student['id'],
                                'cluster': 1 if social_score > 50 else 0,
                                'social_wellbeing': social_score
                            })
                        self.data_cache['social'] = pd.DataFrame(social_data)
                        print(f"[INFO] Generated {len(social_data)} placeholder social records")
        else:
            print("[INFO] All essential data types are loaded")

    def _generate_synthetic_comments(self):
        """Generate synthetic teacher comments based on the loaded datasets"""
        try:
            # Check if we have the necessary datasets
            required_data = ['wellbeing', 'bullying', 'gpa', 'social']
            available_data = [k for k in required_data if k in self.data_cache]
            
            if len(available_data) < 2:
                print("[WARN] Not enough datasets available to generate synthetic comments")
                return
                
            print(f"[INFO] Generating synthetic teacher comments from {len(available_data)} datasets")
            
            # Create a mapping of student IDs to their data
            student_data = {}
            
            # Process wellbeing data
            if 'wellbeing' in self.data_cache:
                for _, row in self.data_cache['wellbeing'].iterrows():
                    if 'Participant-ID' in row:
                        student_id = row['Participant-ID']
                        if student_id not in student_data:
                            student_data[student_id] = {}
                        
                        # Assign wellbeing data
                        if 'wellbeing_label' in row:
                            student_data[student_id]['wellbeing_label'] = row['wellbeing_label']
                        
                        # Get other features if available
                        for col in ['School_support_engage6', 'GrowthMindset']:
                            if col in row:
                                student_data[student_id][col] = row.get(col, 0)
            
            # Process bullying data
            if 'bullying' in self.data_cache:
                for _, row in self.data_cache['bullying'].iterrows():
                    if 'Student_ID' in row:
                        student_id = row['Student_ID']
                        if student_id not in student_data:
                            student_data[student_id] = {}
                        
                        # Assign bullying data
                        student_data[student_id]['is_bully'] = row.get('Is_Bully', 0) == 1
                        student_data[student_id]['community_id'] = row.get('Community_ID', 0)
            
            # Process GPA data
            if 'gpa' in self.data_cache:
                for _, row in self.data_cache['gpa'].iterrows():
                    if 'Student_ID' in row:
                        student_id = row['Student_ID']
                        if student_id not in student_data:
                            student_data[student_id] = {}
                        
                        # Assign GPA data
                        student_data[student_id]['predicted_gpa'] = row.get('Predicted_GPA', 0)
                        student_data[student_id]['gpa_bin'] = row.get('GPA_Bin', 0)
            
            # Process social data
            if 'social' in self.data_cache:
                for _, row in self.data_cache['social'].iterrows():
                    if 'Participant-ID' in row:
                        student_id = row['Participant-ID']
                        if student_id not in student_data:
                            student_data[student_id] = {}
                        
                        # Assign social data
                        if 'social_wellbeing' in row:
                            student_data[student_id]['social_wellbeing'] = row['social_wellbeing']
                        elif 'wellbeing_label' in row:
                            # Map wellbeing labels to scores
                            wellbeing_map = {
                                'High Wellbeing': 80,
                                'Moderate Wellbeing': 50,
                                'Low Wellbeing': 30
                            }
                            student_data[student_id]['social_wellbeing'] = wellbeing_map.get(row['wellbeing_label'], 50)
                        
                        # Get cluster if available
                        if 'cluster' in row:
                            student_data[student_id]['cluster'] = row['cluster']
                        elif 'cluster_label' in row:
                            student_data[student_id]['cluster'] = row['cluster_label']
            
            # Generate synthetic comments and recommendations
            comments = []
            recommendations = []
            
            comment_templates = {
                'high_gpa': [
                    "Student {} is performing excellently academically.",
                    "Student {} has shown great academic progress.",
                    "Student {} consistently achieves high grades."
                ],
                'low_gpa': [
                    "Student {} is struggling academically.",
                    "Student {} needs academic support.",
                    "Student {} is not meeting academic expectations."
                ],
                'high_wellbeing': [
                    "Student {} appears to have good mental wellbeing.",
                    "Student {} shows positive emotional health.",
                    "Student {} demonstrates resilience and positive attitude."
                ],
                'low_wellbeing': [
                    "Student {} seems to be experiencing stress.",
                    "Student {} shows signs of emotional distress.",
                    "Student {} may benefit from wellbeing support."
                ],
                'is_bully': [
                    "Student {} has been identified as exhibiting bullying behaviors.",
                    "Student {} needs intervention for bullying behaviors.",
                    "Student {} has been bullying others in their community."
                ],
                'high_social': [
                    "Student {} has strong social connections.",
                    "Student {} demonstrates good social skills.",
                    "Student {} is well-integrated socially."
                ],
                'low_social': [
                    "Student {} appears socially isolated.",
                    "Student {} struggles with peer relationships.",
                    "Student {} would benefit from social support."
                ]
            }
            
            recommendation_templates = {
                'academic': [
                    "Increase gpa_penalty_weight to {} and prioritize_academic to {}.",
                    "Focus on academic support with gpa_penalty_weight set to {} and prioritize_academic to {}.",
                    "Balance academic considerations with gpa_penalty_weight at {} and prioritize_academic at {}."
                ],
                'wellbeing': [
                    "Set wellbeing_penalty_weight to {} and prioritize_wellbeing to {}.",
                    "Support wellbeing needs with wellbeing_penalty_weight at {} and prioritize_wellbeing at {}.",
                    "Address emotional health with wellbeing_penalty_weight set to {} and prioritize_wellbeing to {}."
                ],
                'bullying': [
                    "Set bully_penalty_weight to {} and prioritize_bullying to {}.",
                    "Prevent bullying with bully_penalty_weight at {} and prioritize_bullying at {}.",
                    "Address social safety with bully_penalty_weight set to {} and prioritize_bullying to {}."
                ],
                'social': [
                    "Adjust influence_std_weight to {} and isolated_std_weight to {} with prioritize_social_influence at {}.",
                    "Balance social dynamics with influence_std_weight at {} and isolated_std_weight at {}, prioritize_social_influence at {}.",
                    "Support social integration with influence_std_weight at {} and isolated_std_weight at {}, prioritize_social_influence set to {}."
                ],
                'friendship': [
                    "Set min_friends_required to {} and friend_inclusion_weight to {} with prioritize_friendship at {}.",
                    "Support friendship development with min_friends_required at {} and friend_inclusion_weight at {}, prioritize_friendship at {}.",
                    "Facilitate peer connections with min_friends_required at {} and friend_inclusion_weight at {}, prioritize_friendship set to {}."
                ]
            }
            
            # Generate synthetic comments and recommendations for each student
            for student_id, data in student_data.items():
                comment_parts = []
                
                # Add academic comment if GPA data is available
                if 'predicted_gpa' in data:
                    gpa = data['predicted_gpa']
                    if gpa > 60:  # Assuming 60 is the threshold for good performance
                        template = random.choice(comment_templates['high_gpa'])
                        comment_parts.append(template.format(student_id))
                    else:
                        template = random.choice(comment_templates['low_gpa'])
                        comment_parts.append(template.format(student_id))
                
                # Add wellbeing comment if wellbeing data is available
                if 'wellbeing_label' in data:
                    if data['wellbeing_label'] == 'High Wellbeing':
                        template = random.choice(comment_templates['high_wellbeing'])
                        comment_parts.append(template.format(student_id))
                    else:
                        template = random.choice(comment_templates['low_wellbeing'])
                        comment_parts.append(template.format(student_id))
                
                # Add bullying comment if bullying data is available
                if 'is_bully' in data and data['is_bully']:
                    template = random.choice(comment_templates['is_bully'])
                    comment_parts.append(template.format(student_id))
                
                # Add social comment if social data is available
                if 'social_wellbeing' in data:
                    social_score = data['social_wellbeing']
                    if social_score > 50:  # Assuming 50 is the threshold for good social wellbeing
                        template = random.choice(comment_templates['high_social'])
                        comment_parts.append(template.format(student_id))
                    else:
                        template = random.choice(comment_templates['low_social'])
                        comment_parts.append(template.format(student_id))
                
                if comment_parts:
                    # Combine comments with "and" or semicolons
                    comment = " ".join(comment_parts)
                    comments.append(comment)
                    
                    # Generate recommendation based on student data
                    rec_parts = []
                    
                    # Academic recommendation
                    if 'predicted_gpa' in data:
                        gpa = data['predicted_gpa']
                        if gpa < 60:
                            # Low GPA needs higher academic priority
                            gpa_weight = random.randint(70, 90)
                            academic_priority = random.randint(4, 5)
                            template = random.choice(recommendation_templates['academic'])
                            rec_parts.append(template.format(gpa_weight, academic_priority))
                    
                    # Wellbeing recommendation
                    if 'wellbeing_label' in data:
                        if data['wellbeing_label'] == 'Low Wellbeing':
                            # Low wellbeing needs higher wellbeing priority
                            wellbeing_weight = random.randint(70, 90)
                            wellbeing_priority = random.randint(4, 5)
                            template = random.choice(recommendation_templates['wellbeing'])
                            rec_parts.append(template.format(wellbeing_weight, wellbeing_priority))
                    
                    # Bullying recommendation
                    if 'is_bully' in data and data['is_bully']:
                        # Bullying needs higher bullying priority
                        bully_weight = random.randint(80, 95)
                        bully_priority = random.randint(4, 5)
                        template = random.choice(recommendation_templates['bullying'])
                        rec_parts.append(template.format(bully_weight, bully_priority))
                    
                    # Social recommendation
                    if 'social_wellbeing' in data:
                        social_score = data['social_wellbeing']
                        if social_score < 50:
                            # Low social wellbeing needs higher social priority
                            influence_weight = random.randint(70, 85)
                            isolated_weight = random.randint(70, 85)
                            social_priority = random.randint(4, 5)
                            template = random.choice(recommendation_templates['social'])
                            rec_parts.append(template.format(influence_weight, isolated_weight, social_priority))
                    
                    # Friendship recommendation
                    if 'social_wellbeing' in data and 'cluster' in data:
                        social_score = data['social_wellbeing']
                        if social_score < 50:
                            # Low social wellbeing needs more friends
                            min_friends = random.randint(2, 3)
                            friendship_weight = random.randint(70, 85)
                            friendship_priority = random.randint(3, 5)
                            template = random.choice(recommendation_templates['friendship'])
                            rec_parts.append(template.format(min_friends, friendship_weight, friendship_priority))
                    
                    if rec_parts:
                        recommendation = " ".join(rec_parts)
                        recommendations.append(recommendation)
                    else:
                        # Default balanced recommendation if no specific issues
                        recommendations.append("Maintain balanced parameters with moderate weights across all domains.")
            
            # Create a synthetic teacher comments dataframe
            if comments and recommendations:
                synthetic_df = pd.DataFrame({
                    'comment': comments,
                    'recommendation': recommendations
                })
                
                # Store in data cache
                self.data_cache['teacher_comments'] = synthetic_df
                print(f"[INFO] Generated {len(synthetic_df)} synthetic teacher comments")
            else:
                print("[WARN] Could not generate synthetic teacher comments")
        
        except Exception as e:
            print(f"[ERROR] Failed to generate synthetic comments: {str(e)}")
            # Create at least some dummy data
            comments = ["Default synthetic comment"] * 5
            recommendations = ["Balance all factors equally"] * 5
            self.data_cache['teacher_comments'] = pd.DataFrame({
                'comment': comments,
                'recommendation': recommendations
            })
            print(f"[INFO] Generated 5 fallback synthetic comments after error")
            
    def _prepare_integrated_dataset(self):
        """Prepare an integrated dataset that combines all data sources for enhanced ML modeling"""
        try:
            print("[INFO] Preparing integrated dataset from all available sources")
            
            # Create a student-centered dataset with all available features
            student_ids = set()
            
            # Collect all student IDs from different datasets
            if 'wellbeing' in self.data_cache:
                id_col = 'Participant-ID' if 'Participant-ID' in self.data_cache['wellbeing'].columns else None
                if id_col:
                    student_ids.update(self.data_cache['wellbeing'][id_col].unique())
            
            if 'bullying' in self.data_cache:
                id_col = 'Student_ID' if 'Student_ID' in self.data_cache['bullying'].columns else None
                if id_col:
                    student_ids.update(self.data_cache['bullying'][id_col].unique())
            
            if 'gpa' in self.data_cache:
                id_col = 'Student_ID' if 'Student_ID' in self.data_cache['gpa'].columns else None
                if id_col:
                    student_ids.update(self.data_cache['gpa'][id_col].unique())
            
            if 'social' in self.data_cache:
                id_col = 'Participant-ID' if 'Participant-ID' in self.data_cache['social'].columns else None
                if id_col:
                    student_ids.update(self.data_cache['social'][id_col].unique())
            
            # Create integrated dataset
            integrated_data = []
            
            for student_id in student_ids:
                student_record = {'StudentID': student_id}
                
                # Add wellbeing data
                if 'wellbeing' in self.data_cache:
                    wb_df = self.data_cache['wellbeing']
                    id_col = 'Participant-ID' if 'Participant-ID' in wb_df.columns else None
                    if id_col:
                        wb_row = wb_df[wb_df[id_col] == student_id]
                        if not wb_row.empty:
                            if 'wellbeing_label' in wb_row.columns:
                                student_record['wellbeing_label'] = wb_row['wellbeing_label'].iloc[0]
                            
                            for feature in ['School_support_engage6', 'GrowthMindset']:
                                if feature in wb_row.columns:
                                    student_record[feature.replace('_engage6', '')] = wb_row[feature].iloc[0]
                
                # Add bullying data
                if 'bullying' in self.data_cache:
                    bully_df = self.data_cache['bullying']
                    id_col = 'Student_ID' if 'Student_ID' in bully_df.columns else None
                    if id_col:
                        bully_row = bully_df[bully_df[id_col] == student_id]
                        if not bully_row.empty:
                            if 'Is_Bully' in bully_row.columns:
                                student_record['is_bully'] = bully_row['Is_Bully'].iloc[0]
                            if 'Community_ID' in bully_row.columns:
                                student_record['community_id'] = bully_row['Community_ID'].iloc[0]
                
                # Add GPA data
                if 'gpa' in self.data_cache:
                    gpa_df = self.data_cache['gpa']
                    id_col = 'Student_ID' if 'Student_ID' in gpa_df.columns else None
                    if id_col:
                        gpa_row = gpa_df[gpa_df[id_col] == student_id]
                        if not gpa_row.empty:
                            if 'Predicted_GPA' in gpa_row.columns:
                                student_record['predicted_gpa'] = gpa_row['Predicted_GPA'].iloc[0]
                            if 'GPA_Bin' in gpa_row.columns:
                                student_record['gpa_bin'] = gpa_row['GPA_Bin'].iloc[0]
                
                # Add social data
                if 'social' in self.data_cache:
                    social_df = self.data_cache['social']
                    id_col = 'Participant-ID' if 'Participant-ID' in social_df.columns else None
                    if id_col:
                        social_row = social_df[social_df[id_col] == student_id]
                        if not social_row.empty:
                            if 'social_wellbeing' in social_row.columns:
                                student_record['social_wellbeing'] = social_row['social_wellbeing'].iloc[0]
                            
                            # Get cluster
                            cluster_col = None
                            for col in ['cluster', 'cluster_label']:
                                if col in social_row.columns:
                                    cluster_col = col
                                    break
                                    
                            if cluster_col:
                                student_record['social_cluster'] = social_row[cluster_col].iloc[0]
                
                integrated_data.append(student_record)
            
            # Create dataframe
            integrated_df = pd.DataFrame(integrated_data)
            self.data_cache['integrated'] = integrated_df
            print(f"[INFO] Created integrated dataset with {len(integrated_df)} student records")
            return integrated_df
            
        except Exception as e:
            print(f"[ERROR] Failed to prepare integrated dataset: {str(e)}")
            # Create minimal integrated dataset for fallback
            minimal_df = pd.DataFrame([{'StudentID': 'default'}])
            self.data_cache['integrated'] = minimal_df
            return minimal_df

    def _initialize_vectorizer(self):
        if 'teacher_comments' in self.data_cache:
            comments = self.data_cache['teacher_comments']['comment'].tolist()
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(comments)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(["academic", "wellbeing", "bullying", "social", "friendship"])
    
    def _initialize_ml_model(self):
        """Initialize enhanced ML model that leverages all available data sources"""
        try:
            # Check if we have teacher comments data (real or synthetic)
            if 'teacher_comments' in self.data_cache:
                print("[INFO] Initializing enhanced ML classification model...")
                df = self.data_cache['teacher_comments']
                
                # Encode recommendation labels
                self.label_encoder = LabelEncoder()
                df["label"] = self.label_encoder.fit_transform(df["recommendation"])
                
                # Create enhanced ML pipeline with TF-IDF and LogisticRegression
                self.ml_model = Pipeline([
                    ("tfidf", TfidfVectorizer(stop_words="english", 
                                             ngram_range=(1, 2),  # Include bigrams for better context
                                             max_features=1000)),  # Limit features to prevent overfitting
                    ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
                ])
                
                # Train the model on comment-recommendation pairs
                self.ml_model.fit(df["comment"], df["label"])
                print(f"[INFO] ML model successfully trained on {len(df)} teacher comments")
                
                # Store the recommendations for inverse transform
                self.unique_recommendations = df["recommendation"].unique()
                self.is_ml_model_available = True
                
                # Create a more advanced recommendation system
                print("[INFO] Initializing advanced recommendation system with available data sources")
                self._initialize_advanced_recommendation_system()
            else:
                print("[WARN] Teacher comments data not available for ML model training")
                self.is_ml_model_available = False
        except Exception as e:
            print(f"[ERROR] Failed to initialize ML model: {e}")
            self.is_ml_model_available = False
            
    def _initialize_advanced_recommendation_system(self):
        """Initialize a more advanced recommendation system that uses all data sources"""
        try:
            # Create an enriched recommendation system that factors in wellbeing, bullying, GPA and social data
            self.enhanced_recommendations = {}
            
            # Extract wellbeing patterns if available
            if 'wellbeing' in self.data_cache:
                wb_data = self.data_cache['wellbeing']
                # Find patterns in high vs low wellbeing students
                high_wb = wb_data[wb_data['wellbeing_label'] == 'High Wellbeing']
                low_wb = wb_data[wb_data['wellbeing_label'] == 'Low Wellbeing']
                
                # Calculate percentage of students with high wellbeing
                high_wb_pct = len(high_wb) / len(wb_data) * 100
                
                # Create recommendation patterns based on wellbeing distribution
                self.enhanced_recommendations['high_wellbeing'] = {
                    'wellbeing_penalty_weight': 60,
                    'prioritize_wellbeing': 3,
                    'friend_inclusion_weight': 50
                }
                
                self.enhanced_recommendations['low_wellbeing'] = {
                    'wellbeing_penalty_weight': 85,
                    'prioritize_wellbeing': 5,
                    'friend_inclusion_weight': 75,
                    'min_friends_required': 2
                }
                
                print(f"[INFO] Wellbeing analysis: {high_wb_pct:.1f}% students have high wellbeing")
            
            # Extract bullying patterns if available
            if 'bullying' in self.data_cache:
                bully_data = self.data_cache['bullying']
                # Count number of bullies
                bully_count = bully_data[bully_data['Is_Bully'] == 1].shape[0]
                bully_pct = bully_count / len(bully_data) * 100
                
                # Create recommendation patterns based on bullying prevalence
                self.enhanced_recommendations['high_bullying'] = {
                    'bully_penalty_weight': 90,
                    'prioritize_bullying': 5
                }
                
                self.enhanced_recommendations['low_bullying'] = {
                    'bully_penalty_weight': 60,
                    'prioritize_bullying': 3
                }
                
                print(f"[INFO] Bullying analysis: {bully_pct:.1f}% students identified as bullies")
            
            # Extract GPA patterns if available
            if 'gpa' in self.data_cache:
                gpa_data = self.data_cache['gpa']
                # Calculate average GPA and distribution across bins
                avg_gpa = gpa_data['Predicted_GPA'].mean()
                bin_counts = gpa_data['GPA_Bin'].value_counts(normalize=True) * 100
                
                # Create recommendation patterns based on GPA distribution
                self.enhanced_recommendations['high_gpa'] = {
                    'gpa_penalty_weight': 60,
                    'prioritize_academic': 3
                }
                
                self.enhanced_recommendations['low_gpa'] = {
                    'gpa_penalty_weight': 85,
                    'prioritize_academic': 5
                }
                
                print(f"[INFO] GPA analysis: Average GPA is {avg_gpa:.1f}")
            
            # Extract social cluster patterns if available
            if 'social' in self.data_cache:
                soc_data = self.data_cache['social']
                # Identify clusters
                cluster_0 = soc_data[soc_data['cluster'] == 0]
                cluster_1 = soc_data[soc_data['cluster'] == 1]
                
                # Calculate average social wellbeing by cluster
                avg_social_0 = cluster_0['social_wellbeing'].mean()
                avg_social_1 = cluster_1['social_wellbeing'].mean()
                
                # Create recommendation patterns based on social clusters
                self.enhanced_recommendations['cluster_0'] = {
                    'influence_std_weight': 75,
                    'isolated_std_weight': 80,
                    'prioritize_social_influence': 4
                }
                
                self.enhanced_recommendations['cluster_1'] = {
                    'influence_std_weight': 60,
                    'isolated_std_weight': 50,
                    'prioritize_social_influence': 3
                }
                
                print(f"[INFO] Social analysis: Cluster 0 avg wellbeing {avg_social_0:.1f}, Cluster 1 avg {avg_social_1:.1f}")
            
            # Use integrated dataset if available
            if 'integrated' in self.data_cache:
                # Analyze correlations and patterns in the integrated dataset
                integrated_df = self.data_cache['integrated']
                
                if 'wellbeing_label' in integrated_df.columns and 'social_wellbeing' in integrated_df.columns:
                    # Extract correlation between wellbeing and social wellbeing
                    high_wb_rows = integrated_df[integrated_df['wellbeing_label'] == 'High Wellbeing']
                    low_wb_rows = integrated_df[integrated_df['wellbeing_label'] == 'Low Wellbeing']
                    
                    if not high_wb_rows.empty and 'social_wellbeing' in high_wb_rows.columns:
                        high_wb_social = high_wb_rows['social_wellbeing'].mean()
                        low_wb_social = low_wb_rows['social_wellbeing'].mean() if not low_wb_rows.empty else 0
                        
                        # If high wellbeing correlates with higher social wellbeing
                        if high_wb_social > low_wb_social:
                            self.enhanced_recommendations['wellbeing_social'] = {
                                'wellbeing_penalty_weight': 75,
                                'influence_std_weight': 75,
                                'friend_inclusion_weight': 75,
                                'prioritize_wellbeing': 4,
                                'prioritize_social_influence': 4
                            }
                            
                            print(f"[INFO] Found positive correlation between wellbeing and social wellbeing")
                
                if 'is_bully' in integrated_df.columns and 'social_cluster' in integrated_df.columns:
                    # Analyze if bullies tend to be in a specific social cluster
                    bully_rows = integrated_df[integrated_df['is_bully'] == 1]
                    if not bully_rows.empty and len(bully_rows) > 5:
                        bully_cluster_counts = bully_rows['social_cluster'].value_counts(normalize=True)
                        
                        # If bullies are more likely to be in a specific cluster
                        dominant_cluster = bully_cluster_counts.idxmax()
                        if bully_cluster_counts[dominant_cluster] > 0.6:  # More than 60% in one cluster
                            self.enhanced_recommendations['bully_cluster'] = {
                                'bully_penalty_weight': 85,
                                'influence_std_weight': 80,
                                'prioritize_bullying': 5,
                                'prioritize_social_influence': 4
                            }
                            
                            print(f"[INFO] Found correlation between bullying and social cluster {dominant_cluster}")
            
            print("[INFO] Advanced recommendation system initialized successfully")
            self.has_advanced_recommendations = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize advanced recommendation system: {e}")
            self.has_advanced_recommendations = False

    def get_recommendation_ml(self, user_input):
        """Get recommendation using the enhanced ML model"""
        try:
            if self.is_ml_model_available:
                # Predict the label
                predicted_label = self.ml_model.predict([user_input])[0]
                
                # Convert back to recommendation text
                recommendation = self.data_cache['teacher_comments'].loc[
                    self.data_cache['teacher_comments']["label"] == predicted_label, 
                    "recommendation"
                ].iloc[0]
                
                # Use the enhanced recommendation system if available
                if hasattr(self, 'has_advanced_recommendations') and self.has_advanced_recommendations:
                    recommendation = self._enhance_recommendation(recommendation, user_input)
                
                print(f"[INFO] ML model predicted recommendation for input: {user_input}")
                return recommendation
            else:
                print("[WARN] ML model not available, falling back to similarity search")
                return self.get_similar_comment(user_input)
        except Exception as e:
            print(f"[ERROR] Error in ML recommendation: {e}")
            return self.get_similar_comment(user_input)
            
    def _enhance_recommendation(self, base_recommendation, user_input):
        """Enhance a recommendation with insights from the integrated dataset"""
        try:
            # Analyze the user input to determine which enhanced recommendations to apply
            lower_input = user_input.lower()
            
            # Check for wellbeing-related terms
            if any(term in lower_input for term in ["wellbeing", "stress", "anxious", "mental health"]):
                # Determine if this seems like a high or low wellbeing case
                if any(term in lower_input for term in ["struggling", "anxious", "stressed", "overwhelmed"]):
                    # Apply recommendations for low wellbeing
                    rec_pattern = self.enhanced_recommendations.get('low_wellbeing', {})
                else:
                    # Apply recommendations for high wellbeing
                    rec_pattern = self.enhanced_recommendations.get('high_wellbeing', {})
                    
                # Modify the base recommendation
                for param, value in rec_pattern.items():
                    pattern = fr'({param}).*?(\d+)'
                    if re.search(pattern, base_recommendation):
                        base_recommendation = re.sub(
                            pattern, f"{param} to {value}", base_recommendation
                        )
                    else:
                        base_recommendation += f" Also set {param} to {value} to support student wellbeing."
            
            # Check for bullying-related terms
            if any(term in lower_input for term in ["bully", "bullying", "victim", "harassment"]):
                # Apply recommendations for bullying prevention
                rec_pattern = self.enhanced_recommendations.get('high_bullying', {})
                
                # Modify the base recommendation
                for param, value in rec_pattern.items():
                    pattern = fr'({param}).*?(\d+)'
                    if re.search(pattern, base_recommendation):
                        base_recommendation = re.sub(
                            pattern, f"{param} to {value}", base_recommendation
                        )
                    else:
                        base_recommendation += f" Set {param} to {value} to address bullying concerns."
            
            # Check for academic-related terms
            if any(term in lower_input for term in ["academic", "grade", "gpa", "performance"]):
                # Determine if this is about improving low performance
                if any(term in lower_input for term in ["low", "poor", "failing", "struggling"]):
                    # Apply recommendations for low GPA
                    rec_pattern = self.enhanced_recommendations.get('low_gpa', {})
                else:
                    # Apply recommendations for high GPA maintenance
                    rec_pattern = self.enhanced_recommendations.get('high_gpa', {})
                
                # Modify the base recommendation
                for param, value in rec_pattern.items():
                    pattern = fr'({param}).*?(\d+)'
                    if re.search(pattern, base_recommendation):
                        base_recommendation = re.sub(
                            pattern, f"{param} to {value}", base_recommendation
                        )
                    else:
                        base_recommendation += f" Also set {param} to {value} for academic considerations."
            
            # Check for social-related terms
            if any(term in lower_input for term in ["social", "friends", "isolated", "inclusion"]):
                # Determine which social recommendation to apply
                if any(term in lower_input for term in ["isolated", "alone", "lacks friends", "withdrawn"]):
                    # Apply recommendations for potentially isolated students (cluster 0)
                    rec_pattern = self.enhanced_recommendations.get('cluster_0', {})
                else:
                    # Apply recommendations for socially connected students (cluster 1)
                    rec_pattern = self.enhanced_recommendations.get('cluster_1', {})
                    
                # Modify the base recommendation
                for param, value in rec_pattern.items():
                    pattern = fr'({param}).*?(\d+)'
                    if re.search(pattern, base_recommendation):
                        base_recommendation = re.sub(
                            pattern, f"{param} to {value}", base_recommendation
                        )
                    else:
                        base_recommendation += f" Additionally, set {param} to {value} for better social dynamics."
            
            # Apply combined wellbeing-social recommendations if applicable
            if ('wellbeing_social' in self.enhanced_recommendations and 
                any(term in lower_input for term in ["wellbeing", "social", "friends"]) and
                any(term in lower_input for term in ["balance", "combined", "together", "both"])):
                
                rec_pattern = self.enhanced_recommendations.get('wellbeing_social', {})
                # Add a special note about the correlation
                base_recommendation += " Based on our data analysis, there's a positive correlation between wellbeing and social connectivity."
                
                # Modify the base recommendation
                for param, value in rec_pattern.items():
                    pattern = fr'({param}).*?(\d+)'
                    if re.search(pattern, base_recommendation):
                        base_recommendation = re.sub(
                            pattern, f"{param} to {value}", base_recommendation
                        )
            
            return base_recommendation
        except Exception as e:
            print(f"[ERROR] Error enhancing recommendation: {e}")
            return base_recommendation

    def get_similar_comment(self, user_input):
        try:
            if 'teacher_comments' not in self.data_cache:
                print("[WARN] No teacher comments loaded in data cache.")
                return random.choice(list(self.sample_responses.values()))
            
            comments_df = self.data_cache['teacher_comments']
            if not self.vectorizer:
                print("[INFO] Vectorizer not found. Initializing.")
                self._initialize_vectorizer()
            
            user_vector = self.vectorizer.transform([user_input])
            comment_vectors = self.vectorizer.transform(comments_df["comment"].tolist())
            
            similarities = cosine_similarity(user_vector, comment_vectors)[0]
            max_index = similarities.argmax()
            max_score = similarities[max_index]
            print(f"[DEBUG] Top similarity score: {max_score:.3f} for input: {user_input}")

            if max_score < 0.25:
                print("[DEBUG] Low similarity detected — using random fallback.")
                return random.choice(list(self.sample_responses.values()))

            return comments_df["recommendation"].iloc[max_index]

        except Exception as e:
            print(f"[ERROR] Failed to retrieve similar comment: {e}")
            return random.choice(list(self.sample_responses.values()))
    
    def interpret_natural_language(self, user_input):
        """
        Advanced function to interpret natural language and convert it to configuration parameters
        This implements the mappings suggested in the enhancement request with data-driven insights
        """
        # Initialize with current config
        config = self.get_current_config().copy()
        modified = False
        matched_intents = []
        
        # Lowercase the input for easier matching
        user_input_lower = user_input.lower()
        
        # 1. Direct phrase matching from our mapping schema
        for phrase, params in self.nl_to_param_mapping.items():
            if phrase in user_input_lower:
                matched_intents.append(phrase)
                for param, value in params.items():
                    config[param] = value
                    modified = True
        
        # 2. Intent detection with keywords
        detected_intents = {}
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    detected_intents[intent] = detected_intents.get(intent, 0) + 1
        
        # Sort intents by frequency of keyword occurrence
        sorted_intents = sorted(detected_intents.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Identify modifiers (intensity)
        modifiers = []
        total_modifier = 1.0
        for mod, value in self.intensity_modifiers.items():
            if mod in user_input_lower:
                modifiers.append(mod)
                total_modifier *= value
        
        # 4. Apply detected intents to config with intensity modifiers
        if sorted_intents:
            # Apply modifiers to the primary intents
            for intent, _ in sorted_intents[:2]:  # Focus on top 2 intents
                if intent == "academic":
                    config["gpa_penalty_weight"] = min(100, int(70 * total_modifier))
                    config["prioritize_academic"] = min(5, int(4 * total_modifier))
                    modified = True
                elif intent == "wellbeing":
                    config["wellbeing_penalty_weight"] = min(100, int(70 * total_modifier))
                    config["prioritize_wellbeing"] = min(5, int(4 * total_modifier))
                    modified = True
                elif intent == "bullying":
                    config["bully_penalty_weight"] = min(100, int(80 * total_modifier))
                    config["prioritize_bullying"] = min(5, int(4 * total_modifier))
                    modified = True
                elif intent == "social":
                    config["influence_std_weight"] = min(100, int(70 * total_modifier))
                    config["isolated_std_weight"] = min(100, int(70 * total_modifier))
                    config["prioritize_social_influence"] = min(5, int(4 * total_modifier))
                    modified = True
                elif intent == "friendship":
                    config["friend_inclusion_weight"] = min(100, int(70 * total_modifier))
                    config["friendship_balance_weight"] = min(100, int(60 * total_modifier))
                    if total_modifier > 1.1:  # If "strong" friendship emphasis
                        config["min_friends_required"] = 2
                    config["prioritize_friendship"] = min(5, int(4 * total_modifier))
                    modified = True
        
        # 5. Apply enhanced data-driven recommendations if available
        if hasattr(self, 'has_advanced_recommendations') and self.has_advanced_recommendations:
            config = self._apply_data_driven_recommendations(config, user_input_lower)
            modified = True
        
        # If we couldn't interpret via rules, try ML model if available
        if not modified and self.is_ml_model_available:
            recommendation = self.get_recommendation_ml(user_input)
            # Extract parameter values from the recommendation
            self._extract_config_from_recommendation(recommendation, config)
            modified = True
            
        # If still no modification, use similarity search
        if not modified:
            recommendation = self.get_similar_comment(user_input)
            # Extract parameter values from the recommendation
            self._extract_config_from_recommendation(recommendation, config)
            modified = True
            
        return config, modified
        
    def _apply_data_driven_recommendations(self, config, user_input):
        """Apply data-driven recommendations based on patterns found in the loaded datasets"""
        try:
            # Extract wellbeing signals from the input
            wellbeing_signals = {
                "high": ["thriving", "happy", "content", "balanced", "good mental health"],
                "low": ["stressed", "anxious", "unhappy", "struggling", "overwhelmed"]
            }
            
            # Extract social signals from the input
            social_signals = {
                "high": ["popular", "many friends", "social", "engaged", "connected"],
                "low": ["isolated", "few friends", "lonely", "withdrawn", "disconnected"]
            }
            
            # Extract academic signals from the input
            academic_signals = {
                "high": ["good grades", "high performance", "academic success", "high achiever"],
                "low": ["poor grades", "failing", "academic problems", "low achievement"]
            }
            
            # Extract bullying signals from the input
            bullying_signals = {
                "high": ["bullying problem", "harassment", "victimization", "aggressive behavior"],
                "low": ["no bullying", "safe environment", "positive interactions"]
            }
            
            # Convert input to lowercase for comparison
            user_input = user_input.lower()
            
            # Detect wellbeing signals in user input
            detected_wellbeing = None
            for level, signals in wellbeing_signals.items():
                if any(signal in user_input for signal in signals):
                    detected_wellbeing = level
                    break
                    
            # Detect social signals in user input
            detected_social = None
            for level, signals in social_signals.items():
                if any(signal in user_input for signal in signals):
                    detected_social = level
                    break
                    
            # Detect academic signals in user input
            detected_academic = None
            for level, signals in academic_signals.items():
                if any(signal in user_input for signal in signals):
                    detected_academic = level
                    break
                    
            # Detect bullying signals in user input
            detected_bullying = None
            for level, signals in bullying_signals.items():
                if any(signal in user_input for signal in signals):
                    detected_bullying = level
                    break
            
            # Apply wellbeing patterns if detected
            if detected_wellbeing == "low":
                # For low wellbeing, apply patterns found in low wellbeing cluster data
                if hasattr(self, 'enhanced_recommendations') and 'low_wellbeing' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['low_wellbeing']
                    for param, value in pattern.items():
                        config[param] = value
            elif detected_wellbeing == "high":
                # For high wellbeing, apply patterns found in high wellbeing cluster data
                if hasattr(self, 'enhanced_recommendations') and 'high_wellbeing' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['high_wellbeing']
                    for param, value in pattern.items():
                        config[param] = value
            
            # Apply social patterns if detected
            if detected_social == "low":
                # For low social connectivity, apply patterns from cluster 0
                if hasattr(self, 'enhanced_recommendations') and 'cluster_0' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['cluster_0']
                    for param, value in pattern.items():
                        config[param] = value
            elif detected_social == "high":
                # For high social connectivity, apply patterns from cluster 1
                if hasattr(self, 'enhanced_recommendations') and 'cluster_1' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['cluster_1']
                    for param, value in pattern.items():
                        config[param] = value
            
            # Apply academic patterns if detected
            if detected_academic == "low":
                # For low academic performance, apply patterns from low GPA data
                if hasattr(self, 'enhanced_recommendations') and 'low_gpa' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['low_gpa']
                    for param, value in pattern.items():
                        config[param] = value
            elif detected_academic == "high":
                # For high academic performance, apply patterns from high GPA data
                if hasattr(self, 'enhanced_recommendations') and 'high_gpa' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['high_gpa']
                    for param, value in pattern.items():
                        config[param] = value
            
            # Apply bullying patterns if detected
            if detected_bullying == "high":
                # For high bullying concerns, apply patterns from high bullying data
                if hasattr(self, 'enhanced_recommendations') and 'high_bullying' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['high_bullying']
                    for param, value in pattern.items():
                        config[param] = value
            elif detected_bullying == "low":
                # For low bullying concerns, apply patterns from low bullying data
                if hasattr(self, 'enhanced_recommendations') and 'low_bullying' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['low_bullying']
                    for param, value in pattern.items():
                        config[param] = value
            
            # Apply combined patterns if multiple needs detected
            if detected_wellbeing == "low" and detected_social == "low":
                # Special case for students with both wellbeing and social issues
                if hasattr(self, 'enhanced_recommendations') and 'wellbeing_social' in self.enhanced_recommendations:
                    pattern = self.enhanced_recommendations['wellbeing_social']
                    for param, value in pattern.items():
                        config[param] = value
            
            # Apply bullying and social influence combined patterns if applicable
            if detected_bullying == "high" and 'bully_cluster' in self.enhanced_recommendations:
                pattern = self.enhanced_recommendations['bully_cluster']
                for param, value in pattern.items():
                    config[param] = value
            
            return config
        except Exception as e:
            print(f"[ERROR] Error applying data-driven recommendations: {e}")
            return config

    def get_chat_history(self, session_id=None, limit=10):
        """Get recent chat history, optionally filtered by session_id"""
        if session_id:
            history = [entry for entry in self.conversation_history if entry.get("session_id") == session_id]
        else:
            history = self.conversation_history.copy()
            
        # Return most recent conversations up to the limit
        return history[-limit:] if len(history) > limit else history
    
    def clear_chat_history(self, session_id=None):
        """Clear chat history, optionally only for a specific session"""
        if session_id:
            # Remove entries for the specific session
            self.conversation_history = [entry for entry in self.conversation_history if entry.get("session_id") != session_id]
        else:
            # Clear all conversation history
            self.conversation_history = []
        return True
    
    def get_priority_recommendations(self):
        """Get recommendations for setting priorities on the set-priorities page"""
        try:
            # Return a list of recommendations for setting priorities
            return [
                "Consider prioritizing bullying prevention to foster a safe learning environment.",
                "Balance academic performance with student wellbeing for optimal outcomes.",
                "Social dynamics can significantly impact student learning - consider adjusting influence weights.",
                "For students with social challenges, increase the minimum friends required setting.",
                "If academic achievement is your focus, increase the GPA penalty weight and academic priority."
            ]
        except Exception as e:
            print(f"Error getting priority recommendations: {e}")
            return []
        
    def get_recommendations(self, student_data=None):
        """Get general recommendations for visualizations"""
        try:
            config = self.get_current_config()
            
            # A list of general recommendations
            recommendations = [
                "Consider exploring the social connections graph to understand student relationships.",
                "Review the dashboard for a high-level overview of class dynamics.",
                "The network graphs can provide insights into student interactions."
            ]
            
            # Add student-specific recommendations if student data is provided
            if student_data:
                # Check for student-specific characteristics that might warrant recommendations
                student_id = student_data.get("id")
                if student_id:
                    recommendations.append(f"You're viewing data for student {student_id}. Consider comparing with class averages.")
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def get_current_config(self):
        """Get current constraints configuration from API"""
        if self.api_client:
            try:
                # Try to get config from API
                constraints = self.api_client.get_constraints()
                if constraints:
                    return constraints
            except Exception as e:
                print(f"[ERROR] Error getting config from API: {e}")
        
        # Fallback to file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.default_config
        return self.default_config
    
    def save_config(self, config):
        """Save updated configuration via API"""
        if self.api_client:
            try:
                # Save via API
                result = self.api_client.save_constraints(config)
                if result:
                    print("[INFO] Successfully saved config via API")
                    return config
            except Exception as e:
                print(f"[ERROR] Error saving config to API: {e}")
        
        # Fallback to file
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
                print("[INFO] Saved config to file (API unavailable)")
            return config
        except Exception as e:
            print(f"[ERROR] Error saving config: {e}")
            return config
    
    def analyze_request(self, user_input, session_id=None):
        """Main function to analyze user input and generate a response"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # First check for greetings
            if self.is_greeting(user_input):
                response = "Hello! I'm your classroom optimization assistant. I can help with balancing academic performance, student wellbeing, bullying prevention, social dynamics, and friendship connections. How can I assist you today?"
                conversation_entry = {
                    "timestamp": timestamp,
                    "session_id": session_id if session_id else str(random.randint(1000, 9999)),
                    "user_input": user_input,
                    "response": response,
                    "modified_config": None
                }
                self.conversation_history.append(conversation_entry)
                
                return {
                    "success": True,
                    "message": response,
                    "modified_config": self.get_current_config(),
                    "is_modified": False,
                    "redirect": False,
                    "redirect_url": None
                }
            
            # Use enhanced NLP interpretation
            modified_config, is_modified = self.interpret_natural_language(user_input)
            
            # Generate response explanation
            response = self.generate_response_explanation(modified_config, user_input)
            
            # Log the conversation
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(random.randint(1000, 9999)),
                "user_input": user_input,
                "response": response,
                "modified_config": modified_config if is_modified else None
            }
            self.conversation_history.append(conversation_entry)
            
            return {
                "success": True,
                "message": response,
                "modified_config": modified_config,
                "is_modified": is_modified,
                "redirect": True,
                "redirect_url": "/customisation/set-priorities"
            }
        except Exception as e:
            print(f"Error analyzing request: {e}")
            return {
                "success": False,
                "message": "Sorry, I encountered an error processing your request. Please try again.",
                "modified_config": self.get_current_config(),
                "is_modified": False,
                "redirect": False,
                "redirect_url": None
            }
    
    def generate_response_explanation(self, config, user_input):
        """Generate a human-readable explanation of the configuration changes with data-driven insights"""
        orig_config = self.get_current_config()
        changes = []
        insights = []
        
        # Group parameters by domain
        param_groups = {
            "Academic": ["gpa_penalty_weight", "prioritize_academic"],
            "Wellbeing": ["wellbeing_penalty_weight", "prioritize_wellbeing"],
            "Bullying": ["bully_penalty_weight", "prioritize_bullying"],
            "Social": ["influence_std_weight", "isolated_std_weight", "prioritize_social_influence"],
            "Friendship": ["min_friends_required", "friend_inclusion_weight", "friendship_balance_weight", "prioritize_friendship"]
        }
        
        # Check for changes by group
        for group_name, params in param_groups.items():
            group_changes = []
            for param in params:
                if param in config and param in orig_config and config[param] != orig_config[param]:
                    if "weight" in param:
                        group_changes.append(f"{param.replace('_weight', '').replace('_', ' ')} weight to {config[param]}")
                    elif "prioritize" in param:
                        group_changes.append(f"{param.replace('prioritize_', '')} priority to {config[param]}")
                    else:
                        group_changes.append(f"{param.replace('_', ' ')} to {config[param]}")
            
            if group_changes:
                changes.append(f"{group_name}: " + ", ".join(group_changes))
        
        # Add data-driven insights based on the wellbeing and social data
        user_input_lower = user_input.lower()
        
        # Add wellbeing insights if relevant
        if "wellbeing" in user_input_lower or "mental health" in user_input_lower or "stress" in user_input_lower:
            if hasattr(self, 'data_cache') and 'wellbeing' in self.data_cache:
                wellbeing_df = self.data_cache['wellbeing']
                high_wb_count = (wellbeing_df['wellbeing_label'] == 'High Wellbeing').sum()
                low_wb_count = (wellbeing_df['wellbeing_label'] == 'Low Wellbeing').sum()
                total_count = len(wellbeing_df)
                
                # Add insight about wellbeing distribution
                wb_pct = int((high_wb_count / total_count) * 100)
                insights.append(f"Our analysis shows that {wb_pct}% of students report high wellbeing. Adjusting parameters to support those who need additional help.")
                
                # Add more specific insights based on the data
                if 'School_support_engage6' in wellbeing_df.columns:
                    # Find correlation between school support and wellbeing
                    high_support = wellbeing_df[wellbeing_df['wellbeing_label'] == 'High Wellbeing']['School_support_engage6'].mean()
                    low_support = wellbeing_df[wellbeing_df['wellbeing_label'] == 'Low Wellbeing']['School_support_engage6'].mean()
                    
                    if high_support > low_support:
                        insights.append("Data indicates that higher school support correlates with better wellbeing outcomes.")
        
        # Add social insights if relevant
        if "social" in user_input_lower or "friends" in user_input_lower or "connection" in user_input_lower:
            if hasattr(self, 'data_cache') and 'social' in self.data_cache:
                social_df = self.data_cache['social']
                
                # Calculate average social wellbeing by cluster
                if 'cluster' in social_df.columns and 'social_wellbeing' in social_df.columns:
                    cluster_0_avg = social_df[social_df['cluster'] == 0]['social_wellbeing'].mean()
                    cluster_1_avg = social_df[social_df['cluster'] == 1]['social_wellbeing'].mean()
                    
                    if cluster_1_avg > cluster_0_avg:
                        insights.append(f"Students in the higher social wellbeing cluster score on average {int(cluster_1_avg - cluster_0_avg)} points higher on wellbeing metrics.")
        
        # Generate the response with changes and insights
        if changes:
            response = f"Based on your input, I recommend the following settings: {'. '.join(changes)}."
        else:
            response = "I've analyzed your request but don't recommend any specific changes to the current settings."
            
        # Add insights if available
        if insights:
            response += f" {' '.join(insights)}"
            
        return response
    
    def is_greeting(self, text):
        """Check if the text is a greeting"""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in greetings)

    def _extract_parameter_patterns(self):
        """Extract parameter patterns from teacher comments to learn parameter values"""
        if 'teacher_comments' not in self.data_cache:
            print("[WARN] No teacher comments data available for parameter pattern extraction")
            return
            
        df = self.data_cache['teacher_comments']
        self.param_patterns = {}
        
        # Commonly used parameters in recommendations
        parameters = [
            'gpa_penalty_weight', 'wellbeing_penalty_weight', 'bully_penalty_weight',
            'influence_std_weight', 'isolated_std_weight', 'min_friends_required',
            'friend_inclusion_weight', 'friendship_balance_weight'
        ]
        
        # Extract parameter values from recommendations
        for param in parameters:
            pattern = rf'{param}.*?(\d+)'
            self.param_patterns[param] = []
            
            for rec in df['recommendation']:
                matches = re.findall(pattern, rec)
                if matches:
                    try:
                        self.param_patterns[param].append(int(matches[0]))
                    except:
                        pass
        
        # Calculate typical values for each parameter (median)
        self.param_typical_values = {}
        for param, values in self.param_patterns.items():
            if values:
                self.param_typical_values[param] = int(np.median(values))
            else:
                self.param_typical_values[param] = self.default_config.get(param, 50)
                
        print(f"[INFO] Extracted parameter patterns from {len(df)} teacher comments")

# Create an instance for import in other modules
chatbot = RuleBasedChatbot()

# For backwards compatibility with existing code
class AssistantModel:
    def __init__(self):
        self.chatbot = RuleBasedChatbot()
        self.initialize_model()
    
    def initialize_model(self):
        # Any additional initialization can go here
        pass
    
    def analyze_request(self, user_input, session_id=None):
        return self.chatbot.analyze_request(user_input, session_id)
    
    def get_current_config(self):
        return self.chatbot.get_current_config()
    
    def save_config(self, config):
        return self.chatbot.save_config(config)
    
    def get_chat_history(self, session_id=None, limit=10):
        return self.chatbot.get_chat_history(session_id, limit)
    
    def clear_chat_history(self, session_id=None):
        return self.chatbot.clear_chat_history(session_id)
    
    def reset_chat_history(self, session_id=None):
        """Reset the chat history for a specific session"""
        return self.chatbot.clear_chat_history(session_id)
    
    def get_priority_recommendations(self):
        return self.chatbot.get_priority_recommendations()
    
    def get_recommendations(self, student_data=None):
        return self.chatbot.get_recommendations(student_data)
