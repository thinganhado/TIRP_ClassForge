import json
import os
import random
import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Try to import transformers for BERT models
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel
    try:
        from sentence_transformers import SentenceTransformer, util
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        print("Sentence transformers loaded successfully")
    except Exception as e:
        print(f"Error loading sentence-transformers: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

    TRANSFORMERS_AVAILABLE = True
    print("Transformers library loaded successfully")
except Exception as e:
    print(f"Error loading transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

# Try to import network analysis libraries
NETWORK_ANALYSIS_AVAILABLE = False
try:
    import networkx as nx
    NETWORK_ANALYSIS_AVAILABLE = True
    print("NetworkX loaded successfully for social network analysis")
except ImportError:
    print("NetworkX not available for network analysis")

# Constants
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")

class AssistantModel:
    def __init__(self):
        self.config_file = "soft_constraints_config.json"
        self.default_config = {
            "gpa_penalty_weight": 30,
            "wellbeing_penalty_weight": 50,
            "bully_penalty_weight": 60,
            "influence_std_weight": 60, 
            "isolated_std_weight": 60,
            "min_friends_required": 1,
            "friend_inclusion_weight": 50,
            "friendship_balance_weight": 60,
            "prioritize_academic": 5,
            "prioritize_wellbeing": 4,
            "prioritize_bullying": 3,
            "prioritize_social_influence": 2,
            "prioritize_friendship": 1
        }
        
        # Training data for the model
        self.teacher_comments_file = "app/models/teacher_comments.csv"
        
        # Sample NLP model responses
        self.sample_responses = {
            "academic_priority": "I recommend prioritizing academic balance with a higher GPA penalty weight (60-70) and setting academic as the top priority (5).",
            "wellbeing_priority": "For student wellbeing focus, I suggest increasing the wellbeing_penalty_weight to 70-80 and setting it as the top priority (5).",
            "bullying_prevention": "To address bullying concerns, set bully_penalty_weight to 80-90 and make bullying prevention the top priority (5).",
            "social_balance": "For better social dynamics, increase influence_std_weight and isolated_std_weight to 70, and prioritize social influence (4-5).",
            "friendship_focus": "To optimize friendship connections, set min_friends_required to 2 and increase friend_inclusion_weight to 70-80."
        }
        
        # Categories and their related terms
        self.category_terms = {
            "academic": ["academic", "grades", "gpa", "performance", "achievement", "study", "learning", "smart", "gpa balance", "gpa penalty", "academic performance", "grade variance", "grade distribution"],
            "wellbeing": ["wellbeing", "well-being", "mental health", "support", "emotional", "stress", "anxiety", "happy", "happiness", "student wellbeing", "wellbeing balance", "wellbeing penalty"],
            "bullying": ["bully", "bullying", "harassment", "victim", "protect", "safety", "secure", "cruel", "mean", "bully-victim", "bullying prevention", "vulnerability", "bully penalty", "separation"],
            "social": ["social", "influence", "isolated", "popular", "interaction", "group", "clique", "leaders", "influence balance", "isolation balance", "social influence", "influential"],
            "friendship": ["friend", "friendship", "relationship", "peer", "buddy", "companion", "pals", "together", "friend inclusion", "friendship balance", "social connections", "minimum friends"]
        }
        
        # Priority words
        self.increase_priority = ["prioritize", "focus on", "increase", "more important", "higher", "enhance", "boost", "maximize"]
        self.decrease_priority = ["less", "decrease", "lower", "reduce", "minimize", "less important", "downplay"]
        
        # Conversation memory
        self.conversation_history = []
        self.chat_log_file = "app/static/data/chat_history.json"
        
        # Load existing chat history if available
        self._load_chat_history()
        
        # Initialize model
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentence_transformer = None
        self.data_cache = {}  # Cache for loaded data
        self.initialize_model()
        
    def _load_chat_history(self):
        """Load existing chat history from file"""
        try:
            if os.path.exists(self.chat_log_file):
                with open(self.chat_log_file, 'r') as f:
                    self.conversation_history = json.load(f)
                print(f"Loaded {len(self.conversation_history)} previous conversations")
        except Exception as e:
            print(f"Error loading chat history: {e}")
            self.conversation_history = []
            
    def _save_chat_history(self):
        """Save chat history to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.chat_log_file), exist_ok=True)
            
            with open(self.chat_log_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
        
    def initialize_model(self):
        """Initialize BERT models for text processing"""
        # Initialize BERT models
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentence_transformer = None
        
        try:
            # Load BERT model for intent recognition
            print("Initializing BERT models...")
            
            # Check if fine-tuned model exists and try to load it
            fine_tuned_dir = "app/models/fine_tuned_bert"
            if os.path.exists(fine_tuned_dir):
                try:
                    print("Loading fine-tuned BERT model")
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_dir)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_dir)
                    
                    # Load label mapping
                    label_mapping_path = os.path.join(fine_tuned_dir, "label_mapping.json")
                    if os.path.exists(label_mapping_path):
                        with open(label_mapping_path, "r") as f:
                            label_mapping = json.load(f)
                            print(f"Loaded fine-tuned model with {len(label_mapping['id2label'])} categories")
                    
                    print("Fine-tuned BERT model loaded successfully")
                except Exception as e:
                    print(f"Error loading fine-tuned model: {e}")
                    self.bert_model = None
                    self.bert_tokenizer = None
            
            # Initialize sentence transformer for semantic search
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                    print("Loaded Sentence Transformer model")
                except Exception as e:
                    print(f"Error loading Sentence Transformer: {e}")
                    self.sentence_transformer = None
            
            # If fine-tuned model not loaded, load base model
            if self.bert_model is None:
                try:
                    # Load smaller BERT model for classification
                    model_name = "distilbert-base-uncased"
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=len(self.category_terms),
                        id2label={i: k for i, k in enumerate(self.category_terms.keys())},
                        label2id={k: i for i, k in enumerate(self.category_terms.keys())}
                    )
                    print("Loaded DistilBERT model for classification")
                except Exception as e:
                    print(f"Error loading DistilBERT model: {e}")
                    self.bert_model = None
                    self.bert_tokenizer = None
                    
            # Load CSV data for better responses
            self._load_csv_data()
                
        except Exception as e:
            print(f"Error initializing BERT models: {e}")
            self.bert_model = None
            self.bert_tokenizer = None
            self.sentence_transformer = None
            
    def _load_csv_data(self):
        """Load CSV data for context-aware responses"""
        try:
            # Load teacher comments if available
            if os.path.exists(self.teacher_comments_file):
                self.data_cache['teacher_comments'] = pd.read_csv(self.teacher_comments_file)
                print(f"Loaded {len(self.data_cache['teacher_comments'])} teacher comments")
            else:
                # Generate and save sample data if needed
                csv_data = self.generate_csv_data()
                os.makedirs(os.path.dirname(self.teacher_comments_file), exist_ok=True)
                with open(self.teacher_comments_file, 'w') as f:
                    f.write(csv_data)
                self.data_cache['teacher_comments'] = pd.read_csv(self.teacher_comments_file)
                print(f"Generated and loaded {len(self.data_cache['teacher_comments'])} teacher comments")
                
            # Try to load student data if available (for future use)
            try:
                student_data_file = "app/static/data/student_summary.csv"
                if os.path.exists(student_data_file):
                    self.data_cache['students'] = pd.read_csv(student_data_file)
                    print(f"Loaded {len(self.data_cache['students'])} student records")
            except Exception as e:
                print(f"Note: Student data not available: {e}")
                
        except Exception as e:
            print(f"Error loading CSV data: {e}")
    
    def get_current_config(self):
        """Get current constraints configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return json.load(f)
        return self.default_config
    
    def save_config(self, config):
        """Save updated configuration"""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
        return config
    
    def analyze_request_with_bert(self, user_input):
        """Use BERT model for text analysis"""
        try:
            if self.bert_model is not None and self.bert_tokenizer is not None:
                # Tokenize the input
                inputs = self.bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    
                # Convert to probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Map to categories
                category_scores = {}
                for i, category in enumerate(self.category_terms.keys()):
                    category_scores[category] = float(probabilities[i])
                
                # Adjust scores based on priority keywords
                for category, terms in self.category_terms.items():
                    # Check for explicit priority words
                    for increase in self.increase_priority:
                        if any(increase + " " + term in user_input.lower() for term in terms):
                            category_scores[category] += 0.3
                            
                    for decrease in self.decrease_priority:
                        if any(decrease + " " + term in user_input.lower() for term in terms):
                            category_scores[category] -= 0.3
                
                return category_scores
            
            # If BERT model is not available, use fallback
            return self.analyze_request_rule_based(user_input)
        except Exception as e:
            print(f"Error using BERT model: {e}")
            return self.analyze_request_rule_based(user_input)
    
    def analyze_request_rule_based(self, user_input):
        """Fall back to rule-based analysis"""
        user_input = user_input.lower()
        scores = {}
        
        # Count matches for each category
        for category, terms in self.category_terms.items():
            scores[category] = sum(term in user_input for term in terms)
            
            # Adjust scores based on priority keywords
            for increase in self.increase_priority:
                if any(increase + " " + term in user_input for term in terms):
                    scores[category] += 3
                    
            for decrease in self.decrease_priority:
                if any(decrease + " " + term in user_input for term in terms):
                    scores[category] -= 3
                    
        return scores
    
    def get_similar_comment(self, user_input):
        """Find similar teacher comment from the dataset using BERT/Sentence Transformers"""
        try:
            if 'teacher_comments' in self.data_cache:
                comments = self.data_cache['teacher_comments']['teacher_comment'].tolist()
                
                # Normalize hyphenated terms and handle common variations
                user_input_normalized = user_input.lower()
                user_input_normalized = user_input_normalized.replace("well-being", "wellbeing")
                user_input_normalized = user_input_normalized.replace("well being", "wellbeing")
                user_input_normalized = user_input_normalized.replace("gpa balance", "academic")
                user_input_normalized = user_input_normalized.replace("academic performance", "academic")
                user_input_normalized = user_input_normalized.replace("influence balance", "social influence")
                user_input_normalized = user_input_normalized.replace("friend inclusion", "friendship")
                user_input_normalized = user_input_normalized.replace("friendship balance", "friendship")
                user_input_normalized = user_input_normalized.replace("social connections", "friendship")
                user_input_normalized = user_input_normalized.replace("balaced", "balanced") # Fix common typo
                user_input_normalized = user_input_normalized.replace("friend ship", "friendship") # Fix common split
                user_input_normalized = user_input_normalized.replace("freindship", "friendship") # Fix common typo
                user_input_normalized = user_input_normalized.replace("disrepect", "bullying") # Map disrespect to bullying
                user_input_normalized = user_input_normalized.replace("disrespect", "bullying") # Map disrespect to bullying
                
                # Special case handling for specific scenarios
                user_input_lower = user_input_normalized
                
                # Friendship case
                if ('balanced friendship' in user_input_lower or 'friendship group' in user_input_lower or 
                    'friend group' in user_input_lower or 'friends' in user_input_lower):
                    for i, comment in enumerate(comments):
                        comment_lower = comment.lower()
                        if 'friend' in comment_lower or 'social connection' in comment_lower:
                            return self.data_cache['teacher_comments'].iloc[i]
                
                # Disrespect/bullying case
                if ('disrespect' in user_input_lower or 'avoid bullying' in user_input_lower or 
                    'prevent bullying' in user_input_lower or 'bullying balanced' in user_input_lower):
                    for i, comment in enumerate(comments):
                        comment_lower = comment.lower()
                        if 'bully' in comment_lower or 'bullying' in comment_lower:
                            return self.data_cache['teacher_comments'].iloc[i]
                
                # Wellbeing case
                if 'wellbeing' in user_input_lower or 'balanced' in user_input_lower or 'student wellbeing' in user_input_lower:
                    for i, comment in enumerate(comments):
                        comment_lower = comment.lower()
                        if 'wellbeing' in comment_lower or 'balanced' in comment_lower:
                            return self.data_cache['teacher_comments'].iloc[i]
                
                # Immigrant case
                if 'immigrant' in user_input_lower or 'integration' in user_input_lower:
                    for i, comment in enumerate(comments):
                        if 'immigrant' in comment.lower() or 'transfer' in comment.lower() or 'integration' in comment.lower():
                            return self.data_cache['teacher_comments'].iloc[i]
                
                # First try exact keyword matching
                keyword_matches = []
                
                # Define important keywords to match
                important_keywords = [
                    'adhd', 'anxiety', 'math', 'language', 'immigrant', 'bully', 'bullying',
                    'social', 'verbal', 'creative', 'leadership', 'gifted', 'trauma',
                    'behavior', 'wellbeing', 'academic', 'friendship', 'integration',
                    'support', 'structure', 'organization', 'challenge', 'special', 'learning',
                    'balanced', 'balance', 'wellness', 'health', 'friends', 'disrespect',
                    # Added keywords from set_priorities page
                    'gpa', 'grade', 'performance', 'optimization', 'objective', 
                    'bully-victim', 'separation', 'prevention', 'influence', 'isolation', 
                    'connection', 'inclusion', 'priority', 'minimum', 'class', 'balance', 
                    'weight', 'social influence', 'distribution', 'friend inclusion',
                    'friendship balance', 'vulnerability', 'variance', 'soft constraint',
                    'hard constraint', 'maximum'
                ]
                
                # Find comments with matching keywords
                for i, comment in enumerate(comments):
                    comment_lower = comment.lower()
                    # Count matching keywords
                    matching_keywords = sum(1 for keyword in important_keywords 
                                          if keyword in user_input_lower and keyword in comment_lower)
                    if matching_keywords > 0:
                        keyword_matches.append((i, matching_keywords))
                
                # If we have keyword matches, use the best one
                if keyword_matches:
                    # Sort by number of matching keywords (descending)
                    keyword_matches.sort(key=lambda x: x[1], reverse=True)
                    best_match_idx = keyword_matches[0][0]
                    return self.data_cache['teacher_comments'].iloc[best_match_idx]
                
                # Use sentence transformer for semantic search if available
                if self.sentence_transformer is not None:
                    try:
                        # Encode the query and all comments
                        query_embedding = self.sentence_transformer.encode(user_input, convert_to_tensor=True)
                        comment_embeddings = self.sentence_transformer.encode(comments, convert_to_tensor=True)
                        
                        # Calculate similarity
                        similarities = util.pytorch_cos_sim(query_embedding, comment_embeddings)[0]
                        
                        # Get the most similar comment
                        best_idx = int(torch.argmax(similarities))
                        if similarities[best_idx] > 0.3:  # Set similarity threshold
                            return self.data_cache['teacher_comments'].iloc[best_idx]
                    except Exception as e:
                        print(f"Error using sentence transformer: {e}")
                
                # If no good match found with sentence transformer, use BERT if available
                if self.bert_model is not None and self.bert_tokenizer is not None:
                    try:
                        # Tokenize the input and comments with truncation and padding
                        query_encoding = self.bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                        
                        # Get query embedding
                        with torch.no_grad():
                            query_outputs = self.bert_model(**query_encoding, output_hidden_states=True)
                            # Use the last hidden state of [CLS] token as embedding
                            query_embedding = query_outputs.hidden_states[-1][:, 0, :]
                        
                        # Find most similar comment
                        best_sim = -1
                        best_idx = -1
                        
                        # Process comments in batches to avoid memory issues
                        batch_size = 16
                        for i in range(0, len(comments), batch_size):
                            batch_comments = comments[i:i+batch_size]
                            
                            # Tokenize batch
                            batch_encodings = self.bert_tokenizer(batch_comments, return_tensors="pt", 
                                                              truncation=True, padding=True)
                            
                            # Get embeddings
                            with torch.no_grad():
                                batch_outputs = self.bert_model(**batch_encodings, output_hidden_states=True)
                                # Use the last hidden state of [CLS] token
                                batch_embeddings = batch_outputs.hidden_states[-1][:, 0, :]
                            
                            # Calculate similarities
                            for j, embedding in enumerate(batch_embeddings):
                                # Normalize embeddings
                                norm_query = query_embedding / query_embedding.norm()
                                norm_comment = embedding / embedding.norm()
                                
                                # Calculate cosine similarity
                                similarity = torch.dot(norm_query[0], norm_comment)
                                similarity_val = similarity.item()
                                
                                if similarity_val > best_sim:
                                    best_sim = similarity_val
                                    best_idx = i + j
                        
                        if best_sim > 0.3 and best_idx >= 0:  # Set similarity threshold
                            return self.data_cache['teacher_comments'].iloc[best_idx]
                    except Exception as e:
                        print(f"Error using BERT for similarity: {e}")
            
            return None
        except Exception as e:
            print(f"Error finding similar comment: {e}")
            return None
    
    def generate_response(self, user_input, modified_config, is_modified):
        """Generate a conversational response based on the user input and analysis"""
        # Check for special commands
        if "help" in user_input.lower():
            return ("I'm ClassForge's AI assistant. I can help you optimize class allocation by adjusting weights "
                    "and priorities. Try asking me to focus on specific aspects like academic performance, "
                    "wellbeing, bullying prevention, or social dynamics.")
        
        # Get similar teacher comments
        similar_comment = self.get_similar_comment(user_input)
        
        # Check if there's a configuration change
        if is_modified:
            # Create explanation of changes
            explanations = []
            current_config = self.get_current_config()
            
            for key, value in modified_config.items():
                if key in current_config and value != current_config[key]:
                    if "weight" in key:
                        explanations.append(f"{key.replace('_', ' ').title()}: {current_config[key]} → {value}")
                    elif "prioritize" in key:
                        priority_name = key.replace('prioritize_', '').title()
                        explanations.append(f"{priority_name} Priority: {current_config[key]} → {value}")
            
            changes_text = "\n".join(explanations)
            
            base_response = f"I've analyzed your request and recommend the following changes:\n\n{changes_text}\n\n"
            
            # Add recommendation from similar teacher comment if available
            if similar_comment is not None:
                recommendation = similar_comment['recommendation']
                base_response += f"\nBased on teacher feedback for similar situations: '{recommendation}'\n"
                
            base_response += f"\nWould you like me to apply these changes to your class allocation settings?"
        else:
            # No changes to config
            if similar_comment is not None:
                base_response = f"I don't see specific customization needs in your request, but based on similar teacher feedback: '{similar_comment['recommendation']}'"
            else:
                base_response = "I don't see any specific customization needs in your request. Would you like to focus on a particular aspect like academic performance, wellbeing, bullying prevention, or social dynamics?"
        
        return base_response
    
    def analyze_request(self, user_input, session_id=None):
        """Analyze user request to determine what constraints to modify"""
        # First, detect conversation intents
        if self._is_simple_greeting(user_input):
            greeting_response = "Hello! I'm ClassForge's AI assistant. I can help you optimize class allocation settings. Try asking about academic performance, wellbeing, or bullying prevention."
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": greeting_response,
                "modified_config": None
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": greeting_response,
                "modified_config": self.get_current_config(),
                "is_modified": False
            }
            
        # Check if we have a BERT classification with low confidence
        if self.bert_model and self.bert_tokenizer:
            confidence = self._get_bert_confidence(user_input)
            if confidence < 0.4 and len(user_input.split()) < 5:
                # Low confidence for short inputs that aren't greetings - likely random input
                general_response = "I can help you optimize class allocation settings. Try asking about academic performance, wellbeing, bullying prevention, social dynamics, or friendship connections."
                
                # Log conversation
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conversation_entry = {
                    "timestamp": timestamp,
                    "session_id": session_id if session_id else str(int(time.time())),
                    "user_input": user_input,
                    "response": general_response,
                    "modified_config": None
                }
                
                self.conversation_history.append(conversation_entry)
                self._save_chat_history()
                
                return {
                    "success": True,
                    "message": general_response,
                    "modified_config": self.get_current_config(),
                    "is_modified": False
                }
        
        # Check for social network analysis keywords
        sna_keywords = ["network analysis", "social network", "community detection", 
                        "isolated students", "friendship groups", "influential students",
                        "centrality", "community structure", "network density"]
        
        is_sna_request = any(keyword in user_input.lower() for keyword in sna_keywords)
        
        if is_sna_request:
            # Handle social network analysis request
            sna_response = self._generate_sna_response(user_input)
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": sna_response,
                "modified_config": None
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": sna_response,
                "modified_config": self.get_current_config(),
                "is_modified": False,
                "is_sna_request": True
            }
            
        # Check for special cases first
        user_input_lower = user_input.lower()
        
        # Normalize common variations and typos
        user_input_lower = user_input_lower.replace("disrepect", "bullying")
        user_input_lower = user_input_lower.replace("disrespect", "bullying")
        user_input_lower = user_input_lower.replace("balaced", "balanced")
        user_input_lower = user_input_lower.replace("friend ship", "friendship")
        
        config = self.get_current_config()
        modified_config = config.copy()
        explanation = []
        is_modified = False
        
        # Friendship balance/optimization case
        if ('balanced friendship' in user_input_lower or 'friendship group' in user_input_lower or 
            'friend group' in user_input_lower or 'create balanced friendship' in user_input_lower):
            # Focus on friendship settings
            modified_config["friend_inclusion_weight"] = 75
            modified_config["friendship_balance_weight"] = 70
            modified_config["min_friends_required"] = 2
            modified_config["prioritize_friendship"] = 4
            
            explanation.append("Increased friendship inclusion weight for better social grouping")
            explanation.append("Set friendship balance weight higher for more equal distribution")
            explanation.append("Set minimum of 2 friends for each student")
            
            is_modified = True
            
            # Generate response and return early
            response = self.generate_response(user_input, modified_config, is_modified)
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": response,
                "modified_config": modified_config
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": response,
                "modified_config": modified_config,
                "is_modified": is_modified,
                "explanation": explanation
            }
        
        # Immigrant/integration case
        if 'immigrant' in user_input_lower or 'integration' in user_input_lower:
            # For immigrant students, focus on wellbeing and friendship
            modified_config["wellbeing_penalty_weight"] = 75
            modified_config["friend_inclusion_weight"] = 80
            modified_config["min_friends_required"] = 2
            modified_config["prioritize_wellbeing"] = 5
            modified_config["prioritize_friendship"] = 4
            
            explanation.append("Increased wellbeing priority for immigrant student integration")
            explanation.append("Set higher friendship inclusion weight for better integration")
            explanation.append("Ensured minimum of 2 friends for social support")
            
            is_modified = True
            
            # Generate response and return early
            response = self.generate_response(user_input, modified_config, is_modified)
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": response,
                "modified_config": modified_config
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": response,
                "modified_config": modified_config,
                "is_modified": is_modified,
                "explanation": explanation
            }
            
        # ADHD case
        elif 'adhd' in user_input_lower:
            # For ADHD students needing structure
            modified_config["wellbeing_penalty_weight"] = 75
            modified_config["influence_std_weight"] = 65
            modified_config["prioritize_wellbeing"] = 5
            
            explanation.append("Increased wellbeing priority for ADHD student support")
            explanation.append("Adjusted influence standard weight to manage classroom dynamics")
            
            is_modified = True
            
            # Generate response and return early
            response = self.generate_response(user_input, modified_config, is_modified)
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": response,
                "modified_config": modified_config
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": response,
                "modified_config": modified_config,
                "is_modified": is_modified,
                "explanation": explanation
            }
                
        # Bullying/disrespect case
        elif 'bully' in user_input_lower or 'bullying' in user_input_lower or 'disrespect' in user_input_lower:
            # For bullying prevention
            modified_config["bully_penalty_weight"] = 90
            modified_config["wellbeing_penalty_weight"] = 80
            modified_config["prioritize_bullying"] = 5
            modified_config["prioritize_wellbeing"] = 4
            
            explanation.append("Increased bully penalty weight for stronger prevention")
            explanation.append("Raised wellbeing penalty weight to support vulnerable students")
            explanation.append("Set bullying prevention as top priority")
            
            is_modified = True
            
            # Generate response and return early
            response = self.generate_response(user_input, modified_config, is_modified)
            
            # Log conversation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(int(time.time())),
                "user_input": user_input,
                "response": response,
                "modified_config": modified_config
            }
            
            self.conversation_history.append(conversation_entry)
            self._save_chat_history()
            
            return {
                "success": True,
                "message": response,
                "modified_config": modified_config,
                "is_modified": is_modified,
                "explanation": explanation
            }
        
        # Try to find a similar teacher comment first
        similar_comment = self.get_similar_comment(user_input)
        has_similar_comment = similar_comment is not None
            
        # Use BERT model for analysis
        scores = self.analyze_request_with_bert(user_input)
            
        # Generate response based on analysis
        modified_config = config.copy()
        explanation = []
        
        # Incorporate similar teacher comment recommendation if available
        if has_similar_comment:
            recommendation = similar_comment['recommendation'].lower()
            
            # Extract specific weight values from recommendation
            if 'gpa penalty weight' in recommendation and ('60-70' in recommendation or '65-75' in recommendation or '75-85' in recommendation):
                if '75-85' in recommendation:
                    modified_config["gpa_penalty_weight"] = 80
                elif '65-75' in recommendation:
                    modified_config["gpa_penalty_weight"] = 70
                else:
                    modified_config["gpa_penalty_weight"] = 65
                explanation.append(f"Set GPA penalty weight to {modified_config['gpa_penalty_weight']} based on teacher recommendation")
                
            if 'wellbeing_penalty_weight' in recommendation and ('70-80' in recommendation or '75-85' in recommendation):
                if '75-85' in recommendation:
                    modified_config["wellbeing_penalty_weight"] = 80
                else:
                    modified_config["wellbeing_penalty_weight"] = 75
                explanation.append(f"Set wellbeing penalty weight to {modified_config['wellbeing_penalty_weight']} based on teacher recommendation")
                
            if 'bully_penalty_weight' in recommendation and ('80-90' in recommendation or '85-95' in recommendation):
                if '85-95' in recommendation:
                    modified_config["bully_penalty_weight"] = 90
                else:
                    modified_config["bully_penalty_weight"] = 85
                explanation.append(f"Set bully penalty weight to {modified_config['bully_penalty_weight']} based on teacher recommendation")
                
            if 'influence_std_weight' in recommendation and ('70' in recommendation or '65' in recommendation or '75' in recommendation):
                if '75' in recommendation:
                    modified_config["influence_std_weight"] = 75
                elif '70' in recommendation:
                    modified_config["influence_std_weight"] = 70
                else:
                    modified_config["influence_std_weight"] = 65
                explanation.append(f"Set influence standard weight to {modified_config['influence_std_weight']} based on teacher recommendation")
                
            if 'friend_inclusion_weight' in recommendation and ('70-80' in recommendation or '75-85' in recommendation):
                if '75-85' in recommendation:
                    modified_config["friend_inclusion_weight"] = 80
                else:
                    modified_config["friend_inclusion_weight"] = 75
                explanation.append(f"Set friend inclusion weight to {modified_config['friend_inclusion_weight']} based on teacher recommendation")
                
            if 'min_friends_required' in recommendation and '2' in recommendation:
                modified_config["min_friends_required"] = 2
                explanation.append(f"Set minimum friends required to 2 based on teacher recommendation")
        
        # Sort areas by score to identify priorities
        priority_areas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply modifications based on identified priorities
        for area, score in priority_areas:
            if score > 0.2:  # Only consider scores above a threshold
                if area == "academic":
                    if "gpa_penalty_weight" not in explanation:  # Only set if not already set from recommendation
                        if "gpa_penalty_weight" in config:
                            modified_config["gpa_penalty_weight"] = min(100, int(config["gpa_penalty_weight"] + score * 30))
                        else:
                            modified_config["gpa_penalty_weight"] = min(100, int(30 + score * 30))  # Use default value of 30
                    modified_config["prioritize_academic"] = 5
                    explanation.append(f"Increased academic priority to {modified_config['prioritize_academic']}")
                    
                elif area == "wellbeing":
                    if "wellbeing_penalty_weight" in config:
                        modified_config["wellbeing_penalty_weight"] = min(100, int(config["wellbeing_penalty_weight"] + score * 30))
                    else:
                        modified_config["wellbeing_penalty_weight"] = min(100, int(50 + score * 30))  # Use default value of 50
                    modified_config["prioritize_wellbeing"] = 5
                    explanation.append(f"Increased wellbeing priority to {modified_config['prioritize_wellbeing']}")
                    
                elif area == "bullying":
                    if "bully_penalty_weight" in config:
                        modified_config["bully_penalty_weight"] = min(100, int(config["bully_penalty_weight"] + score * 30))
                    else:
                        modified_config["bully_penalty_weight"] = min(100, int(60 + score * 30))  # Use default value of 60
                    modified_config["prioritize_bullying"] = 5
                    explanation.append(f"Increased bullying prevention priority to {modified_config['prioritize_bullying']}")
                    
                elif area == "social":
                    if "influence_std_weight" in config:
                        modified_config["influence_std_weight"] = min(100, int(config["influence_std_weight"] + score * 30))
                    else:
                        modified_config["influence_std_weight"] = min(100, int(60 + score * 30))  # Use default value of 60
                        
                    if "isolated_std_weight" in config:
                        modified_config["isolated_std_weight"] = min(100, int(config["isolated_std_weight"] + score * 30))
                    else:
                        modified_config["isolated_std_weight"] = min(100, int(60 + score * 30))  # Use default value of 60
                        
                    modified_config["prioritize_social_influence"] = 4
                    explanation.append(f"Increased social influence priority to {modified_config['prioritize_social_influence']}")
                    
                elif area == "friendship":
                    if "friend_inclusion_weight" in config:
                        modified_config["friend_inclusion_weight"] = min(100, int(config["friend_inclusion_weight"] + score * 30))
                    else:
                        modified_config["friend_inclusion_weight"] = min(100, int(50 + score * 30))  # Use default value of 50
                        
                    if "friendship_balance_weight" in config:
                        modified_config["friendship_balance_weight"] = min(100, int(config["friendship_balance_weight"] + score * 30))
                    else:
                        modified_config["friendship_balance_weight"] = min(100, int(60 + score * 30))  # Use default value of 60
                        
                    if "increase friends" in user_input.lower() or "more friends" in user_input.lower():
                        modified_config["min_friends_required"] = min(3, config.get("min_friends_required", 1) + 1)
                        explanation.append(f"Increased minimum friends required to {modified_config['min_friends_required']}")
                    modified_config["prioritize_friendship"] = 4
                    explanation.append(f"Increased friendship priority to {modified_config['prioritize_friendship']}")
        
        # Check for "impossible" scenarios
        impossible_request = False
        recommendation = ""
        
        if "100 friends" in user_input.lower() or "50 friends" in user_input.lower():
            impossible_request = True
            recommendation = "Having 50+ friends in each class isn't feasible. I recommend setting minimum friends to 2-3 instead."
        
        is_modified = modified_config != config
        
        # Log conversation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_entry = {
            "timestamp": timestamp,
            "session_id": session_id if session_id else str(int(time.time())),
            "user_input": user_input,
            "response": self.generate_response(user_input, modified_config, is_modified),
            "modified_config": modified_config if is_modified else None
        }
        
        self.conversation_history.append(conversation_entry)
        self._save_chat_history()
        
        if modified_config == config:
            return {
                "success": True,
                "message": self.generate_response(user_input, modified_config, False),
                "modified_config": config,
                "is_modified": False
            }
        
        if impossible_request:
            return {
                "success": False,
                "message": recommendation,
                "modified_config": config,
                "is_modified": False
            }
        
        return {
            "success": True,
            "message": self.generate_response(user_input, modified_config, True),
            "modified_config": modified_config,
            "is_modified": True
        }
    
    def _is_simple_greeting(self, text):
        """Check if text is a simple greeting"""
        text = text.lower().strip()
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
        return text in greetings or text.startswith(tuple(g + ' ' for g in greetings))
    
    def _get_bert_confidence(self, text):
        """Get confidence score from BERT model"""
        try:
            if self.bert_model and self.bert_tokenizer:
                # Tokenize
                inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Return highest probability as confidence
                return torch.max(probabilities).item()
            return 0.0
        except Exception as e:
            print(f"Error getting BERT confidence: {e}")
            return 0.0
    
    def _generate_sna_response(self, user_input):
        """Generate a response for social network analysis queries"""
        user_input_lower = user_input.lower()
        
        # Define response categories
        responses = {
            "general_sna": "I can help analyze social networks in your class allocation. You can upload relationship data through our API to get insights on community structures, influential students, and isolated individuals.",
            
            "communities": "Community detection in social networks can help identify natural friendship groups. Using algorithms like Louvain or modularity optimization, we can detect clusters of students with dense connections.",
            
            "isolation": "Identifying isolated students is important for wellbeing. Students with few or no connections may need special attention in class allocation to ensure they're placed with compatible peers.",
            
            "influential": "Influential students (those with high centrality) act as social hubs. Distributing these students evenly across classes can help maintain balanced social dynamics.",
            
            "betweenness": "Students with high betweenness centrality act as bridges between different social groups. These students are crucial for information flow and social cohesion across the class.",
            
            "density": "Network density measures the proportion of possible connections that actually exist. Higher density indicates stronger overall connectedness among students.",
            
            "recommendations": "\n".join(self.get_network_recommendations())
        }
        
        # Determine which response to use based on keywords
        if "community" in user_input_lower or "group" in user_input_lower:
            return responses["communities"]
        elif "isolated" in user_input_lower or "alone" in user_input_lower:
            return responses["isolation"]
        elif "influential" in user_input_lower or "leader" in user_input_lower:
            return responses["influential"]
        elif "between" in user_input_lower or "bridge" in user_input_lower:
            return responses["betweenness"]
        elif "dense" in user_input_lower or "density" in user_input_lower:
            return responses["density"]
        elif "recommend" in user_input_lower or "suggest" in user_input_lower:
            return responses["recommendations"]
        else:
            return responses["general_sna"]
    
    def get_chat_history(self, session_id=None, limit=10):
        """Get chat history, optionally filtered by session ID"""
        if session_id:
            history = [entry for entry in self.conversation_history if entry.get("session_id") == session_id]
        else:
            history = self.conversation_history.copy()
            
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit results
        return history[:limit]
        
    def get_recommendations(self, student_data=None):
        """Generate recommendations based on student data or recent conversations"""
        recommendations = []
        
        # Try to use actual student data if provided
        if student_data:
            # Here we would analyze actual student data
            # This would be expanded in a production system
            pass
        
        # Pull from recent conversations if available
        if self.conversation_history:
            recent_configs = []
            
            # Get recently modified configurations
            for entry in self.conversation_history[-10:]:  # Last 10 entries
                if entry.get("modified_config"):
                    recent_configs.append(entry.get("modified_config"))
            
            if recent_configs:
                # Analyze recent configurations to find patterns
                avg_gpa_weight = sum(cfg.get("gpa_penalty_weight", 0) for cfg in recent_configs) / len(recent_configs)
                avg_wellbeing_weight = sum(cfg.get("wellbeing_penalty_weight", 0) for cfg in recent_configs) / len(recent_configs)
                avg_bully_weight = sum(cfg.get("bully_penalty_weight", 0) for cfg in recent_configs) / len(recent_configs)
                
                # Generate specific recommendations based on patterns
                if avg_gpa_weight > 70:
                    recommendations.append("Academic performance has been a strong focus in recent allocations. Consider balancing with social factors.")
                
                if avg_wellbeing_weight > 70:
                    recommendations.append("Student wellbeing has been prioritized in recent allocations, which may be impacting academic balance.")
                
                if avg_bully_weight > 80:
                    recommendations.append("Bullying prevention has been set very high. This is appropriate for addressing serious concerns but may constrain other optimization goals.")
        
        # If we don't have enough data-driven recommendations, add some general ones
        default_recommendations = [
            "Consider setting a higher priority on bullying prevention to protect vulnerable students.",
            "Academic balance could be improved by increasing the GPA penalty weight.",
            "For better social dynamics, try increasing the influence standard weight.",
            "To improve overall class wellbeing, consider increasing the minimum friends required to 2.",
            "The current friendship balance weight may be too low for optimal social connections."
        ]
        
        # Fill in with default recommendations if needed
        while len(recommendations) < 2:
            random_rec = random.choice(default_recommendations)
            if random_rec not in recommendations:
                recommendations.append(random_rec)
        
        return recommendations[:2]  # Return top 2 recommendations

    def load_training_data(self):
        """Load training data from CSV file"""
        if os.path.exists(self.teacher_comments_file):
            try:
                df = pd.read_csv(self.teacher_comments_file)
                comments = df['teacher_comment'].tolist()
                recommendations = df['recommendation'].tolist()
                return comments, recommendations
            except Exception as e:
                print(f"Error loading training data: {e}")
                return [], []
        return [], []
    
    def fine_tune_model(self):
        """Fine-tune the last layer of BERT on teacher comments data"""
        try:
            # Only proceed if transformers are available
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers not available, cannot fine-tune BERT")
                return False
                
            # Load training data
            comments, recommendations = self.load_training_data()
            
            if not comments or not recommendations:
                print("No training data available")
                return False
                
            # Convert text recommendations to category labels
            labels = []
            for rec in recommendations:
                for category in self.category_terms:
                    if any(term in rec.lower() for term in self.category_terms[category]):
                        labels.append(category)
                        break
                else:
                    labels.append("academic")  # Default
                    
            # Get unique labels
            unique_labels = list(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            label_ids = [label_to_id[label] for label in labels]
            
            # Load pre-trained model
            print("Loading pre-trained BERT model for fine-tuning")
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(unique_labels),
                id2label={i: label for i, label in enumerate(unique_labels)},
                label2id=label_to_id
            )
            
            # Freeze all layers except the classification head
            print("Freezing base BERT layers, keeping only the classification head trainable")
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Freeze everything except classifier
                    param.requires_grad = False
                    
            # Tokenize the input texts
            print("Tokenizing training data")
            encodings = tokenizer(comments, truncation=True, padding=True, return_tensors="pt")
            
            # Create dataset
            from torch.utils.data import DataLoader, TensorDataset
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            labels_tensor = torch.tensor(label_ids)
            
            dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
            
            # Set up training parameters (minimal for fine-tuning just the last layer)
            from torch.optim import AdamW
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Use small batch size and learning rate for fine-tuning
            batch_size = 4
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Use lower learning rate for fine-tuning
            optimizer = AdamW(model.parameters(), lr=5e-5)
            
            # Train for a small number of epochs
            print(f"Fine-tuning on {len(comments)} examples using {device}")
            epochs = 3
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for batch in train_loader:
                    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
                    
                    # Clear previous gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update parameters
                    optimizer.step()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save the fine-tuned model
            print("Saving fine-tuned model")
            output_dir = "app/models/fine_tuned_bert"
            os.makedirs(output_dir, exist_ok=True)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save label mapping
            with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
                json.dump({
                    "id2label": {str(i): label for i, label in enumerate(unique_labels)},
                    "label2id": {label: i for i, label in enumerate(unique_labels)}
                }, f)
            
            # Update the model reference
            self.bert_model = model
            self.bert_tokenizer = tokenizer
            
            print("Fine-tuning completed successfully")
            return True
                
        except Exception as e:
            print(f"Error fine-tuning model: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def generate_csv_data(self):
        """Generate sample CSV data for training the chatbot"""
        teacher_comments = [
            "Student shows great academic potential but struggles with peer relationships.",
            "Very social student who needs academic support.",
            "Student has experienced bullying and requires a supportive environment.",
            "Highly influential student who can be disruptive if not engaged.",
            "Quiet student who benefits from having close friends in class.",
            "Student with specific learning needs who thrives with the right peer support.",
            "High achiever who sometimes dominates group activities.",
            "Student with wellbeing concerns who needs a stable social environment.",
            "Student who performs better when separated from certain peers.",
            "Student who needs to develop social skills through structured interaction."
        ]
        
        csv_data = "teacher_comment,recommendation\n"
        
        for comment in teacher_comments:
            # Determine which category this comment best matches
            academic = any(term in comment.lower() for term in ["academic", "achiever", "grades", "potential"])
            wellbeing = any(term in comment.lower() for term in ["wellbeing", "concerns", "support", "stable"])
            bullying = any(term in comment.lower() for term in ["bully", "bullying", "victim", "protective"])
            social = any(term in comment.lower() for term in ["social", "influential", "quiet", "interaction"])
            friendship = any(term in comment.lower() for term in ["friend", "peer", "relationship"])
            
            if academic:
                recommendation = self.sample_responses["academic_priority"]
            elif wellbeing:
                recommendation = self.sample_responses["wellbeing_priority"]
            elif bullying:
                recommendation = self.sample_responses["bullying_prevention"]
            elif social:
                recommendation = self.sample_responses["social_balance"]
            elif friendship:
                recommendation = self.sample_responses["friendship_focus"]
            else:
                recommendation = self.sample_responses["academic_priority"]
                
            csv_data += f"\"{comment}\",\"{recommendation}\"\n"
            
        return csv_data
        
    def get_priority_recommendations(self):
        """Get recommendations for priority settings"""
        current_config = self.get_current_config()
        
        recommendations = []
        
        # Check if any weights are below 30
        low_weights = []
        if current_config.get("gpa_penalty_weight", 0) < 30:
            low_weights.append("GPA penalty")
        if current_config.get("wellbeing_penalty_weight", 0) < 30:
            low_weights.append("wellbeing penalty")
        if current_config.get("bully_penalty_weight", 0) < 30:
            low_weights.append("bullying prevention")
            
        if low_weights:
            recommendations.append(f"The following weights may be too low for optimal results: {', '.join(low_weights)}.")
        
        # Check for minimum friends setting
        if current_config.get("min_friends_required", 0) == 0:
            recommendations.append("Setting minimum friends to 0 may lead to social isolation for some students.")
        elif current_config.get("min_friends_required", 0) > 3:
            recommendations.append("Setting minimum friends above 3 may be difficult to satisfy for all students.")
            
        # Check balance of priorities
        priority_counts = {}
        for key, value in current_config.items():
            if key.startswith("prioritize_") and value > 0:
                priority_counts[value] = priority_counts.get(value, 0) + 1
                
        if any(count > 1 for count in priority_counts.values()):
            recommendations.append("You have multiple criteria with the same priority level. Consider differentiating priorities for better results.")
        
        return recommendations
    
    # Social Network Analysis Methods
    
    def analyze_network_structure(self, relationships_data):
        """Analyze the structure of a social network
        
        Args:
            relationships_data: Dictionary with student_id pairs and relationship strength
                Format: {(student_id1, student_id2): strength}
        
        Returns:
            Dict containing network metrics and insights
        """
        if not NETWORK_ANALYSIS_AVAILABLE:
            return {"success": False, "message": "Network analysis libraries not available"}
        
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add edges with weights
            for (student1, student2), strength in relationships_data.items():
                G.add_edge(student1, student2, weight=strength)
            
            # Calculate basic network metrics
            metrics = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "average_clustering": nx.average_clustering(G),
                "connected_components": nx.number_connected_components(G)
            }
            
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Find top influential students
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Find potential communities
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G))
                community_sizes = [len(c) for c in communities]
                metrics["communities"] = community_sizes
            except:
                metrics["communities"] = "Unable to detect communities"
            
            # Generate insights based on network metrics
            insights = self.generate_network_insights(metrics, {
                "degree": dict(top_degree),
                "closeness": dict(top_closeness),
                "betweenness": dict(top_betweenness)
            })
            
            return {
                "success": True,
                "metrics": metrics,
                "top_influential_students": {
                    "degree_centrality": dict(top_degree),
                    "closeness_centrality": dict(top_closeness),
                    "betweenness_centrality": dict(top_betweenness)
                },
                "insights": insights
            }
        except Exception as e:
            return {"success": False, "message": f"Error analyzing network: {str(e)}"}
    
    def generate_network_insights(self, metrics, centrality_data):
        """Generate insights based on network metrics"""
        insights = []
        
        # Density insights
        if metrics["density"] < 0.1:
            insights.append("The social network is sparse, indicating limited overall connectivity between students.")
        elif metrics["density"] > 0.5:
            insights.append("The social network is dense, showing strong overall connectivity between students.")
        
        # Clustering insights
        if metrics["average_clustering"] > 0.6:
            insights.append("High clustering coefficient indicates strong friend groups forming tight-knit communities.")
        
        # Component insights
        if metrics["connected_components"] > 1:
            insights.append(f"The network has {metrics['connected_components']} separate groups with no connections between them.")
        
        # Centrality insights
        betweenness_values = list(centrality_data["betweenness"].values())
        if betweenness_values and max(betweenness_values) > 0.3:
            insights.append("Some students have high betweenness centrality, acting as bridges between different social groups.")
        
        return insights
    
    def identify_isolated_students(self, relationships_data):
        """Identify potentially isolated students in the network"""
        if not NETWORK_ANALYSIS_AVAILABLE:
            return {"success": False, "message": "Network analysis libraries not available"}
        
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add edges
            for (student1, student2), _ in relationships_data.items():
                G.add_edge(student1, student2)
            
            # Get all students with degree 0 or 1
            isolated_students = []
            for node, degree in G.degree():
                if degree <= 1:
                    isolated_students.append({"student_id": node, "connections": degree})
            
            return {
                "success": True,
                "isolated_students": isolated_students,
                "count": len(isolated_students),
                "recommendation": "Consider increasing the minimum friends parameter to reduce social isolation"
            }
        except Exception as e:
            return {"success": False, "message": f"Error identifying isolated students: {str(e)}"}
    
    def analyze_friendship_groups(self, relationships_data):
        """Analyze friendship groups and communities in the network"""
        if not NETWORK_ANALYSIS_AVAILABLE:
            return {"success": False, "message": "Network analysis libraries not available"}
        
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add edges with weights
            for (student1, student2), strength in relationships_data.items():
                G.add_edge(student1, student2, weight=strength)
            
            # Find communities using Louvain method
            try:
                communities = list(nx.algorithms.community.louvain_communities(G))
            except:
                # Fallback to greedy modularity
                try:
                    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
                except:
                    return {"success": False, "message": "Unable to detect communities"}
            
            # Analyze each community
            community_analysis = []
            for i, community in enumerate(communities):
                subgraph = G.subgraph(community)
                
                community_analysis.append({
                    "community_id": i + 1,
                    "size": len(community),
                    "density": nx.density(subgraph),
                    "average_clustering": nx.average_clustering(subgraph),
                    "students": list(community)
                })
            
            return {
                "success": True,
                "community_count": len(communities),
                "communities": community_analysis,
                "recommendation": "Consider optimizing class allocation based on these natural friendship groups"
            }
        except Exception as e:
            return {"success": False, "message": f"Error analyzing friendship groups: {str(e)}"}
            
    def get_network_recommendations(self):
        """Generate recommendations based on social network patterns"""
        recommendations = [
            "Consider identifying bridging students (high betweenness centrality) to facilitate class cohesion.",
            "Analyze friendship clusters to create balanced class distributions.",
            "Monitor isolated students and ensure they have at least 2-3 connections in their assigned class.",
            "Balance social leaders (high degree centrality) across different classes.",
            "Use network density metrics to validate social balance between classes."
        ]
        
        return random.sample(recommendations, min(3, len(recommendations))) 