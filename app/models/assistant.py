import json
import os
import random
import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import network analysis libraries
NETWORK_ANALYSIS_AVAILABLE = False
try:
    import networkx as nx
    NETWORK_ANALYSIS_AVAILABLE = True
    print("NetworkX loaded successfully for social network analysis")
except ImportError:
    print("NetworkX not available for network analysis")

# Constants for API endpoints if needed
API_BASE_URL = os.environ.get("API_BASE_URL", "")

class RuleBasedChatbot:
    def __init__(self):
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
        
        # Training data for the model
        self.teacher_comments_file = "app/models/teacher_comments.csv"
        
        # Sample responses for different priorities
        self.sample_responses = {
            "academic_priority": "I recommend prioritizing academic balance with a higher GPA penalty weight (60-70) and setting academic as the top priority (5).",
            "wellbeing_priority": "For student wellbeing focus, I suggest increasing the wellbeing_penalty_weight to 70-80 and setting it as the top priority (5).",
            "bullying_prevention": "To address bullying concerns, set bully_penalty_weight to 80-90 and make bullying prevention the top priority (5).",
            "social_balance": "For better social dynamics, increase influence_std_weight and isolated_std_weight to 70, and prioritize social influence (4-5).",
            "friendship_focus": "To optimize friendship connections, set min_friends_required to 2 and increase friend_inclusion_weight to 70-80."
        }
        
        # Categories and their related terms
        self.category_terms = {
            "academic": ["academic", "grades", "gpa", "performance", "achievement", "study", "learning", "smart", "grade", "class", "subject"],
            "wellbeing": ["wellbeing", "well-being", "mental health", "support", "emotional", "stress", "anxiety", "happy", "happiness", "health"],
            "bullying": ["bully", "bullying", "harassment", "victim", "protect", "safety", "secure", "cruel", "mean", "separation"],
            "social": ["social", "influence", "isolated", "popular", "interaction", "group", "clique", "leaders", "influence"],
            "friendship": ["friend", "friendship", "relationship", "peer", "buddy", "companion", "pals", "together", "connections"]
        }
        
        # Priority words
        self.increase_priority = ["prioritize", "focus on", "increase", "more important", "higher", "enhance", "boost", "maximize"]
        self.decrease_priority = ["less", "decrease", "lower", "reduce", "minimize", "less important", "downplay"]
        
        # Conversation memory
        self.conversation_history = []
        
        # Load CSV data
        self._load_csv_data()
        
        # Initialize vectorizer for text similarity
        self.vectorizer = None
        self._initialize_vectorizer()
        
        # Greetings and bad words detection
        self.greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "what's up", "howdy"]
        self.bad_words = ["damn", "hell", "shit", "fuck", "asshole", "bitch", "crap", "idiot", "stupid", "dumb"]
        
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer with teacher comments corpus"""
        try:
            if 'teacher_comments' in self.data_cache:
                comments = self.data_cache['teacher_comments']['teacher_comment'].tolist()
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.vectorizer.fit(comments)
                print("TF-IDF vectorizer initialized successfully with teacher comments corpus")
                
                # Save vectorizer metadata in multiple locations for compatibility
                try:
                    # Get default stop words since vectorizer might not have stop_words_ attribute
                    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                    stop_words_sample = list(ENGLISH_STOP_WORDS)[:10] + ["..."]
                    
                    # Create in app/models/tfidf_vectorizer (new location)
                    app_models_dir = "app/models/tfidf_vectorizer"
                    os.makedirs(app_models_dir, exist_ok=True)
                    with open(f"{app_models_dir}/vectorizer_info.json", "w") as f:
                        json.dump({
                            "vocab_size": len(self.vectorizer.vocabulary_),
                            "stop_words": stop_words_sample,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                except Exception as e:
                    print(f"Warning: Could not save vectorizer metadata: {e}")
            else:
                # If no data is available, initialize with basic corpus
                basic_corpus = list(self.sample_responses.values())
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.vectorizer.fit(basic_corpus)
                print("TF-IDF vectorizer initialized with basic response corpus")
        except Exception as e:
            print(f"Error initializing vectorizer: {e}")
            # Create a simple vectorizer with a minimal corpus
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(["academic", "wellbeing", "bullying", "social", "friendship"])
    
    def _load_csv_data(self):
        """Load CSV data for context-aware responses"""
        self.data_cache = {}
        try:
            # Load teacher comments if available
            if os.path.exists(self.teacher_comments_file):
                self.data_cache['teacher_comments'] = pd.read_csv(self.teacher_comments_file)
                print(f"Loaded {len(self.data_cache['teacher_comments'])} teacher comments")
            else:
                print("Teacher comments file not found!")
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
    
    def get_similar_comment(self, user_input):
        """Find the most similar teacher comment based on TF-IDF similarity"""
        try:
            if 'teacher_comments' not in self.data_cache:
                return random.choice(list(self.sample_responses.values()))
            
            # Transform the input text
            user_vector = self.vectorizer.transform([user_input])
            
            # Transform all comments
            comment_vectors = self.vectorizer.transform(self.data_cache['teacher_comments']['teacher_comment'].tolist())
            
            # Calculate similarities
            similarities = cosine_similarity(user_vector, comment_vectors)[0]
            
            # Get the most similar comment
            max_index = similarities.argmax()
            
            # Return corresponding recommendation
            return self.data_cache['teacher_comments']['recommendation'].iloc[max_index]
        except Exception as e:
            print(f"Error finding similar comment: {e}")
            return random.choice(list(self.sample_responses.values()))
    
    def is_greeting(self, text):
        """Check if the input text is a greeting"""
        text = text.lower()
        return any(greeting in text for greeting in self.greetings)
    
    def contains_bad_words(self, text):
        """Check if the input text contains bad words"""
        text = text.lower()
        return any(bad_word in text for bad_word in self.bad_words)
    
    def detect_categories(self, text):
        """Detect which educational categories are mentioned in the text"""
        text = text.lower()
        detected_categories = {}
        
        # Count mentions of each category
        for category, terms in self.category_terms.items():
            count = sum(1 for term in terms if term in text)
            if count > 0:
                detected_categories[category] = count
                
        return detected_categories
    
    def detect_priority_changes(self, text):
        """Detect if the user wants to increase or decrease priority for categories"""
        text = text.lower()
        priority_changes = {}
        
        # For each category, check if there are priority changes
        for category in self.category_terms.keys():
            # Check if category is mentioned
            if any(term in text for term in self.category_terms[category]):
                # Enhanced detection with more synonyms
                increase_words = self.increase_priority + ["boost", "elevate", "emphasize", "strengthen", "improve", "raise"]
                decrease_words = self.decrease_priority + ["diminish", "weaken", "de-emphasize", "lessen", "lower", "reduce"]
                
                # Check for increase keywords near category terms
                increase = any(inc in text for inc in increase_words)
                decrease = any(dec in text for dec in decrease_words)
                
                if increase and not decrease:
                    priority_changes[category] = "increase"
                elif decrease and not increase:
                    priority_changes[category] = "decrease"
                    
        return priority_changes
    
    def generate_config_changes(self, categories, priority_changes):
        """Generate configuration changes based on detected categories and priority changes"""
        current_config = self.get_current_config()
        modified_config = current_config.copy()
        changes_made = False
        
        # If no specific categories detected, return current config
        if not categories and not priority_changes:
            return modified_config, changes_made
        
        # Handle priorities based on detected categories
        for category, count in categories.items():
            if category == "academic" and count > 0:
                modified_config["gpa_penalty_weight"] = min(80, current_config["gpa_penalty_weight"] + 10)
                modified_config["prioritize_academic"] = 5
                changes_made = True
                
            elif category == "wellbeing" and count > 0:
                modified_config["wellbeing_penalty_weight"] = min(80, current_config["wellbeing_penalty_weight"] + 10)
                modified_config["prioritize_wellbeing"] = 5
                changes_made = True
                
            elif category == "bullying" and count > 0:
                modified_config["bully_penalty_weight"] = min(90, current_config["bully_penalty_weight"] + 10)
                modified_config["prioritize_bullying"] = 5
                changes_made = True
                
            elif category == "social" and count > 0:
                modified_config["influence_std_weight"] = min(80, current_config["influence_std_weight"] + 10)
                modified_config["isolated_std_weight"] = min(80, current_config["isolated_std_weight"] + 10)
                modified_config["prioritize_social_influence"] = 4
                changes_made = True
                
            elif category == "friendship" and count > 0:
                modified_config["friend_inclusion_weight"] = min(80, current_config["friend_inclusion_weight"] + 10)
                modified_config["friendship_balance_weight"] = min(70, current_config["friendship_balance_weight"] + 10)
                modified_config["min_friends_required"] = min(3, current_config["min_friends_required"] + 1)
                modified_config["prioritize_friendship"] = 4
                changes_made = True
                
        # Apply specific priority changes
        for category, change in priority_changes.items():
            if category == "academic":
                if change == "increase":
                    modified_config["prioritize_academic"] = 5
                    modified_config["gpa_penalty_weight"] = min(90, current_config["gpa_penalty_weight"] + 20)
                else:
                    modified_config["prioritize_academic"] = max(1, current_config["prioritize_academic"] - 1)
                    modified_config["gpa_penalty_weight"] = max(10, current_config["gpa_penalty_weight"] - 10)
                changes_made = True
                
            elif category == "wellbeing":
                if change == "increase":
                    modified_config["prioritize_wellbeing"] = 5
                    modified_config["wellbeing_penalty_weight"] = min(90, current_config["wellbeing_penalty_weight"] + 20)
                else:
                    modified_config["prioritize_wellbeing"] = max(1, current_config["prioritize_wellbeing"] - 1)
                    modified_config["wellbeing_penalty_weight"] = max(10, current_config["wellbeing_penalty_weight"] - 10)
                changes_made = True
                
            elif category == "bullying":
                if change == "increase":
                    modified_config["prioritize_bullying"] = 5
                    modified_config["bully_penalty_weight"] = min(95, current_config["bully_penalty_weight"] + 20)
                else:
                    modified_config["prioritize_bullying"] = max(1, current_config["prioritize_bullying"] - 1)
                    modified_config["bully_penalty_weight"] = max(20, current_config["bully_penalty_weight"] - 10)
                changes_made = True
                
            elif category == "social":
                if change == "increase":
                    modified_config["prioritize_social_influence"] = 5
                    modified_config["influence_std_weight"] = min(90, current_config["influence_std_weight"] + 20)
                    modified_config["isolated_std_weight"] = min(90, current_config["isolated_std_weight"] + 20)
                else:
                    modified_config["prioritize_social_influence"] = max(1, current_config["prioritize_social_influence"] - 1)
                    modified_config["influence_std_weight"] = max(20, current_config["influence_std_weight"] - 10)
                    modified_config["isolated_std_weight"] = max(20, current_config["isolated_std_weight"] - 10)
                changes_made = True
                
            elif category == "friendship":
                if change == "increase":
                    modified_config["prioritize_friendship"] = 5
                    modified_config["friend_inclusion_weight"] = min(90, current_config["friend_inclusion_weight"] + 20)
                    modified_config["friendship_balance_weight"] = min(80, current_config["friendship_balance_weight"] + 20)
                else:
                    modified_config["prioritize_friendship"] = max(1, current_config["prioritize_friendship"] - 1)
                    modified_config["friend_inclusion_weight"] = max(20, current_config["friend_inclusion_weight"] - 10)
                    modified_config["friendship_balance_weight"] = max(20, current_config["friendship_balance_weight"] - 10)
                changes_made = True
                
        return modified_config, changes_made
    
    def generate_response(self, user_input, modified_config, is_modified):
        """Generate a response based on the user input and configuration changes"""
        if self.is_greeting(user_input):
            return "Hello! I'm your classroom optimization assistant. I can help with balancing academic performance, student wellbeing, bullying prevention, social dynamics, and friendship connections. How can I assist you today?"
        
        if self.contains_bad_words(user_input):
            return "I'd appreciate it if we could keep our conversation respectful and professional. How can I help you with classroom optimization?"
        
        # If the configuration was modified based on the input
        if is_modified:
            # Get similar comment from the teacher comments dataset
            recommendation = self.get_similar_comment(user_input)
            
            # Format the response with a suggestion to visit the set_priorities page
            response = f"Based on your request, I recommend the following settings: {recommendation} To apply these changes, please visit the Settings > Set Priorities page."
            return response
        else:
            # If no specific modifications were made
            categories = self.detect_categories(user_input)
            
            if categories:
                # Return a general response based on the most mentioned category
                top_category = max(categories.items(), key=lambda x: x[1])[0]
                
                if top_category == "academic":
                    return f"{self.sample_responses['academic_priority']} To apply these changes, please visit the Settings > Set Priorities page."
                elif top_category == "wellbeing":
                    return f"{self.sample_responses['wellbeing_priority']} To apply these changes, please visit the Settings > Set Priorities page."
                elif top_category == "bullying":
                    return f"{self.sample_responses['bullying_prevention']} To apply these changes, please visit the Settings > Set Priorities page."
                elif top_category == "social":
                    return f"{self.sample_responses['social_balance']} To apply these changes, please visit the Settings > Set Priorities page."
                elif top_category == "friendship":
                    return f"{self.sample_responses['friendship_focus']} To apply these changes, please visit the Settings > Set Priorities page."
            
            # Default response if no specific category is detected
            return "I can help you optimize class allocation settings. Please specify if you're interested in academic performance, student wellbeing, bullying prevention, social dynamics, or friendship connections. I can provide recommendations which you can apply from the Settings > Set Priorities page."
    
    def analyze_request(self, user_input, session_id=None):
        """Main function to analyze user input and generate a response"""
        # Log the interaction
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            # Skip processing for very short inputs
            if len(user_input.strip()) < 2:
                response = "I didn't quite catch that. How can I help you with classroom optimization today?"
                
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
                    "is_modified": False
                }
            
            # Handle profanity first
            if self.contains_bad_words(user_input):
                profanity_response = "I'd appreciate it if we could keep our conversation respectful and professional. How can I help you with classroom optimization?"
                
                conversation_entry = {
                    "timestamp": timestamp,
                    "session_id": session_id if session_id else str(random.randint(1000, 9999)),
                    "user_input": user_input,
                    "response": profanity_response,
                    "modified_config": None
                }
                
                self.conversation_history.append(conversation_entry)
                
                return {
                    "success": True,
                    "message": profanity_response,
                    "modified_config": self.get_current_config(),
                    "is_modified": False
                }
            
            # Handle greetings
            if self.is_greeting(user_input):
                greeting_response = "Hello! I'm your classroom optimization assistant. I can help with balancing academic performance, student wellbeing, bullying prevention, social dynamics, and friendship connections. How can I assist you today?"
                
                conversation_entry = {
                    "timestamp": timestamp,
                    "session_id": session_id if session_id else str(random.randint(1000, 9999)),
                    "user_input": user_input,
                    "response": greeting_response,
                    "modified_config": None
                }
                
                self.conversation_history.append(conversation_entry)
                
                return {
                    "success": True,
                    "message": greeting_response,
                    "modified_config": self.get_current_config(),
                    "is_modified": False
                }
            
            # Detect categories and priority changes
            categories = self.detect_categories(user_input)
            priority_changes = self.detect_priority_changes(user_input)
            
            # Generate configuration changes
            modified_config, is_modified = self.generate_config_changes(categories, priority_changes)
            
            # No longer save configuration changes automatically - users will need to do this manually
            # if is_modified:
            #     self.save_config(modified_config)
            
            # Generate a response
            response = self.generate_response(user_input, modified_config, is_modified)
            
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
                "is_modified": is_modified
            }
        except Exception as e:
            print(f"Error analyzing request: {e}")
            return {
                "success": False,
                "message": "Sorry, I encountered an error processing your request. Please try again.",
                "modified_config": self.get_current_config(),
                "is_modified": False
            }
    
    def get_chat_history(self, session_id=None, limit=10):
        """Get recent chat history, optionally filtered by session_id"""
        if session_id:
            history = [entry for entry in self.conversation_history if entry.get("session_id") == session_id]
        else:
            history = self.conversation_history.copy()
            
        # Return most recent conversations up to the limit
        return history[-limit:] if len(history) > limit else history


# Create an instance for import in other modules
chatbot = RuleBasedChatbot()


# For backwards compatibility with existing code
class AssistantModel:
    def __init__(self):
        self.chatbot = chatbot
    
    def analyze_request(self, user_input, session_id=None):
        return self.chatbot.analyze_request(user_input, session_id)
    
    def get_current_config(self):
        return self.chatbot.get_current_config()
    
    def save_config(self, config):
        return self.chatbot.save_config(config)
    
    def get_chat_history(self, session_id=None, limit=10):
        return self.chatbot.get_chat_history(session_id, limit)
    
    def get_recommendations(self, student_data=None):
        """Get general recommendations for overall visualizations"""
        try:
            config = self.get_current_config()
            
            # A list of general recommendations based on current configuration
            recommendations = [
                "Consider exploring the social connections graph to understand student relationships.",
                "Review the Power BI dashboard for a high-level overview of class dynamics.",
                "The network graphs can provide insights into specific types of student interactions."
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
    
    def get_priority_recommendations(self):
        """Get prioritization recommendations based on current configuration"""
        try:
            config = self.get_current_config()
            recommendations = []
            
            # Academic recommendations
            if config.get("prioritize_academic", 0) >= 4:
                recommendations.append("Your current settings prioritize academic balance. Consider adjusting the GPA Balance Weight to fine-tune this priority.")
            
            # Wellbeing recommendations
            if config.get("wellbeing_penalty_weight", 0) < 40:
                recommendations.append("Consider increasing the Wellbeing Balance Weight to better support student mental health.")
            
            # Bullying recommendations
            if config.get("bully_penalty_weight", 0) >= 70:
                recommendations.append("Your bullying prevention setting is high, which will strongly separate students with negative interactions.")
            
            # Social influence recommendations
            if config.get("influence_std_weight", 0) < 40:
                recommendations.append("Increasing the Influence Balance Weight would create more balanced classes in terms of social dynamics.")
            
            # Friendship recommendations
            if config.get("min_friends_required", 0) < 1:
                recommendations.append("Setting Minimum In-Class Friends to at least 1 ensures students have some social support.")
            
            # If no specific recommendations, provide general ones
            if not recommendations:
                recommendations = [
                    "Adjust priorities by dragging items in the Objective Priorities section.",
                    "Balance friendship inclusion with other priorities for optimal class dynamics.",
                    "Consider increasing the Bully-Victim Separation Weight if bullying is a concern.",
                    "Aim for a balanced distribution of influential students across classes."
                ]
            
            return recommendations
        except Exception as e:
            print(f"Error getting priority recommendations: {e}")
            return [
                "Adjust priorities by dragging items in the Objective Priorities section.",
                "Balance academic and social goals for optimal class allocation."
            ] 