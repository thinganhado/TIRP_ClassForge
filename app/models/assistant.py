import os
import random
import pandas as pd
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class RuleBasedChatbot:
    def __init__(self):
        self.teacher_comments_file = "app/models/cleaned_teacher_comments (1).csv"
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
        self.sample_responses = {
            "academic_priority": "I recommend prioritizing academic balance with a higher GPA penalty weight (60-70) and setting academic as the top priority (5).",
            "wellbeing_priority": "For student wellbeing focus, I suggest increasing the wellbeing_penalty_weight to 70-80 and setting it as the top priority (5).",
            "bullying_prevention": "To address bullying concerns, set bully_penalty_weight to 80-90 and make bullying prevention the top priority (5).",
            "social_balance": "For better social dynamics, increase influence_std_weight and isolated_std_weight to 70, and prioritize social influence (4-5).",
            "friendship_focus": "To optimize friendship connections, set min_friends_required to 2 and increase friend_inclusion_weight to 70-80."
        }
        # Conversation memory
        self.conversation_history = []
        self._load_csv_data()
        self._initialize_vectorizer()
        self._initialize_ml_model()

    def _load_csv_data(self):
        self.data_cache = {}
        if os.path.exists(self.teacher_comments_file):
            self.data_cache['teacher_comments'] = pd.read_csv(self.teacher_comments_file)
        else:
            print("Teacher comments file not found.")

    def _initialize_vectorizer(self):
        if 'teacher_comments' in self.data_cache:
            comments = self.data_cache['teacher_comments']['comment'].tolist()
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(comments)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(["academic", "wellbeing", "bullying", "social", "friendship"])
    
    def _initialize_ml_model(self):
        """Initialize ML model for better classification of user requests"""
        try:
            if 'teacher_comments' in self.data_cache:
                print("[INFO] Initializing ML classification model...")
                df = self.data_cache['teacher_comments']
                
                # Encode recommendation labels
                self.label_encoder = LabelEncoder()
                df["label"] = self.label_encoder.fit_transform(df["recommendation"])
                
                # Create ML pipeline with TF-IDF and LogisticRegression
                self.ml_model = Pipeline([
                    ("tfidf", TfidfVectorizer(stop_words="english")),
                    ("clf", LogisticRegression(max_iter=1000))
                ])
                
                # Train the model on comment-recommendation pairs
                self.ml_model.fit(df["comment"], df["label"])
                print("[INFO] ML model successfully trained on teacher comments data")
                
                # Store the original recommendations for inverse transform
                self.unique_recommendations = df["recommendation"].unique()
                self.is_ml_model_available = True
            else:
                print("[WARN] Teacher comments data not available for ML model training")
                self.is_ml_model_available = False
        except Exception as e:
            print(f"[ERROR] Failed to initialize ML model: {e}")
            self.is_ml_model_available = False

    def get_recommendation_ml(self, user_input):
        """Get recommendation using the ML model"""
        try:
            if self.is_ml_model_available:
                # Predict the label
                predicted_label = self.ml_model.predict([user_input])[0]
                
                # Convert back to recommendation text
                recommendation = self.data_cache['teacher_comments'].loc[
                    self.data_cache['teacher_comments']["label"] == predicted_label, 
                    "recommendation"
                ].iloc[0]
                
                print(f"[INFO] ML model predicted recommendation for input: {user_input}")
                return recommendation
            else:
                print("[WARN] ML model not available, falling back to similarity search")
                return self.get_similar_comment(user_input)
        except Exception as e:
            print(f"[ERROR] Error in ML recommendation: {e}")
            return self.get_similar_comment(user_input)

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
        """Get current constraints configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.default_config
        return self.default_config
    
    def save_config(self, config):
        """Save updated configuration"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            return config
        except Exception as e:
            print(f"Error saving config file: {e}")
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
            
            # Try ML model first, then fallback to similarity search
            if hasattr(self, 'is_ml_model_available') and self.is_ml_model_available:
                recommendation = self.get_recommendation_ml(user_input)
            else:
                recommendation = self.get_similar_comment(user_input)
            
            # Log the conversation
            conversation_entry = {
                "timestamp": timestamp,
                "session_id": session_id if session_id else str(random.randint(1000, 9999)),
                "user_input": user_input,
                "response": f"Based on your request, I recommend the following settings: {recommendation}",
                "modified_config": None
            }
            self.conversation_history.append(conversation_entry)
            
            return {
                "success": True,
                "message": f"Based on your request, I recommend the following settings: {recommendation}",
                "modified_config": self.get_current_config(),
                "is_modified": True,
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
    
    def is_greeting(self, text):
        """Check if the text is a greeting"""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in greetings)

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
