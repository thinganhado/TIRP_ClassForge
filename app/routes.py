# app/routes.py
from flask import Blueprint, render_template, jsonify
from flask import request, redirect, url_for, flash, session
import json
import uuid
from app.database.student_queries import (
    fetch_all_students,
    fetch_student_details,
)
from app.database.class_queries import fetch_unique_classes
from app.database.softcons_queries import SoftConstraint
from app import db
import subprocess
import pandas as pd
from sqlalchemy import text
from app.models.assistant import AssistantModel
from datetime import datetime
import os
import sys

# Initialize the assistant model
assistant = AssistantModel()

main = Blueprint("main", __name__)

# ─────────────  core pages  ──────────────
@main.route("/")
def dashboard():
    return render_template("Index.html")

@main.route("/students")
def students():
    return render_template("Students.html")

# ─────────────  visualisation  ───────────
@main.route("/visualization/overall")
def overall():
    # Simply render the template without calling get_recommendations
    return render_template("Overall.html")

@main.route("/visualization/individual")
def individual():
    # Simply render the template without calling get_recommendations
    return render_template("studentindividual.html")   # ← file name

# ─────────────  customisation section ────
@main.route("/customisation")
def customisation_home():
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("customisation.html", session_id=session['session_id'])

@main.route("/customisation/set-priorities")
def set_priorities():
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    # Get priority recommendations from the assistant
    priority_recommendations = assistant.get_priority_recommendations()
    return render_template("set_priorities.html", recommendations=priority_recommendations, session_id=session['session_id'])

@main.route("/customisation/specifications")
def specification():
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("specification.html", session_id=session['session_id'])

@main.route("/customisation/ai-assistant")
def ai_assistant():
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Get chat history for current session
    chat_history = assistant.get_chat_history(session_id=session['session_id'])
    
    return render_template("ai_assistant.html", 
                          chat_history=chat_history, 
                          session_id=session['session_id'])

@main.route("/customisation/history")
def history():
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Get all chat history for admin view
    all_chat_history = assistant.get_chat_history(limit=50)  # Get last 50 conversations
    return render_template("history.html", chat_history=all_chat_history, session_id=session['session_id'])

@main.route('/submit_customisation', methods=['POST'])
def submit_customisation():
    try:
        # --- Extract form data ---
        # Hard constraints (read but don't store in db for now)
        class_size = int(request.form.get("class-size", 30))
        max_classes = int(request.form.get("max-classes", 6))
        
        # Soft constraints
        gpa_penalty_weight = int(request.form.get("gpa_penalty_weight", 30))
        wellbeing_penalty_weight = int(request.form.get("wellbeing_penalty_weight", 50))
        bully_penalty_weight = int(request.form.get("bully_penalty_weight", 60))
        influence_std_weight = int(request.form.get("influence_std_weight", 60))
        isolated_std_weight = int(request.form.get("isolated_std_weight", 60))
        min_friends_required = int(request.form.get("min_friends_required", 1))
        friend_inclusion_weight = int(request.form.get("friend_inclusion_weight", 60))
        friendship_balance_weight = int(request.form.get("friend_balance_weight", 60))

        priority_csv = request.form.get("priority_order", "")
        priority_list = priority_csv.split(",") if priority_csv else []

        # --- Map priority list to weights ---
        priority_mapping = {
            "academic_performance": "prioritize_academic",
            "student_wellbeing": "prioritize_wellbeing",
            "bullying_prevention": "prioritize_bullying",
            "social_influence": "prioritize_social_influence",
            "friendship_connections": "prioritize_friendship"
        }
        priority_weights = {v: 0 for v in priority_mapping.values()}
        for rank, key in enumerate(priority_list[::-1], start=1):  # Higher rank = higher weight
            if key in priority_mapping:
                priority_weights[priority_mapping[key]] = rank

        # --- Store in SQL ---
        new_entry = SoftConstraint(
            # Removed class_size and max_classes to avoid DB error
            gpa_penalty_weight=gpa_penalty_weight,
            wellbeing_penalty_weight=wellbeing_penalty_weight,
            bully_penalty_weight=bully_penalty_weight,
            influence_std_weight=influence_std_weight,
            isolated_std_weight=isolated_std_weight,
            min_friends_required=min_friends_required,
            friend_inclusion_weight=friend_inclusion_weight,
            friendship_balance_weight=friendship_balance_weight,
            **priority_weights
        )
        db.session.add(new_entry)
        db.session.commit()

        # --- Optional: Save JSON too ---
        constraints = {
            "class_size": class_size,  # Still include in JSON file
            "max_classes": max_classes, # Still include in JSON file
            "gpa_penalty_weight": gpa_penalty_weight,
            "wellbeing_penalty_weight": wellbeing_penalty_weight,
            "bully_penalty_weight": bully_penalty_weight,
            "influence_std_weight": influence_std_weight,
            "isolated_std_weight": isolated_std_weight,
            "min_friends_required": min_friends_required,
            "friend_inclusion_weight": friend_inclusion_weight,
            "friendship_balance_weight": friendship_balance_weight,
            **priority_weights
        }
        with open("soft_constraints_config.json", "w") as f:
            json.dump(constraints, f, indent=2)

        return redirect(url_for("main.customisation_loading"))

    except Exception as e:
        flash(f"Submission error: {e}")
        return redirect(url_for("main.set_priorities"))
    
@main.route("/customisation/loading")
def customisation_loading():
    # Just render the loading template - don't automatically trigger allocation
    # Add a flag to indicate this is coming from set priorities page not chatbot
    return render_template("customisation_loading.html", from_set_priorities=True)
    
@main.route("/run_allocation")
def run_allocation():
    """Run the allocation algorithm and redirect to Students page when complete"""
    try:
        # Get the absolute path to the finalallocation.py script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_models", "finalallocation.py"))
        
        # Run the allocation script
        result = subprocess.run([sys.executable, script_path], 
                                capture_output=True, 
                                text=True,
                                check=True)
        
        if result.returncode == 0:
            # If successful, redirect to Students page
            return redirect(url_for("main.students"))
        else:
            # If there was an error, show error page
            flash(f"Allocation failed: {result.stderr}")
            return redirect(url_for("main.set_priorities"))
    except Exception as e:
        flash(f"Error running allocation: {str(e)}")
        return redirect(url_for("main.set_priorities"))
    
# ─────────────  (placeholder) classes  ───
@main.route("/classes")
def classes():
    # create a simple placeholder page so the menu link works
    return "<h1>Classes – coming soon</h1>"

# ─────────────  JSON APIs  ───────────────
@main.route("/api/students")
def api_students():
    return jsonify(fetch_all_students())

@main.route("/api/students/<int:sid>")
def api_student_detail(sid):
    return jsonify(fetch_student_details(sid))

@main.route("/api/classes")
def api_classes():
    return jsonify(fetch_unique_classes())

# ─────────────  Chatbot API Endpoints  ───────────────
@main.route("/api/assistant/analyze", methods=["POST"])
def analyze_request():
    """Analyze a user request and provide customization recommendations"""
    try:
        data = request.get_json()
        user_input = data.get("input", "")
        session_id = data.get("session_id", None) or session.get('session_id')
        
        if not user_input:
            return jsonify({"success": False, "message": "No input provided"}), 400
        
        # Check if the assistant has an initialize_model method (for backwards compatibility)
        if hasattr(assistant, 'initialize_model') and hasattr(assistant, 'models_loaded'):
            if not assistant.models_loaded:
                load_success = assistant.initialize_model()
                if not load_success:
                    print("Warning: Could not load NLP models. Using rule-based analysis only.")
        
        # Process the request using our assistant model
        result = assistant.analyze_request(user_input, session_id=session_id)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in analyze_request: {e}")
        return jsonify({"success": False, "message": f"Error processing request: {str(e)}"}), 500

@main.route("/api/assistant/confirm", methods=["POST"])
def confirm_changes():
    """Confirm and apply changes recommended by the assistant"""
    try:
        data = request.get_json()
        config = data.get("config", {})
        
        if not config:
            return jsonify({"success": False, "message": "No configuration provided"}), 400
        
        # Save the configuration
        updated_config = assistant.save_config(config)
        
        # Store in database
        new_entry = SoftConstraint(
            # Removed class_size and max_classes to avoid DB error
            gpa_penalty_weight=config.get("gpa_penalty_weight", 30),
            wellbeing_penalty_weight=config.get("wellbeing_penalty_weight", 50),
            bully_penalty_weight=config.get("bully_penalty_weight", 60),
            influence_std_weight=config.get("influence_std_weight", 60),
            isolated_std_weight=config.get("isolated_std_weight", 60),
            min_friends_required=config.get("min_friends_required", 1),
            friendship_score_weight=config.get("friendship_score_weight", 50),
            friendship_balance_weight=config.get("friendship_balance_weight", 60),
            prioritize_academic=config.get("prioritize_academic", 5),
            prioritize_wellbeing=config.get("prioritize_wellbeing", 4),
            prioritize_bullying=config.get("prioritize_bullying", 3),
            prioritize_social_influence=config.get("prioritize_social_influence", 2),
            prioritize_friendship=config.get("prioritize_friendship", 1)
        )
        db.session.add(new_entry)
        db.session.commit()
        
        # Add a success message to the conversation history
        if 'session_id' in session:
            # Create a conversation entry and add it directly to the chatbot's history
            conversation_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session['session_id'],
                "user_input": "Applied changes",
                "response": "Changes have been applied successfully. The optimization will now use your customized settings.",
                "modified_config": None
            }
            
            # Use the chatbot's method to add to history
            if hasattr(assistant.chatbot, 'conversation_history'):
                assistant.chatbot.conversation_history.append(conversation_entry)
            
        # Just return success JSON - don't redirect to customisation_loading which would trigger allocation
        return jsonify({"success": True, "message": "Changes applied successfully", "config": updated_config})
    except Exception as e:
        print(f"Error applying changes: {e}")
        return jsonify({"success": False, "message": f"Error applying changes: {str(e)}"}), 500

@main.route("/api/assistant/recommendations", methods=["GET"])
def get_recommendations():
    """Get general recommendations from the assistant"""
    try:
        # Provide default recommendations without trying to call assistant.get_recommendations()
        recommendations = [
            "Consider exploring different visualization options to understand student relationships better.",
            "Review class allocation settings periodically to optimize for changing needs.",
            "Analyze social networks to identify isolated students who may need more support."
        ]
        
        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({"success": False, "message": f"Error getting recommendations: {str(e)}"}), 500

@main.route("/api/assistant/chat_history", methods=["GET"])
def get_chat_history():
    """Get chat history for the current session or all sessions"""
    try:
        # Get session_id from request
        session_id = request.args.get("session_id") or session.get('session_id')
        limit = int(request.args.get("limit", 10))
        
        # Get history from the assistant
        history = assistant.get_chat_history(session_id=session_id, limit=limit)
        
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error getting chat history: {str(e)}"}), 500

@main.route("/api/assistant/priority_recommendations", methods=["GET"])
def get_priority_recommendations():
    """Get recommendations for priority settings"""
    try:
        # Get recommendations for priorities
        recommendations = assistant.get_priority_recommendations()
        
        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error getting priority recommendations: {str(e)}"}), 500

@main.route("/api/assistant/export_training_data", methods=["GET"])
def export_training_data():
    """Generate and export training data for the chatbot"""
    try:
        # Generate CSV data
        csv_data = assistant.generate_csv_data()
        
        return jsonify({"success": True, "data": csv_data})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error generating training data: {str(e)}"}), 500

@main.route("/api/assistant/fine_tune", methods=["POST"])
def fine_tune_model():
    """Fine-tune the model with the provided training data"""
    try:
        # Fine-tune the model
        success = assistant.fine_tune_model()
        
        if success:
            return jsonify({"success": True, "message": "Model fine-tuned successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to fine-tune model. See server logs for details."}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fine-tuning model: {str(e)}"}), 500

# ─────────────  Social Network Analysis Endpoints  ───────────────

@main.route("/api/network/analyze", methods=["POST"])
def analyze_network():
    """Analyze the social network structure"""
    try:
        data = request.get_json()
        relationships = data.get("relationships", {})
        
        # Convert relationship data format if needed
        formatted_relationships = {}
        for rel in relationships:
            student1 = rel.get("student1")
            student2 = rel.get("student2")
            strength = rel.get("strength", 1.0)
            if student1 and student2:
                formatted_relationships[(student1, student2)] = strength
        
        # Perform network analysis
        result = assistant.analyze_network_structure(formatted_relationships)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error analyzing network: {str(e)}"}), 500

@main.route("/api/network/isolated", methods=["POST"])
def identify_isolated_students():
    """Identify isolated students in the network"""
    try:
        data = request.get_json()
        relationships = data.get("relationships", {})
        
        # Convert relationship data format if needed
        formatted_relationships = {}
        for rel in relationships:
            student1 = rel.get("student1")
            student2 = rel.get("student2")
            strength = rel.get("strength", 1.0)
            if student1 and student2:
                formatted_relationships[(student1, student2)] = strength
        
        # Find isolated students
        result = assistant.identify_isolated_students(formatted_relationships)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error identifying isolated students: {str(e)}"}), 500

@main.route("/api/network/communities", methods=["POST"])
def analyze_communities():
    """Analyze friendship groups and communities"""
    try:
        data = request.get_json()
        relationships = data.get("relationships", {})
        
        # Convert relationship data format if needed
        formatted_relationships = {}
        for rel in relationships:
            student1 = rel.get("student1")
            student2 = rel.get("student2")
            strength = rel.get("strength", 1.0)
            if student1 and student2:
                formatted_relationships[(student1, student2)] = strength
        
        # Analyze friendship groups
        result = assistant.analyze_friendship_groups(formatted_relationships)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error analyzing communities: {str(e)}"}), 500

@main.route("/api/network/recommendations", methods=["GET"])
def get_network_recommendations():
    """Get recommendations based on social network patterns"""
    try:
        recommendations = assistant.get_network_recommendations()
        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error getting recommendations: {str(e)}"}), 500