# app/routes.py
from flask import Blueprint, render_template, jsonify
from flask import request, redirect, url_for, flash
import json
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
    return render_template("Overall.html")

@main.route("/visualization/individual")
def individual():
    return render_template("studentindividual.html")   # ← file name

# ─────────────  customisation section ────
@main.route("/customisation")
def customisation_home():
    return render_template("customisation.html")

@main.route("/customisation/set-priorities")
def set_priorities():
    return render_template("set_priorities.html")

@main.route("/customisation/specifications")
def specification():
    return render_template("specification.html")

@main.route("/customisation/ai-assistant")
def ai_assistant():
    return render_template("ai_assistant.html")

@main.route("/customisation/history")
def history():
    return render_template("history.html")

@main.route('/submit_customisation', methods=['POST'])
def submit_customisation():
    try:
        # --- Extract form data ---
        gpa_penalty_weight = int(request.form.get("gpa_penalty_weight", 30))
        wellbeing_penalty_weight = int(request.form.get("wellbeing_penalty_weight", 50))
        bully_penalty_weight = int(request.form.get("bully_penalty_weight", 60))
        influence_std_weight = int(request.form.get("influence_std_weight", 60))
        isolated_std_weight = int(request.form.get("isolated_std_weight", 60))
        min_friends_required = int(request.form.get("min_friends_required", 1))
        friend_inclusion_weight = int(request.form.get("friend_inclusion_weight", 60))
        friendship_balance_weight = int(request.form.get("friendship_balance_weight", 60))

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

        # --- Save JSON config for GA ---
        constraints = {
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
        with open("app/ml_models/soft_constraints_config.json", "w") as f:
            json.dump(constraints, f, indent=2)

        # --- Run GA script (which handles DB insertion itself) ---
        result = subprocess.run(
            ["python", "finalallocation.py"],
            capture_output=True,
            text=True,
            cwd="app/ml_models"
        )

        if result.returncode != 0:
            print("GA Script Error Output:\n", result.stderr)
            flash(f"GA Allocation failed. Error:\n{result.stderr}")
            return redirect(url_for("main.set_priorities"))

        # --- Success ---
        return redirect(url_for("main.overall"))

    except Exception as e:
        flash(f"Submission error: {e}")
        return redirect(url_for("main.set_priorities"))
    
@main.route("/customisation/loading")
def customisation_loading():
    return render_template("customisation_loading.html")
    
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