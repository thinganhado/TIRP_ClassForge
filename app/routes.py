# app/routes.py
from flask import Blueprint, render_template, jsonify
from flask import request, redirect, url_for, flash, session
import json, uuid, os, sys, subprocess
from datetime import datetime
from math import ceil

import pandas as pd
from sqlalchemy import text

from app.database.student_queries import (
    fetch_all_students,
    fetch_student_details,
    fetch_students,
)
from app.database.class_queries import (
    fetch_unique_classes,
    fetch_all_classes,
    fetch_disrespect_for_class,
    get_cohort_averages,
    get_class_metrics,
    fetch_classes_summary,
)
from app.database.softcons_queries import SoftConstraint
from app.models.assistant import AssistantModel
from app import db

# ──────────────────────────────────────────

assistant = AssistantModel()           # NLP / rule-based helper
main      = Blueprint("main", __name__)

# ╭────────  core pages  ─────────╮
@main.route("/")
def dashboard():
    return render_template("Index.html")

@main.route("/students")
def students():
    return render_template("Students.html")

# ─────────────  classes page  ─────────────
@main.route("/classes")
def classes():
    all_classes = fetch_all_classes()        # [{name, students:[{id,name},…]}]
    cohort_avg_gpa, cohort_avg_well = get_cohort_averages()

    for cls in all_classes:
        # split bully / victim / remainder
        edges       = fetch_disrespect_for_class(cls["name"])
        bully_ids   = {e["target"] for e in edges}
        victim_ids  = {e["source"] for e in edges} - bully_ids
        remainder   = [s for s in cls["students"]
                       if s["id"] not in bully_ids | victim_ids]
        remainder.sort(key=lambda s: s["name"])
        half = ceil(len(remainder) / 2)

        cls["left_students"]  = (
            [s for s in cls["students"] if s["id"] in bully_ids] + remainder[:half]
        )
        cls["right_students"] = (
            [s for s in cls["students"] if s["id"] in victim_ids] + remainder[half:]
        )
        cls["edges"] = edges

        # per-class metrics
        m                  = get_class_metrics(cls["name"])
        cls["avg_gpa"]       = m["avg_gpa"]
        cls["avg_well"]      = m["avg_well"]
        cls["num_conflicts"] = m["num_conflicts"]

    return render_template(
        "Classes.html",
        classes=all_classes,
        cohort_avg_gpa=cohort_avg_gpa,
        cohort_avg_well=cohort_avg_well,
    )

# ╭────────  visualisation pages  ─────────╮
def _safe_recommendations():
    """Return assistant recs or a default list if model fails."""
    try:
        return assistant.get_recommendations()
    except Exception as e:
        print(f"[viz] get_recommendations failed: {e}")
        return [
            "Explore the graphs to understand class dynamics.",
            "Look for isolated students who may need support.",
            "Adjust priorities in the customisation section if needed.",
        ]

@main.route("/visualization/overall")
def overall():
    return render_template("Overall.html", recommendations=_safe_recommendations())

@main.route("/visualization/individual")
def individual():
    return render_template("studentindividual.html",
                           recommendations=_safe_recommendations())

# ╭────────  customisation workflow  ───────╮
@main.route("/customisation")
def customisation_home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("customisation.html", session_id=session["session_id"])

@main.route("/customisation/set-priorities")
def set_priorities():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    priority_recs = assistant.get_priority_recommendations()
    return render_template("set_priorities.html",
                           recommendations=priority_recs,
                           session_id=session["session_id"])

@main.route("/customisation/specification")
def specification():
    # students: [{id, name, class_id}, …]
    students = fetch_students()
    # classes_summary: [{class_id, count}, …]
    classes_summary = fetch_classes_summary()
    return render_template(
        "specifications.html",
        students=students,
        classes_summary=classes_summary
    )

@main.route("/customisation/ai-assistant")
def ai_assistant():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    chat_history = assistant.get_chat_history(session_id=session["session_id"])
    return render_template("ai_assistant.html",
                           chat_history=chat_history,
                           session_id=session["session_id"])

@main.route("/customisation/history")
def history():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    all_history = assistant.get_chat_history(limit=50)
    return render_template("history.html",
                           chat_history=all_history,
                           session_id=session["session_id"])

# ---------------- submit_customisation (unchanged) ----------------
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
    return render_template("customisation_loading.html",
                           from_set_priorities=True)   # keep v1 flag

# run_allocation unchanged
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

# ╭────────  JSON APIs  ─────────╮
@main.route("/api/students")
def api_students():
    return jsonify(fetch_all_students())

@main.route("/api/students/<int:sid>")
def api_student_detail(sid):
    return jsonify(fetch_student_details(sid))

@main.route("/api/classes")
def api_classes():
    return jsonify(fetch_unique_classes())

@main.route("/api/classes/<class_id>/avg_gpa")
def api_class_avg_gpa(class_id):
    sql = text("""
        SELECT AVG(p.perc_academic) AS avg_gpa
          FROM participants p
          JOIN allocations a ON p.student_id = a.student_id
         WHERE a.class_id = :cid
    """)
    try:
        row = db.session.execute(sql, {"cid": class_id}).fetchone()
        avg = float(row.avg_gpa) if row.avg_gpa is not None else 0.0
        return jsonify({"class_id": class_id, "avg_gpa": avg})
    except Exception as e:
        return jsonify({"error": str(e), "class_id": class_id}), 500

# ╭────────  assistant endpoints (merged) ─────────╮
@main.route("/api/assistant/analyze", methods=["POST"])
def analyze_request():
    data = request.get_json()
    user_input = data.get("input", "")
    session_id = data.get("session_id") or session.get("session_id")

    if not user_input:
        return jsonify({"success": False, "message": "No input"}), 400

    # lazy-load models once
    if hasattr(assistant, "models_loaded") and not assistant.models_loaded:
        assistant.initialize_model()

    result = assistant.analyze_request(user_input, session_id=session_id)
    return jsonify(result)

@main.route("/api/assistant/recommendations")
def get_recommendations():
    try:
        student_data = request.args.get("student_data")
        data_obj = json.loads(student_data) if student_data else None
        recs = assistant.get_recommendations(student_data=data_obj)
        return jsonify({"success": True, "recommendations": recs})
    except Exception as e:
        default = [
            "Explore the social-network graph for relationship insights.",
            "Balance influential students across classes.",
            "Review class allocation settings regularly.",
        ]
        return jsonify({"success": True, "recommendations": default})

# ──────────────────────────────────────────