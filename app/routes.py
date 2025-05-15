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
)
from app.database.class_queries import (
    fetch_unique_classes,
    fetch_all_classes,
    fetch_disrespect_for_class,
    get_cohort_averages,
    get_class_metrics,
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

@main.route("/customisation/specifications")
def specification():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("specification.html", session_id=session["session_id"])

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
# **identical in both versions – kept as-is**
# … (same code as before) …

@main.route("/customisation/loading")
def customisation_loading():
    return render_template("customisation_loading.html",
                           from_set_priorities=True)   # keep v1 flag

# run_allocation unchanged
# … (same code as before) …

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