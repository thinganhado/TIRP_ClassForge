# app/routes.py
from flask import Blueprint, render_template, jsonify
from app.database.student_queries import (
    fetch_all_students,
    fetch_student_details,
)
from app.database.class_queries import fetch_unique_classes

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