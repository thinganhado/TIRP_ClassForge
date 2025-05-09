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
        gpa_penalty = int(request.form.get("gpa_penalty_weight", 30))
        wellbeing_penalty = int(request.form.get("wellbeing_penalty_weight", 50))
        bully_penalty = int(request.form.get("bully_penalty_weight", 60))
        influence_weight = int(request.form.get("influence_std_weight", 60))
        isolation_weight = int(request.form.get("isolated_std_weight", 60))
        friend_score_toggle = request.form.get("friend_score_toggle", "true") == "true"
        friend_balance_toggle = request.form.get("friend_balance_toggle", "true") == "true"
        min_friends = int(request.form.get("min_friends", 1))
        priority_csv = request.form.get("priority_order", "")
        priority_list = priority_csv.split(",") if priority_csv else []

        # --- Store in SQL ---
        new_entry = SoftConstraint(
            gpa_penalty=gpa_penalty,
            wellbeing_penalty=wellbeing_penalty,
            bully_penalty=bully_penalty,
            influence_weight=influence_weight,
            isolation_weight=isolation_weight,
            friend_score_toggle=friend_score_toggle,
            friend_balance_toggle=friend_balance_toggle,
            min_friends=min_friends,
            priority_order=",".join(priority_list)
        )
        db.session.add(new_entry)
        db.session.commit()

        # --- Optional: Save JSON too ---
        constraints = {
            "gpa_penalty": gpa_penalty,
            "wellbeing_penalty": wellbeing_penalty,
            "bully_penalty": bully_penalty,
            "influence_weight": influence_weight,
            "isolation_weight": isolation_weight,
            "friend_score_toggle": friend_score_toggle,
            "friend_balance_toggle": friend_balance_toggle,
            "min_friends": min_friends,
            "priority_order": priority_list
        }
        with open("soft_constraints_config.json", "w") as f:
            json.dump(constraints, f, indent=2)

        return redirect(url_for("main.customisation_loading"))

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