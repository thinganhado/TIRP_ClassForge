from flask import Blueprint, render_template, jsonify
from app.database.student_queries import fetch_all_students, fetch_student_details
from app.database.class_queries    import fetch_unique_classes

main = Blueprint('main', __name__)

# Dashboard route
@main.route('/')
def dashboard():
    return render_template('Index.html')

# Students page under Management section
@main.route('/students')
def students():
    return render_template('Students.html')

@main.route("/api/students")
def api_students():
    return jsonify(fetch_all_students())

# app/routes.py  (add after /api/students)
@main.route("/api/students/<int:sid>")
def api_student_detail(sid):
    return jsonify(fetch_student_details(sid))

@main.route("/api/classes")          # ← NEW
def api_classes():
    return jsonify(fetch_unique_classes())

# Overall visualization page
@main.route('/visualization/overall')
def overall():
    return render_template('Overall.html')

# Individual visualization page
@main.route('/visualization/individual')
def individual():
    return render_template('Individual.html')