from flask import Blueprint, render_template, jsonify
from app.db.student_queries import fetch_all_students

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
    students = fetch_all_students()
    return jsonify(students)

# Overall visualization page
@main.route('/visualization/overall')
def overall():
    return render_template('Overall.html')

# Individual visualization page
@main.route('/visualization/individual')
def individual():
    return render_template('Individual.html')

# Customisation main page
@main.route('/customisation')
def customisation():
    return render_template('customisation/customisation.html')

# Set priorities page
@main.route('/customisation/set-priorities')
def set_priorities():
    return render_template('customisation/set_priorities.html')

# Specification page
@main.route('/customisation/specification')
def specification():
    return render_template('customisation/specification.html')

# AI Assistant page
@main.route('/customisation/ai-assistant')
def ai_assistant():
    return render_template('customisation/ai_assistant.html')

# History page
@main.route('/customisation/history')
def history():
    return render_template('customisation/history.html')