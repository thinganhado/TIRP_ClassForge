from flask import Blueprint, render_template
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