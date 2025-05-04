from flask import Blueprint, render_template

main = Blueprint('main', __name__)

# Dashboard route
@main.route('/')
def dashboard():
    return render_template('Index.html')

# Students page under Management section
@main.route('/management/students')
def students():
    return render_template('Students.html')

# Overall visualization page
@main.route('/visualization/overall')
def overall():
    return render_template('Overall.html')