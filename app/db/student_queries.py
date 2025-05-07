# app/db/student_queries.py

from app import db  # This assumes SQLAlchemy is initialized in app/__init__.py
from sqlalchemy import text

def fetch_all_students():
    query = text("""
        SELECT 
            p.student_id, 
            p.first_name, 
            p.last_name, 
            p.email, 
            p.house, 
            a.class_id
        FROM tirp_participants p
        LEFT JOIN tirp_allocations a ON p.student_id = a.student_id
    """)
    result = db.session.execute(query)
    students = [dict(row) for row in result.fetchall()]
    return students