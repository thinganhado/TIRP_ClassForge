from sqlalchemy import text
from app import db

def fetch_unique_classes():
    query = text("""
        SELECT DISTINCT class_id
        FROM allocations
        WHERE class_id IS NOT NULL
        ORDER BY class_id;
    """)

    result = db.session.execute(query)
    # each row is a tuple → row[0]
    return [r.class_id for r in result]