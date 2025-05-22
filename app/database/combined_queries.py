# app/database/disrespect_queries.py

import pandas as pd
import numpy as np
import json
from sqlalchemy import text
from app import db

def fetch_all_edges():
    all_edges = []

    for table, relation in [
        ("net_friends", "friends"),
        ("net_advice", "advice"),
        ("net_feedback", "feedback"),
        ("net_influential", "influential"),
        ("net_moretime", "moretime"),
        ("net_disrespect", "disrespect")
    ]:
        sql = text(f"""
            SELECT source_student_id AS source, target_student_id AS target
            FROM {table}
        """)
        rows = db.session.execute(sql).fetchall()
        for row in rows:
            all_edges.append({
                "data": {
                    "source": str(row.source),
                    "target": str(row.target),
                    "type": relation
                }
            })

    return all_edges

def fetch_participants():
    sql = text("""
        SELECT student_id, first_name, last_name
        FROM participants
    """)
    rows = db.session.execute(sql).fetchall()
    return {
        str(row.student_id): f"{row.first_name} {row.last_name}"
        for row in rows
    }

def fetch_allocations():
    sql = text("""
        SELECT student_id, class_id
        FROM allocations
    """)
    rows = db.session.execute(sql).fetchall()
    return {
        str(row.student_id): row.class_id
        for row in rows
    }

def build_combined_graph_json():
    edges_raw = fetch_all_edges()
    participants = fetch_participants()
    allocations = fetch_allocations()

    # Unique student IDs from all edges
    student_ids = set()
    for edge in edges_raw:
        student_ids.add(edge["data"]["source"])
        student_ids.add(edge["data"]["target"])

    # Color palette by class
    class_color_map = {
        0: '#ade8f4', 1: '#48cae4', 2: '#0096c7',
        3: '#0077b6', 4: '#5390d9', 5: '#3a0ca3'
    }

    nodes = []
    for sid in student_ids:
        label = participants.get(sid, "Unknown")
        class_id = allocations.get(sid)
        color = class_color_map.get(
            int(class_id) % len(class_color_map) if class_id is not None else -1,
            "#cccccc"
        )

        nodes.append({
            "data": {
                "id": sid,
                "label": label,
                "color": color,
                "class_id": int(class_id) if class_id is not None else None
            }
        })

    return json.dumps(nodes + edges_raw)
