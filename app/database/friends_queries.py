# app/database/friend_queries.py

import pandas as pd
import numpy as np
import json
from sqlalchemy import text
from app import db

EXCLUDE_STUDENT_ID = '32561'

def fetch_friend_edges():
    sql = text("""
        SELECT
            source_student_id AS source,
            target_student_id AS target
        FROM net_friends
    """)
    return db.session.execute(sql).fetchall()

def fetch_participants():
    sql = text("""
        SELECT student_id, first_name, last_name
        FROM participants
    """)
    rows = db.session.execute(sql).fetchall()
    return {
        row.student_id: f"{row.first_name} {row.last_name}"
        for row in rows
    }

def fetch_allocations():
    sql = text("""
        SELECT student_id, class_id
        FROM allocations
    """)
    rows = db.session.execute(sql).fetchall()
    return {
        row.student_id: row.class_id
        for row in rows
    }

def build_friendship_graph_json():
    """
    Return Cytoscape-compatible JSON (nodes + edges) for friendship graph.
    Filters out a specific EXCLUDE_STUDENT_ID.
    """
    edges_raw     = fetch_friend_edges()
    participants  = fetch_participants()
    allocations   = fetch_allocations()

    class_color_map = {
        0: '#3a0ca3', 1: '#1a759f', 2: '#34a0a4',
        3: '#076c893', 4: '#b5e48c', 5: '#f9c74f'
    }

    # ─── Build degree map ───────────────────────────────
    degree_count = {}
    for row in edges_raw:
        for sid in [row.source, row.target]:
            if str(sid) != EXCLUDE_STUDENT_ID:
                degree_count[sid] = degree_count.get(sid, 0) + 1

    # ─── Build nodes ────────────────────────────────────
    nodes = []
    for sid, deg in degree_count.items():
        if str(sid) == EXCLUDE_STUDENT_ID:
            continue

        label    = participants.get(sid, "Unknown")
        class_id = allocations.get(sid)
        color    = class_color_map.get(
            int(class_id) % len(class_color_map) if class_id is not None else -1,
            "#cccccc"
        )

        nodes.append({
            "data": {
                "id": str(sid),
                "label": label,
                "color": color,
                "size": int(deg),
                "class_id": int(class_id) if class_id is not None else None
            }
        })

    # ─── Build edges ────────────────────────────────────
    edges = []
    for row in edges_raw:
        s, t = str(row.source), str(row.target)
        if s == t or EXCLUDE_STUDENT_ID in (s, t):
            continue
        edges.append({"data": {"source": s, "target": t}})

    return json.dumps(nodes + edges)
