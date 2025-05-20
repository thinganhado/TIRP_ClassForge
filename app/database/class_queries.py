from sqlalchemy import text
from app import db
from collections import defaultdict
from typing import List, Dict, Any

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

def fetch_all_classes() -> List[Dict[str, Any]]:
    """
    Return a list of all classes, each with its roster of students.
    Each student is represented by { id, name } where name = "First Last".
    """
    sql = text("""
        SELECT
            COALESCE(a.class_id, 'Unassigned') AS class_id,
            p.student_id    AS id,
            p.first_name,
            p.last_name
        FROM participants p
        LEFT JOIN allocations a
          ON p.student_id = a.student_id
        WHERE p.student_id <> 'student_id'
        ORDER BY class_id, p.last_name, p.first_name
    """)
    rows = db.session.execute(sql).fetchall()

    # group into a dict[class_id] → list of students
    classes_map = defaultdict(list)
    for r in rows:
        full_name = f"{r.first_name} {r.last_name}"
        classes_map[r.class_id].append({
            "id":   r.id,
            "name": full_name
        })

    # build final list, sorted by class_id
    result = []
    for cls in sorted(classes_map.keys()):
        result.append({
            "name":     cls,
            "students": classes_map[cls]
        })
    return result

def fetch_disrespect_for_class(class_id):
    sql = text("""
      SELECT
        d.source_student_id AS source,
        d.target_student_id AS target
      FROM net_disrespect1 d
      JOIN allocations a1
        ON d.source_student_id = a1.student_id AND a1.class_id = :cid
      JOIN allocations a2
        ON d.target_student_id = a2.student_id AND a2.class_id = :cid
    """)
    rows = db.session.execute(sql, {"cid": class_id}).fetchall()
    return [{"source": r.source, "target": r.target} for r in rows]

def get_cohort_averages():
    """
    Return a tuple (avg_gpa, avg_well) across all students in the cohort.
    """
    sql = text("""
        SELECT
          AVG(p.perc_academic)    AS avg_gpa,
          AVG(r.school_support_engage6) AS avg_well
        FROM participants p
        LEFT JOIN responses r
          ON r.student_id = p.student_id
    """)
    row = db.session.execute(sql).fetchone()
    return (
        float(row.avg_gpa  or 0.0),
        float(row.avg_well or 0.0)
    )

def get_class_metrics(class_id):
    """
    Return a dict with:
      - avg_gpa        (float)
      - avg_well       (float)
      - num_conflicts  (int)
    for the given class_id.
    """
    # 1) Average GPA in this class
    sql_gpa = text("""
        SELECT AVG(p.perc_academic) AS avg_gpa
        FROM participants p
        JOIN allocations a
          ON p.student_id = a.student_id
        WHERE a.class_id = :cid
    """)

    # 2) Average well-being in this class
    sql_well = text("""
        SELECT AVG(r.school_support_engage6) AS avg_well
        FROM responses r
        JOIN allocations a
          ON r.student_id = a.student_id
        WHERE a.class_id = :cid
    """)

    # 3) Number of disrespect edges (both ends in the class)
    sql_conf = text("""
        SELECT COUNT(*) AS num_conflicts
        FROM net_disrespect1 d
        JOIN allocations a1
          ON d.source_student_id = a1.student_id
         AND a1.class_id = :cid
        JOIN allocations a2
          ON d.target_student_id = a2.student_id
         AND a2.class_id = :cid
    """)

    r_gpa  = db.session.execute(sql_gpa,  {"cid": class_id}).fetchone()
    r_well = db.session.execute(sql_well, {"cid": class_id}).fetchone()
    r_conf = db.session.execute(sql_conf, {"cid": class_id}).fetchone()

    return {
        "avg_gpa":        float(r_gpa.avg_gpa       or 0.0),
        "avg_well":       float(r_well.avg_well     or 0.0),
        "num_conflicts":  int(r_conf.num_conflicts or 0)
    }

def fetch_classes_summary():
    sql = text("""
      SELECT a.class_id, COUNT(*) AS count
      FROM allocations a
      WHERE a.class_id IS NOT NULL
      GROUP BY a.class_id
      ORDER BY a.class_id
    """)
    rows = db.session.execute(sql).fetchall()
    return [
      {'class_id': r.class_id, 'count': int(r.count)}
      for r in rows
    ]

def fetch_conflict_pairs_per_class() -> List[Dict[str, Any]]:
    """
    Return [{ class_id: '2',
              pairs: [ {bully:{id,name}, victim:{id,name}}, … ] }, … ]

    Only disrespect edges where bully *and* victim are in the same class
    are included (same criterion used in Classes.html).
    """
    sql = text("""
        WITH intra AS (
            SELECT a1.class_id,
                   d.source_student_id AS bully_id,
                   d.target_student_id AS victim_id
            FROM   net_disrespect1 d
            JOIN   allocations a1 ON a1.student_id = d.source_student_id
            JOIN   allocations a2 ON a2.student_id = d.target_student_id
            WHERE  a1.class_id = a2.class_id           -- same class only
        )
        SELECT  i.class_id,
                b.student_id  AS bully_id,
                b.first_name  AS bully_fn,
                b.last_name   AS bully_ln,
                v.student_id  AS vict_id,
                v.first_name  AS vict_fn,
                v.last_name   AS vict_ln
        FROM    intra i
        JOIN    participants b ON b.student_id = i.bully_id
        JOIN    participants v ON v.student_id = i.victim_id
        ORDER   BY i.class_id, bully_ln, bully_fn
    """)

    rows = db.session.execute(sql).fetchall()

    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_class[r.class_id].append({
            "bully":  {"id": str(r.bully_id),
                       "name": f"{r.bully_fn} {r.bully_ln}"},
            "victim": {"id": str(r.vict_id),
                       "name": f"{r.vict_fn} {r.vict_ln}"}
        })

    return [{"class_id": cid, "pairs": pairs}
            for cid, pairs in sorted(by_class.items())]