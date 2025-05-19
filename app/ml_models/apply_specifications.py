import random
import os, sys                   
sys.path.insert(0,                    
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sqlalchemy import text
from app import create_app, db
from app.database.spec_endpoint import HardConstraint

MAX_PER_CLASS = 30                 # capacity

app = create_app()

def _latest_constraints():
    with app.app_context():
        rec = (HardConstraint.query
               .order_by(HardConstraint.id.desc())
               .first())
    if not rec:
        return [], []
    return rec.separate_pairs, rec.forced_moves

def _class_sizes():
    sql = text("SELECT class_id, COUNT(*) AS n FROM allocations GROUP BY class_id")
    with app.app_context():
        rows = db.session.execute(sql).fetchall()
    return {int(r.class_id): int(r.n) for r in rows}

def _current_class(student_id):
    sql = text("SELECT class_id FROM allocations WHERE student_id = :sid")
    with app.app_context():
        row = db.session.execute(sql, {"sid": str(student_id)}).fetchone()
    return None if not row else int(row.class_id)

def _conflict_count(student_id, target_class):
    """# of ‘net_disrespect’ edges student would have *in* target class."""
    sql = text("""
        SELECT COUNT(*) AS n
          FROM net_disrespect
          JOIN allocations a1 ON a1.student_id = net_disrespect.source_student_id
          JOIN allocations a2 ON a2.student_id = net_disrespect.target_student_id
         WHERE (net_disrespect.source_student_id = :sid OR net_disrespect.target_student_id = :sid)
           AND a1.class_id = a2.class_id
           AND a1.class_id = :cls
    """)
    with app.app_context():
        row = db.session.execute(sql, {"sid": str(student_id),
                                       "cls": target_class}).fetchone()
    return int(row.n)

def main():
    sep_pairs, forced_moves = _latest_constraints()
    if not (sep_pairs or forced_moves):
        print("No hard constraints – nothing to do.")
        return 0

    sizes = _class_sizes()

    with app.app_context():
        # ---------- forced moves ----------
        for item in forced_moves:
            sid = str(item["sid"])
            tgt = int(item["cls"])
            cur = _current_class(sid)
            if cur == tgt:
                continue
            if sizes.get(tgt,0) >= MAX_PER_CLASS:
                print(f"[skip] class {tgt} full – cannot move {sid}", file=sys.stderr)
                continue
            db.session.execute(text("UPDATE allocations SET class_id=:c WHERE student_id=:s"),
                               {"c": tgt, "s": sid})
            sizes[cur]  -= 1
            sizes[tgt]  = sizes.get(tgt,0) + 1

        # ---------- separations ----------
        for pair in sep_pairs:
            s1, s2 = map(str, pair)
            cls1   = _current_class(s1)
            cls2   = _current_class(s2)
            if cls1 != cls2:        # already separated
                continue
            # try moving the *first* student
            cand = [c for c in range(6) if c!=cls1 and sizes.get(c,0)<MAX_PER_CLASS]
            if not cand:
                print(f"[warn] nowhere to move {s1}", file=sys.stderr)
                continue
            # compute conflict score per candidate
            scores = {c: _conflict_count(s1, c) for c in cand}
            best   = min(scores.values())
            best_cls = random.choice([c for c,v in scores.items() if v==best])
            db.session.execute(text("UPDATE allocations SET class_id=:c WHERE student_id=:s"),
                               {"c": best_cls, "s": s1})
            sizes[cls1]  -= 1
            sizes[best_cls] += 1

        db.session.commit()
    print("Specifications applied.")
    return 0

if __name__ == "__main__":
    sys.exit(main())