# app/models.py
from app import db
from sqlalchemy.dialects.mysql import JSON

class HardConstraint(db.Model):
    __tablename__ = "hard_constraints"

    id               = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    timestamp        = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    # JSON arrays (see previous message)
    separate_pairs   = db.Column(JSON,  nullable=False, server_default="[]")
    move_requests    = db.Column(JSON,  nullable=False, server_default="[]")
