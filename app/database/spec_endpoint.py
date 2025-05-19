from app import db
from sqlalchemy.dialects.mysql import JSON

class HardConstraint(db.Model):
    __tablename__ = "hard_constraints"

    id             = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    created_at     = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    separate_pairs = db.Column(JSON, nullable=False, server_default="[]")
    forced_moves   = db.Column(JSON, nullable=False, server_default="[]")
