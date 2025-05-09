from app import db

class SoftConstraint(db.Model):
    __tablename__ = 'soft_constraints'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    gpa_penalty = db.Column(db.Integer)
    wellbeing_penalty = db.Column(db.Integer)
    bully_penalty = db.Column(db.Integer)
    influence_weight = db.Column(db.Integer)
    isolation_weight = db.Column(db.Integer)
    friend_score_toggle = db.Column(db.Boolean)
    friend_balance_toggle = db.Column(db.Boolean)
    min_friends = db.Column(db.Integer)
    priority_order = db.Column(db.Text)  # Store CSV string or JSON string
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
