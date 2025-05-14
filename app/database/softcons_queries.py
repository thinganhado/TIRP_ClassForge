from app import db

class SoftConstraint(db.Model):
    __tablename__ = 'soft_constraints'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    # Hard constraints - commented out until database migration is performed
    # class_size = db.Column(db.Integer, default=30)
    # max_classes = db.Column(db.Integer, default=6)

    # Soft constraints
    gpa_penalty_weight = db.Column(db.Integer)
    wellbeing_penalty_weight = db.Column(db.Integer)
    bully_penalty_weight = db.Column(db.Integer)
    influence_std_weight = db.Column(db.Integer)
    isolated_std_weight = db.Column(db.Integer)

    min_friends_required = db.Column(db.Integer)
    friend_inclusion_weight = db.Column(db.Integer)
    friendship_balance_weight = db.Column(db.Integer)

    prioritize_academic = db.Column(db.Integer)
    prioritize_wellbeing = db.Column(db.Integer)
    prioritize_bullying = db.Column(db.Integer)
    prioritize_social_influence = db.Column(db.Integer)
    prioritize_friendship = db.Column(db.Integer)