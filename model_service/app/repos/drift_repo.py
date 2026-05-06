from sqlalchemy.orm import Session
from core.models import PredictionLog

def get_recent_predictions(db: Session, limit: int = 500):
    """Fetches the most recent predictions from the database to evaluate for drift."""
    # We order by ID or timestamp descending to get the freshest data
    return db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()