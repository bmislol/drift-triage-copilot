from sqlalchemy.orm import Session
from core.models import PredictionLog

def log_prediction(
    db: Session, 
    request_id: str, 
    model_version: str, 
    input_data: dict, 
    prediction: int, 
    probability: float, 
    threshold_used: float, 
    latency_ms: float
):
    """Saves the prediction event to the Postgres database."""
    log_entry = PredictionLog(
        request_id=request_id,
        model_version=model_version,
        input_data=input_data, 
        prediction=prediction,
        probability=probability,
        threshold_used=threshold_used,
        latency_ms=latency_ms
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    
    return log_entry