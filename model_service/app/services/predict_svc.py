import time
import uuid
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.schemas import BankPredictionRequest, BankPredictionResponse
from app.repos import predict_repo

OPERATING_THRESHOLD = 0.07
MODEL_URI = "models:/bank-classifier@Production"

def make_prediction(db: Session, pipeline, request_data: BankPredictionRequest) -> BankPredictionResponse:
    """Handles data prep, ML inference, and orchestrates database logging."""
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    # 1. Prepare data for the model
    input_dict = request_data.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])
    
    # 2. Run the prediction
    try:
        proba = float(pipeline.predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
        
    prediction = 1 if proba >= OPERATING_THRESHOLD else 0
    latency = (time.perf_counter() - start_time) * 1000
    
    # 3. Log to Postgres using the Repo layer
    predict_repo.log_prediction(
        db=db,
        request_id=request_id,
        model_version=MODEL_URI,
        input_data=request_data.model_dump(), # Standard dump for clean JSON
        prediction=prediction,
        probability=proba,
        threshold_used=OPERATING_THRESHOLD,
        latency_ms=latency
    )
    
    # 4. Return formatted response
    return BankPredictionResponse(
        request_id=request_id,
        model_uri=MODEL_URI,
        threshold_used=OPERATING_THRESHOLD,
        prediction=prediction,
        probability=proba,
        latency_ms=latency
    )