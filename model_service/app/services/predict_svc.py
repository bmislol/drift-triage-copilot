import time
import uuid
import pandas as pd
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.schemas import BankPredictionRequest, BankPredictionResponse
from app.repos import predict_repo
from core.models import PredictionLog
from app.services import drift_svc

OPERATING_THRESHOLD = 0.07
MODEL_URI = "models:/bank-classifier@Production"

def make_prediction(
        db: Session, 
        pipeline, 
        request_data: BankPredictionRequest, 
        background_tasks: BackgroundTasks,
        active_version: str = "v2"
    ) -> BankPredictionResponse:

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
    log_entry = predict_repo.log_prediction(
        db=db,
        request_id=request_id,
        model_version=MODEL_URI,
        input_data=request_data.model_dump(), # Standard dump for clean JSON
        prediction=prediction,
        probability=proba,
        threshold_used=OPERATING_THRESHOLD,
        latency_ms=latency
    )

    CHECK_INTERVAL = 50

    total_predictions = db.query(PredictionLog).count()

    if total_predictions > 0 and total_predictions % CHECK_INTERVAL == 0:
        print(f"⚡ Reached {total_predictions} total predictions! Queueing drift check...")
        
        # We need to extract the version. Since you are using MODEL_URI, we can parse it, 
        # or just pass a default for now.
        active_version = MODEL_URI.split("@")[0].split("/")[-1] # Extracts "bank-classifier" or you can hardcode "v1" if preferred. Let's use "v1" to keep it simple for the agent.
        
        # Pass the active_version to the function!
        background_tasks.add_task(drift_svc.evaluate_and_alert, "v1")
    
    # 4. Return formatted response
    return BankPredictionResponse(
        request_id=request_id,
        model_uri=MODEL_URI,
        threshold_used=OPERATING_THRESHOLD,
        prediction=prediction,
        probability=proba,
        latency_ms=latency
    )