import time
import uuid
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

import sys
from pathlib import Path

# 1. Get the absolute paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# 2. Force BOTH into Python's system path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(CURRENT_DIR))

from core.config import settings
from core.database import get_db, engine, Base
from core.models import PredictionLog
from model_service.shemas import BankPredictionRequest, BankPredictionResponse

# Automatically create tables in Postgres if they don't exist yet
Base.metadata.create_all(bind=engine)

# Global variables to hold our model in memory
pipeline = None
OPERATING_THRESHOLD = 0.07  # The exact threshold we tuned in train.py
MODEL_URI = "models:/bank-classifier@Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Loads the model into memory before the API accepts requests."""
    global pipeline
    print(f"Connecting to MLflow at {settings.mlflow_tracking_uri}...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    print(f"Loading model from {MODEL_URI}...")
    pipeline = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Model loaded successfully!")
    yield
    print("Shutting down model service...")

app = FastAPI(title="Drift Triage Model Service", lifespan=lifespan)

@app.post("/predict", response_model=BankPredictionResponse)
def predict(request: BankPredictionRequest, db: Session = Depends(get_db)):
    """Accepts a customer record, predicts churn, and logs the result to Postgres."""
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    # 1. Prepare data for the model 
    # Use by_alias=True so fields like 'emp_var_rate' become 'emp.var.rate' for the model
    input_dict = request.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])
    
    # 2. Run the prediction
    try:
        proba = float(pipeline.predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
        
    prediction = 1 if proba >= OPERATING_THRESHOLD else 0
    latency = (time.perf_counter() - start_time) * 1000
    
    # 3. Log everything to Postgres
    # Use standard dump here so JSON keys stay clean in the database
    log_entry = PredictionLog(
        request_id=request_id,
        model_version=MODEL_URI,
        input_data=request.model_dump(), 
        prediction=prediction,
        probability=proba,
        threshold_used=OPERATING_THRESHOLD,
        latency_ms=latency
    )
    db.add(log_entry)
    db.commit()
    
    # 4. Return the response to the caller
    return BankPredictionResponse(
        request_id=request_id,
        model_uri=MODEL_URI,
        threshold_used=OPERATING_THRESHOLD,
        prediction=prediction,
        probability=proba,
        latency_ms=latency
    )