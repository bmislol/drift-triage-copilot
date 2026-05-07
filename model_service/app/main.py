import sys
from pathlib import Path
from contextlib import asynccontextmanager

import mlflow.sklearn
from fastapi import FastAPI

# 1. Path fixes (Adjusted because main.py is now inside app/)
CURRENT_DIR = Path(__file__).resolve().parent      # model_service/app
SERVICE_DIR = CURRENT_DIR.parent                   # model_service/
ROOT_DIR = SERVICE_DIR.parent                      # project-week5/

# Force BOTH into Python's system path so 'core' and 'app' can be found
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SERVICE_DIR))

from core.config import settings
from core.database import engine, Base
from app.api import predict_api, drift_api, promote_api

# Automatically create tables in Postgres
Base.metadata.create_all(bind=engine)

MODEL_URI = "models:/bank-classifier@Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Loads the model into memory before the API accepts requests."""
    print(f"Connecting to MLflow at {settings.mlflow_tracking_uri}...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    print(f"Loading model from {MODEL_URI}...")
    pipeline = mlflow.sklearn.load_model(MODEL_URI)
    
    # Store the loaded pipeline in app.state instead of a global variable
    app.state.pipeline = pipeline
    print("✅ Model loaded successfully!")
    
    yield
    
    print("Shutting down model service...")
    app.state.pipeline = None

app = FastAPI(title="Drift Triage Model Service", lifespan=lifespan)

# Register our endpoints
app.include_router(predict_api.router)
app.include_router(drift_api.router)
app.include_router(promote_api.router)