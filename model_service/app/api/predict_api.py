from fastapi import APIRouter, Depends, Request, BackgroundTasks
from sqlalchemy.orm import Session

from core.database import get_db
from app.schemas import BankPredictionRequest, BankPredictionResponse
from app.services import predict_svc

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=BankPredictionResponse)
def predict(
    request_data: BankPredictionRequest, 
    request: Request, # Allows us to access app.state.pipeline
    background_tasks: BackgroundTasks, # <-- Inject background tasks here
    db: Session = Depends(get_db)
):
    """Accepts a customer record, predicts churn, and logs the result to Postgres."""
    # Retrieve the model we loaded into memory during startup
    pipeline = request.app.state.pipeline
    
    # Let the service handle the business logic
    return predict_svc.make_prediction(db, pipeline, request_data, background_tasks)