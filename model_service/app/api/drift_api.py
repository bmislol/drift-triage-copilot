from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from app.schemas import DriftReportResponse
from app.services import drift_svc

router = APIRouter(tags=["Drift Monitoring"])

@router.get("/drift-report", response_model=DriftReportResponse)
def get_drift_report(limit: int = 500, db: Session = Depends(get_db)):
    """Generates a real-time drift report using the last N predictions."""
    return drift_svc.generate_drift_report(db=db, limit=limit)