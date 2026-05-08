from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from core.models import Investigation
from sqlalchemy import desc

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Returns high-level stats for the dashboard header."""
    total = db.query(Investigation).count()
    critical = db.query(Investigation).filter(Investigation.severity == "critical").count()
    pending = db.query(Investigation).filter(Investigation.status == "awaiting_approval").count()
    
    return {
        "total_investigations": total,
        "critical_drifts": critical,
        "awaiting_approval": pending
    }

@router.get("/investigations")
def list_investigations(db: Session = Depends(get_db), limit: int = 10):
    """Returns the most recent drift investigations."""
    return db.query(Investigation).order_by(desc(Investigation.created_at)).limit(limit).all()

@router.get("/investigations/{thread_id}")
def get_investigation_details(thread_id: str, db: Session = Depends(get_db)):
    """Returns the full context for a single investigation."""
    inv = db.query(Investigation).filter(Investigation.thread_id == thread_id).first()
    return inv