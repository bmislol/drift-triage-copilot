# core/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime, timezone
from core.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    request_id = Column(String, unique=True, index=True)
    model_version = Column(String, index=True)
    
    # Store the 19 raw bank features as JSON so it's flexible
    input_data = Column(JSON) 
    
    # The output of the model
    prediction = Column(Integer)
    probability = Column(Float)
    threshold_used = Column(Float)
    latency_ms = Column(Float)

class Investigation(Base):
    __tablename__ = "investigations"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)  # LangGraph thread ID
    status = Column(String, default="open")  # open, awaiting_approval, resolved
    severity = Column(String)  # none, warning, critical
    recommended_action = Column(String) 
    summary = Column(String)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))