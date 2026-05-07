# model_service/schemas.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any
from datetime import datetime

class BankPredictionRequest(BaseModel):
    """A validated prediction request for the UCI Bank Marketing dataset."""
    
    # Forbid extra fields and strip whitespace to keep data clean
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, populate_by_name=True)

    age: int = Field(ge=16, le=120)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    
    # 'duration' is deliberately missing here per project guidelines!
    
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: str
    
    # Use aliases for columns with dots in the CSV
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

class BankPredictionResponse(BaseModel):
    """The standard response sent back to the user."""
    request_id: str
    model_uri: str
    threshold_used: float
    prediction: int
    probability: float
    latency_ms: float

class FeatureDrift(BaseModel):
    metric: str      # "psi" or "chi2"
    score: float     # The calculated value (PSI score or Chi2 p-value)
    severity: str    # "none", "warning", "critical"
    details: Dict[str, Any] = {} # Extra context if needed

class DriftReportResponse(BaseModel):
    timestamp: datetime
    record_count: int
    overall_severity: str
    features: Dict[str, FeatureDrift]

class PromoteResponse(BaseModel):
    version: int
    status: str          # "promoted" or "rejected"
    recall: float
    message: str