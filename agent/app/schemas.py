from pydantic import BaseModel
from typing import Dict, Any

class DriftWebhook(BaseModel):
    """Schema for incoming drift events from the model service."""
    active_version: str
    drift_report: Dict[str, Any]

class ApprovalRequest(BaseModel):
    """Schema for human-in-the-loop approval signals."""
    approved: bool