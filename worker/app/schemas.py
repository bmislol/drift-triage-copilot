from pydantic import BaseModel, Field
from typing import Any, Dict

class JobPayload(BaseModel):
    job_id: str
    type: str = Field(..., description="Type of job: retrain, rollback, or replay_test")
    payload: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    error_reason: str | None = None