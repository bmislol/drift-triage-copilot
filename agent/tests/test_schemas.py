import pytest
from pydantic import ValidationError
from app.schemas import DriftWebhook

def test_valid_drift_webhook():
    """Test that the schema accepts a properly formatted drift payload."""
    payload = {
        "active_version": "v1.0-prod",
        "drift_report": {
            "euribor3m": "critical",
            "age": "warning"
        }
    }
    
    webhook = DriftWebhook(**payload)
    
    assert webhook.active_version == "v1.0-prod"
    assert webhook.drift_report["euribor3m"] == "critical"
    assert "age" in webhook.drift_report

def test_invalid_drift_webhook():
    """Test that missing required fields correctly triggers a ValidationError."""
    with pytest.raises(ValidationError):
        # We are intentionally leaving out the required 'drift_report' dictionary
        DriftWebhook(active_version="v1.0-prod")