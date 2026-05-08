import json
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session

from app.repos import drift_repo
from app.schemas import DriftReportResponse, FeatureDrift

import requests
from core.database import SessionLocal
from core.config import settings

# Navigate up from app/services/ to the root data folder
STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "reference_stats.json"

def load_reference_stats() -> dict:
    with open(STATS_PATH, "r") as f:
        return json.load(f)

def calculate_psi(expected_proportions, actual_proportions) -> tuple[float, str]:
    """Calculates Population Stability Index for numerical features."""
    # Use a tiny epsilon to prevent division by zero or log(0)
    epsilon = 1e-4
    expected = np.clip(expected_proportions, epsilon, None)
    actual = np.clip(actual_proportions, epsilon, None)
    
    # Normalize to ensure sums equal 1
    expected = expected / np.sum(expected)
    actual = actual / np.sum(actual)
    
    psi_values = (actual - expected) * np.log(actual / expected)
    psi_score = float(np.sum(psi_values))
    
    # Standard PSI thresholds
    if psi_score < 0.1:
        severity = "none"
    elif psi_score < 0.2:
        severity = "warning"
    else:
        severity = "critical"
        
    return psi_score, severity

def calculate_chi2(expected_proportions, actual_counts) -> tuple[float, str]:
    """Calculates Chi-Squared p-value for categorical features."""
    total_actual = sum(actual_counts)
    if total_actual == 0:
        return 1.0, "none"
        
    expected_counts = [prop * total_actual for prop in expected_proportions]
    
    # Run the scipy chi2 test
    _, p_value = chisquare(f_obs=actual_counts, f_exp=expected_counts)
    p_val = float(p_value)
    
    # Standard p-value thresholds (0.05 significance)
    if p_val > 0.05:
        severity = "none"
    elif p_val > 0.01:
        severity = "warning"
    else:
        severity = "critical"
        
    return p_val, severity

def generate_drift_report(db: Session, limit: int = 500) -> DriftReportResponse:
    """Orchestrates the drift calculation across all logged features."""
    reference_stats = load_reference_stats()
    recent_logs = drift_repo.get_recent_predictions(db, limit)
    
    if not recent_logs:
        # Return a clean baseline if the database is empty
        return DriftReportResponse(
            timestamp=datetime.utcnow(),
            record_count=0,
            overall_severity="none",
            features={}
        )
        
    # Extract the JSON input payload into a Pandas DataFrame
    data = [log.input_data for log in recent_logs]
    df = pd.DataFrame(data)
    
    feature_results = {}
    overall_status = "none"
    
    # Iterate dynamically through the flat JSON structure
    for feature, stats in reference_stats.items():
        if feature not in df.columns:
            continue
            
        # 1. Process Numerical Features (PSI)
        if stats.get("type") == "numeric":
            bins = stats.get("histogram_edges")
            
            # Convert raw counts to percentages (proportions)
            counts = np.array(stats.get("histogram_counts"))
            expected_props = counts / counts.sum()
            
            # pd.cut groups our actual data into the reference bins
            # include_lowest=True ensures the minimum values aren't dropped
            actual_counts = pd.cut(df[feature], bins=bins, include_lowest=True).value_counts(sort=False).values
            actual_props = actual_counts / len(df)
            
            score, sev = calculate_psi(expected_props, actual_props)
            feature_results[feature] = FeatureDrift(metric="psi", score=score, severity=sev)

        # 2. Process Categorical Features (Chi2)
        elif stats.get("type") == "categorical":
            expected_dict = stats.get("distribution")
            categories = list(expected_dict.keys())
            expected_props = [expected_dict[c] for c in categories]
            
            # Count actual occurrences for these specific categories
            actual_val_counts = df[feature].value_counts()
            actual_counts = [actual_val_counts.get(c, 0) for c in categories]
            
            p_val, sev = calculate_chi2(expected_props, actual_counts)
            feature_results[feature] = FeatureDrift(metric="chi2", score=p_val, severity=sev)

        # 3. Update the Overall Severity Level
        if feature in feature_results:
            current_sev = feature_results[feature].severity
            if current_sev == "critical":
                overall_status = "critical"
            elif current_sev == "warning" and overall_status == "none":
                overall_status = "warning"

    return DriftReportResponse(
        timestamp=datetime.utcnow(),
        record_count=len(df),
        overall_severity=overall_status,
        features=feature_results
    )

def evaluate_and_alert(active_version: str):
    """Runs in the background. Calculates drift and fires webhook if needed."""
    db = SessionLocal()
    try:
        print("🔍 Running scheduled background drift check...")
        report = generate_drift_report(db, limit=500)
        
        if report.overall_severity in ["warning", "critical"]:
            print(f"⚠️ {report.overall_severity.upper()} drift detected! Formatting webhook...")
            
            # 1. Map your Pydantic report into the exact JSON format the Agent expects
            drift_payload = {}
            for feature_name, feature_obj in report.features.items():
                if feature_obj.severity in ["warning", "critical"]:
                    drift_payload[feature_name] = {
                        "status": feature_obj.severity,
                        "psi": getattr(feature_obj, 'psi_score', None),
                        "chi2_p": getattr(feature_obj, 'chi2_pvalue', None)
                    }
            
            # 2. Construct the final payload using the dynamic active_version
            payload = {
                "active_version": active_version,
                "drift_report": drift_payload
            }
            
            # 3. Pull the correct URL from your core config
            webhook_url = "http://localhost:8001/investigations/webhook/drift"
            
            try:
                requests.post(webhook_url, json=payload, timeout=30)
                print("✅ Webhook successfully delivered to Triage Agent.")
            except Exception as e:
                print(f"❌ Failed to reach Agent Webhook: {e}")
        else:
            print("✅ Drift check passed. Distributions look normal.")
    finally:
        db.close()