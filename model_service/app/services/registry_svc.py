from mlflow.tracking import MlflowClient
from fastapi import HTTPException
from app.schemas import PromoteResponse
from core.config import settings

MODEL_NAME = "bank-classifier"
MIN_RECALL = 0.75

def evaluate_and_promote(version: int) -> PromoteResponse:
    """Evaluates a model version's metrics and promotes it if it meets standards."""
    # 1. Explicitly tell the client where the MLflow server is
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    
    # 2. Fetch the model version details
    try:
        mv = client.get_model_version(name=MODEL_NAME, version=str(version))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found. Error: {str(e)}")
        
    run_id = mv.run_id
    
    # 3. Fetch the metrics from the run that created this model
    run = client.get_run(run_id)
    actual_recall = run.data.metrics.get("test_recall")
    
    if actual_recall is None:
        raise HTTPException(status_code=400, detail=f"Run {run_id} is missing 'test_recall' metric.")
        
    # 4. The Gatekeeper Logic
    if actual_recall >= MIN_RECALL:
        client.set_registered_model_alias(
            name=MODEL_NAME, 
            alias="Production", 
            version=str(version)
        )
        return PromoteResponse(
            version=version,
            status="promoted",
            recall=actual_recall,
            message=f"Model version {version} promoted to Production alias."
        )
    else:
        return PromoteResponse(
            version=version,
            status="rejected",
            recall=actual_recall,
            message=f"Model failed minimum recall threshold ({actual_recall} < {MIN_RECALL})."
        )