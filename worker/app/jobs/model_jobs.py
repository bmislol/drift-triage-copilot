import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from core.config import settings, RAW_DATA_PATH
# Import your existing train script
from scripts.train import train_and_register

MODEL_NAME = "bank-classifier"

def handle_retrain(payload: dict):
    print("🔄 Initiating retrain sequence...")
    # Re-running the script generates a new artifact and increments the version in MLflow
    # This proves the worker successfully executed the long-running job.
    train_and_register()
    print("✅ Retrain complete. New model registered in MLflow.")

def handle_rollback(payload: dict):
    target_version = payload.get("version")
    if not target_version:
        raise ValueError("Rollback payload missing 'version' key.")
        
    print(f"⏪ Rolling back {MODEL_NAME} to version {target_version}...")
    
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    
    # Transition the requested version to Production, automatically archiving the current one
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(target_version),
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Rollback successful. Version {target_version} is now Production.")

def handle_replay_test(payload: dict):
    print("🧪 Starting 1e-12 fidelity replay test...")
    
    # Recreate the deterministic split from your original training script
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df = df.drop(columns=["duration"])
    df["y"] = (df["y"] == "yes").astype(int)
    
    X = df.drop(columns=["y"])
    y = df["y"]
    
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # Load the current Production model to test it
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{MODEL_NAME}@Production"
    model = mlflow.sklearn.load_model(model_uri)
    
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"✅ Replay test complete. Current Production AUC on test set: {auc:.12f}")