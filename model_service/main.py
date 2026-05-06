from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import os
import pandas as pd
from schemas import BankPredictionRequest, PredictionResponse

# Globals populated at startup
ml_model = None
operating_threshold: float = 0.5


class DummyModel:
    """Fallback for local development before the model is trained and registered."""
    def predict(self, df):
        return [0] * len(df)

    def predict_proba(self, df):
        return [[0.8, 0.2]] * len(df)


def _load_threshold(model_uri: str) -> float:
    """Read the tuned operating threshold from the model card stored in the run."""
    try:
        client = mlflow.tracking.MlflowClient()
        # model_uri is "models:/bank-classifier@Production" — extract name + alias
        model_name = model_uri.split("/")[1].split("@")[0]
        alias      = model_uri.split("@")[1]
        mv = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(mv.run_id)
        threshold = run.data.metrics.get("threshold", 0.5)
        print(f"✅ Loaded threshold from MLflow run: {threshold}")
        return float(threshold)
    except Exception as e:
        print(f"⚠️  Could not load threshold from MLflow ({e}). Defaulting to 0.5.")
        return 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, operating_threshold

    # MLflow 3.x alias URI: models:/name@alias
    model_uri = os.getenv("MODEL_URI", "models:/bank-classifier@Production")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    try:
        ml_model = mlflow.sklearn.load_model(model_uri)
        operating_threshold = _load_threshold(model_uri)
        print(f"✅ Loaded model from {model_uri}  (threshold={operating_threshold})")
    except Exception as e:
        print(f"⚠️  MLflow model not found or registry unreachable: {e}")
        print("🔧 Falling back to DummyModel for API development.")
        ml_model = DummyModel()
        operating_threshold = 0.5

    yield

    ml_model = None


app = FastAPI(title="Drift Triage Model Service", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: BankPredictionRequest):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded into memory.")

    # by_alias=True preserves dot-notation column names (emp.var.rate, etc.)
    # that the trained sklearn pipeline expects.
    input_data = request.model_dump(by_alias=True)
    df = pd.DataFrame([input_data])

    try:
        probability = float(ml_model.predict_proba(df)[0][1])
        prediction  = int(probability >= operating_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")

    # TODO Phase 3: log prediction to Postgres

    return {"prediction": prediction, "probability": probability}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None and not isinstance(ml_model, DummyModel),
        "threshold": operating_threshold,
    }
