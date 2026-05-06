# scripts/train.py
import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import our centralized settings
sys.path.append(str(Path(__file__).parent.parent))
from core.config import settings

# Paths and Constants
from core.config import RAW_DATA_PATH, REFERENCE_STATS_PATH, ARTIFACT_PATH

MODEL_NAME     = "bank-classifier"
EXPERIMENT     = "bank-classifier"
SEED           = 42
np.random.seed(SEED)

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain feature engineering step — lives INSIDE the pipeline."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["never_contacted"] = (X["pdays"] == 999).astype(int)
        return X.drop(columns=["pdays"])

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "auc":       roc_auc_score(y, proba),
        "f1":        f1_score(y, pred),
        "recall":    recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
    }

def make_pipeline(classifier, preprocessor):
    return Pipeline([
        ("feature_engineer", BankFeatureEngineer()),
        ("preprocessor",     clone(preprocessor)),
        ("classifier",       clone(classifier)),
    ])

def train_and_register():
    print("=" * 55)
    print("  Phase 2: Train & Register  —  bank-classifier")
    print("=" * 55)

    if not RAW_DATA_PATH.exists():
        sys.exit(f"\nData file not found: {DATA_PATH}\nPlace bank-additional-full.csv in data/raw/")

    df_raw = pd.read_csv(RAW_DATA_PATH, sep=";")
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()).hexdigest()

    # Data Cleaning: Drop duration to prevent leakage
    df = df_raw.copy()
    df = df.drop(columns=["duration"])
    df["y"] = (df["y"] == "yes").astype(int)

    # Stratified 60/20/20 split
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    # Build Preprocessor
    _engineered = BankFeatureEngineer().fit_transform(X_train)
    numeric_cols = _engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = _engineered.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # Train Models
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=SEED)
    best_pipeline = make_pipeline(clf, preprocessor)
    best_pipeline.fit(X_train, y_train)

    # Tune threshold (highest where val recall >= 0.75)
    val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    for t in np.arange(0.90, 0.05, -0.01):
        t_rounded = round(float(t), 2)
        preds = (val_probs >= t_rounded).astype(int)
        if recall_score(y_val, preds) >= 0.75:
            best_threshold = t_rounded
            break

    # Test evaluation
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)
    test_f1 = f1_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_pre = precision_score(y_test, test_preds, zero_division=0)
    train_auc = roc_auc_score(y_train, best_pipeline.predict_proba(X_train)[:, 1])
    final_gap = train_auc - test_auc

    # Save reference stats
    _eng_train = BankFeatureEngineer().fit_transform(X_train)
    reference_stats = {}
    for col in numeric_cols:
        vals = _eng_train[col].values
        counts, edges = np.histogram(vals, bins=10)
        reference_stats[col] = {
            "type": "numeric", "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "histogram_counts": counts.tolist(), "histogram_edges": edges.tolist(),
        }
    for col in categorical_cols:
        dist = _eng_train[col].value_counts(normalize=True).to_dict()
        reference_stats[col] = {"type": "categorical", "distribution": dist}

    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(reference_stats, f, indent=2)

    # Save artifact
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACT_PATH)
    artifact_hash = sha256_file(ARTIFACT_PATH)

    # MLflow registration
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT)
    mlflow.set_experiment(EXPERIMENT)

    env_meta = {
        "python": platform.python_version(), "platform": platform.platform(),
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "mlflow": mlflow.__version__,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    signature = infer_signature(X_train.head(100), best_pipeline.predict_proba(X_train.head(100)))

    model_card = {
        "model_name": MODEL_NAME, "features_input": list(X_train.columns),
        "features_dropped": ["duration", "pdays"], "features_derived": ["never_contacted"],
        "threshold": best_threshold, "test_auc": test_auc, "test_f1": test_f1,
        "test_recall": test_rec, "test_precision": test_pre, "train_test_gap": final_gap,
        "dataset_md5": df_hash, "artifact_sha256": artifact_hash, "environment": env_meta,
    }

    with mlflow.start_run(run_name="gbm-v1") as run:
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("threshold", best_threshold)
        for k, v in env_meta.items(): mlflow.set_tag(f"env.{k}", v)
        mlflow.sklearn.log_model(sk_model=best_pipeline, artifact_path="model",
                                 registered_model_name=MODEL_NAME, signature=signature,
                                 input_example=X_train.head(2))
        mlflow.log_dict(model_card, "model_card.json")
        mlflow.log_artifact(str(REFERENCE_STATS_PATH))

    # Promote to Production alias
    client = MlflowClient()
    versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda mv: int(mv.version))
    latest_version = versions[-1].version
    client.set_registered_model_alias(MODEL_NAME, "Production", latest_version)
    
    print(f"\n✅ Training complete. Model registered as {MODEL_NAME} v{latest_version} (Production).")
    print(f"✅ Operating Threshold tuned to: {best_threshold}")

if __name__ == "__main__":
    train_and_register()