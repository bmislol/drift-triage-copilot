"""
scripts/train.py
Phase 2: Train the bank marketing classifier and register it to MLflow.

Run once before docker-compose up (MLflow must be reachable):
    uv run python scripts/train.py

Reads:   data/raw/bank-additional-full.csv
Writes:  data/reference_stats.json
         MLflow registry: bank-classifier  (alias: Production)

Key design decisions:
- BankFeatureEngineer is the first step inside the sklearn Pipeline so the
  registered model is self-contained: it accepts raw features (with pdays)
  and handles the sentinel conversion internally.
- MLflow 3.x model aliases are used instead of the removed stages API.
- Threshold is chosen as the HIGHEST value where val-set recall >= 0.75.
"""

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
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── constants ──────────────────────────────────────────────────────────────────
# Resolve all paths relative to this file so the script works regardless of CWD.
# scripts/train.py → scripts/ → project root
_ROOT = Path(__file__).parent.parent

DATA_PATH      = _ROOT / "data/raw/bank-additional-full.csv"
REFERENCE_PATH = _ROOT / "data/reference_stats.json"
ARTIFACT_PATH  = _ROOT / "data/bank_pipeline.joblib"
MODEL_NAME     = "bank-classifier"
EXPERIMENT     = "bank-classifier"
SEED           = 42
np.random.seed(SEED)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ── custom transformer ─────────────────────────────────────────────────────────
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain feature engineering step — lives INSIDE the pipeline.

    Converts the pdays sentinel value (999 = never contacted) into a binary
    flag. Keeping 999 as a raw integer would make StandardScaler treat it as
    a large positive number, distorting the feature distribution.

    By placing this inside the Pipeline the registered model accepts the same
    raw columns as the API schema (pdays included) with no external transforms.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["never_contacted"] = (X["pdays"] == 999).astype(int)
        return X.drop(columns=["pdays"])


# ── helpers ────────────────────────────────────────────────────────────────────
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
    """Always clone — never share a fitted preprocessor across models."""
    return Pipeline([
        ("feature_engineer", BankFeatureEngineer()),
        ("preprocessor",     clone(preprocessor)),
        ("classifier",       clone(classifier)),
    ])


# ── main ───────────────────────────────────────────────────────────────────────
def main():

    # 1. Load & fingerprint ────────────────────────────────────────────────────
    print("=" * 55)
    print("  Phase 2: Train & Register  —  bank-classifier")
    print("=" * 55)

    if not DATA_PATH.exists():
        sys.exit(
            f"\nData file not found: {DATA_PATH}\n"
            "Download bank-additional-full.csv from UCI and place it in data/raw/"
        )

    print(f"\n[1] Loading {DATA_PATH} ...")
    df_raw = pd.read_csv(DATA_PATH, sep=";")
    print(f"    Shape: {df_raw.shape}  (expected ~41188 × 21)")

    df_hash = hashlib.md5(
        pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()
    ).hexdigest()
    print(f"    Dataset MD5: {df_hash}")

    pos_rate = (df_raw["y"] == "yes").mean()
    print(f"    Positive rate: {pos_rate:.1%}  (expected ~11%)")

    # 2. Clean ─────────────────────────────────────────────────────────────────
    print("\n[2] Cleaning ...")
    df = df_raw.copy()

    # duration is recorded AFTER the call ends — it leaks the label
    df = df.drop(columns=["duration"])
    print(f"    Dropped 'duration' (data leakage). Columns: {df.shape[1]}")

    df["y"] = (df["y"] == "yes").astype(int)
    print(f"    Encoded target: {df['y'].sum():,} positives / {len(df):,} total")
    # NOTE: pdays sentinel → never_contacted is handled inside BankFeatureEngineer

    # 3. Split ─────────────────────────────────────────────────────────────────
    print("\n[3] Splitting 60 / 20 / 20 (stratified) ...")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    print(f"    Train {len(X_train):,} ({y_train.mean():.1%} pos) | "
          f"Val {len(X_val):,} ({y_val.mean():.1%} pos) | "
          f"Test {len(X_test):,} ({y_test.mean():.1%} pos)")

    rates = [y_train.mean(), y_val.mean(), y_test.mean()]
    spread = max(rates) - min(rates)
    flag = "✅ OK" if spread < 0.005 else "⚠️  check stratification"
    print(f"    Rate spread: {spread:.4f}  {flag}")

    # 4. Build preprocessor ────────────────────────────────────────────────────
    print("\n[4] Building preprocessor ...")

    # Apply feature engineering to training set so we can infer column types
    # after pdays is replaced by never_contacted.
    _engineered = BankFeatureEngineer().fit_transform(X_train)
    numeric_cols     = _engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = _engineered.select_dtypes(include=["object"]).columns.tolist()
    print(f"    Numeric ({len(numeric_cols)}):     {numeric_cols}")
    print(f"    Categorical ({len(categorical_cols)}): {categorical_cols}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # 5. Train & compare models ────────────────────────────────────────────────
    print("\n[5] Training three models ...")
    base_classifiers = {
        "LR": LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=1.0, solver="lbfgs", random_state=SEED,
        ),
        "RF": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=SEED,
        ),
    }

    fitted_pipelines = {}
    results = {}
    for name, clf in base_classifiers.items():
        pipe = make_pipeline(clf, preprocessor)
        pipe.fit(X_train, y_train)
        train_m = evaluate(pipe, X_train, y_train)
        val_m   = evaluate(pipe, X_val, y_val)
        gap = train_m["auc"] - val_m["auc"]
        results[name] = {
            "train_auc": train_m["auc"],
            "val_auc":   val_m["auc"],
            "val_f1":    val_m["f1"],
            "gap":       gap,
        }
        fitted_pipelines[name] = pipe
        flag = "⚠️  possible overfit" if gap > 0.05 else "✅"
        print(f"    {name:4s} val_auc={val_m['auc']:.4f}  gap={gap:.4f}  {flag}")

    # 6. Pick winner & cross-validate ──────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n[6] Best: {best_name}  (val_auc={results[best_name]['val_auc']:.4f})")

    cv_pipe   = make_pipeline(base_classifiers[best_name], preprocessor)
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(cv_pipe, X_train, y_train,
                                cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_diff = abs(results[best_name]["val_auc"] - cv_scores.mean())
    cv_flag = "✅ consistent" if cv_diff < 0.02 else "⚠️  val was lucky/unlucky — trust CV"
    print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  {cv_flag}")

    # 7. Tune threshold ─────────────────────────────────────────────────────────
    print("\n[7] Tuning threshold (highest where val recall >= 0.75) ...")
    val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold = 0.5  # fallback
    for t in np.arange(0.90, 0.05, -0.01):
        # Round before the check so the threshold used for evaluation is the same
        # as the one stored — raw np.arange floats (0.06999...) can straddle
        # the boundary differently than their rounded representation (0.07).
        t_rounded = round(float(t), 2)
        preds = (val_probs >= t_rounded).astype(int)
        if recall_score(y_val, preds) >= 0.75:
            best_threshold = t_rounded
            break

    val_preds_at_t = (val_probs >= best_threshold).astype(int)
    print(f"    Threshold: {best_threshold}")
    print(f"    Val recall:    {recall_score(y_val, val_preds_at_t):.4f}  (constraint >= 0.75)")
    print(f"    Val precision: {precision_score(y_val, val_preds_at_t, zero_division=0):.4f}")
    print(f"    Val F1:        {f1_score(y_val, val_preds_at_t):.4f}")

    if best_threshold == 0.5 and recall_score(y_val, (val_probs >= 0.5).astype(int)) < 0.75:
        print("    ⚠️  No threshold meets recall >= 0.75. Using 0.5 as fallback.")

    # 8. Test evaluation (run ONCE — test set is now opened) ───────────────────
    print("\n[8] Test evaluation (sealed until now) ...")
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_auc   = roc_auc_score(y_test, test_probs)
    test_f1    = f1_score(y_test, test_preds)
    test_rec   = recall_score(y_test, test_preds)
    test_pre   = precision_score(y_test, test_preds, zero_division=0)
    train_auc  = roc_auc_score(y_train, best_pipeline.predict_proba(X_train)[:, 1])
    final_gap  = train_auc - test_auc

    print(f"    AUC={test_auc:.4f}  F1={test_f1:.4f}  "
          f"Recall={test_rec:.4f}  Precision={test_pre:.4f}  gap={final_gap:.4f}")
    if test_rec < 0.75:
        print("    ⚠️  WARNING: test recall < 0.75")
    if final_gap > 0.05:
        print("    ⚠️  WARNING: train-test gap > 0.05 — model may be overfitting")

    # 9. Save reference stats (training distribution for drift detection) ───────
    print("\n[9] Saving reference stats ...")

    # Compute stats on the engineered training set (pdays already converted)
    _eng_train = BankFeatureEngineer().fit_transform(X_train)
    reference_stats = {}

    for col in numeric_cols:
        vals = _eng_train[col].values
        counts, edges = np.histogram(vals, bins=10)
        reference_stats[col] = {
            "type":             "numeric",
            "mean":             float(vals.mean()),
            "std":              float(vals.std()),
            "min":              float(vals.min()),
            "max":              float(vals.max()),
            "histogram_counts": counts.tolist(),
            "histogram_edges":  edges.tolist(),
        }

    for col in categorical_cols:
        dist = _eng_train[col].value_counts(normalize=True).to_dict()
        reference_stats[col] = {"type": "categorical", "distribution": dist}

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference_stats, f, indent=2)
    print(f"    Saved: {REFERENCE_PATH}  "
          f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

    # 10. Save artifact ─────────────────────────────────────────────────────────
    print("\n[10] Saving pipeline artifact ...")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACT_PATH)
    artifact_hash = sha256_file(ARTIFACT_PATH)
    print(f"    Saved: {ARTIFACT_PATH}  ({ARTIFACT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"    SHA-256: {artifact_hash}")

    # 11. MLflow registration ───────────────────────────────────────────────────
    print(f"\n[11] Registering to MLflow ({MLFLOW_URI}) ...")
    mlflow.set_tracking_uri(MLFLOW_URI)

    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT)
    mlflow.set_experiment(EXPERIMENT)

    env_meta = {
        "python":      platform.python_version(),
        "platform":    platform.platform(),
        "sklearn":     sklearn.__version__,
        "numpy":       np.__version__,
        "pandas":      pd.__version__,
        "mlflow":      mlflow.__version__,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Signature: infer from raw X (before feature engineering) so the serving
    # signature matches what the API sends in — pdays included.
    signature = infer_signature(
        X_train.head(100),
        best_pipeline.predict_proba(X_train.head(100)),
    )

    model_card = {
        "model_name":       MODEL_NAME,
        "algorithm":        best_name,
        "features_input":   list(X_train.columns),
        "features_dropped": ["duration", "pdays"],
        "features_derived": ["never_contacted"],
        "threshold":        best_threshold,
        "test_auc":         test_auc,
        "test_f1":          test_f1,
        "test_recall":      test_rec,
        "test_precision":   test_pre,
        "train_test_gap":   final_gap,
        "cv_auc_mean":      float(cv_scores.mean()),
        "cv_auc_std":       float(cv_scores.std()),
        "dataset_md5":      df_hash,
        "artifact_sha256":  artifact_hash,
        "environment":      env_meta,
    }

    with mlflow.start_run(run_name=f"{best_name}-v1") as run:
        mlflow.log_metric("test_auc",       test_auc)
        mlflow.log_metric("test_f1",        test_f1)
        mlflow.log_metric("test_recall",    test_rec)
        mlflow.log_metric("test_precision", test_pre)
        mlflow.log_metric("threshold",      best_threshold)
        mlflow.log_metric("train_test_gap", final_gap)
        mlflow.log_metric("cv_auc_mean",    cv_scores.mean())
        mlflow.log_metric("cv_auc_std",     cv_scores.std())

        for k, v in env_meta.items():
            mlflow.set_tag(f"env.{k}", v)
        mlflow.set_tag("artifact.sha256", artifact_hash)
        mlflow.set_tag("dataset.md5",     df_hash)
        mlflow.set_tag("best_model",      best_name)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_train.head(2),
        )
        mlflow.log_dict(model_card, "model_card.json")
        mlflow.log_artifact(str(REFERENCE_PATH))

        run_id = run.info.run_id
    print(f"    Run ID: {run_id}")

    # 12. Promote via alias (MLflow 3.x — stages API was removed) ──────────────
    print("\n[12] Promoting to Production alias ...")
    client = MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda mv: int(mv.version),
    )
    latest_version = versions[-1].version
    client.set_registered_model_alias(MODEL_NAME, "Production", latest_version)
    print(f"    ✅ v{latest_version} → alias 'Production'")

    # 13. Round-trip fidelity check ─────────────────────────────────────────────
    print("\n[13] Round-trip fidelity check (atol=1e-12) ...")
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Production")
    orig   = best_pipeline.predict_proba(X_val.head(10))
    reloaded = loaded.predict_proba(X_val.head(10))
    max_diff = float(np.abs(orig - reloaded).max())

    if np.allclose(orig, reloaded, atol=1e-12):
        print(f"    ✅ Fidelity confirmed. Max diff: {max_diff:.2e}")
    else:
        print(f"    ❌ Fidelity FAILED. Max diff: {max_diff:.2e}")
        print("       Check sklearn versions between this env and the Docker container.")
        sys.exit(1)

    # 14. Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  DONE — copy these into README / submission")
    print("=" * 55)
    print(f"  Dataset MD5:       {df_hash}")
    print(f"  Train / Val / Test:{len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"  Best model:        {best_name}")
    print(f"  Test AUC:          {test_auc:.4f}")
    print(f"  Test F1:           {test_f1:.4f}")
    print(f"  Test Recall:       {test_rec:.4f}")
    print(f"  Threshold:         {best_threshold}  (rule: highest where recall >= 0.75)")
    print(f"  Train-test gap:    {final_gap:.4f}")
    print(f"  CV AUC:            {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Artifact SHA-256:  {artifact_hash}")
    print(f"  MLflow run ID:     {run_id}")
    print(f"  Registry:          {MODEL_NAME} v{latest_version}  alias=Production")
    print(f"  Load URI:          models:/{MODEL_NAME}@Production")
    print("=" * 55)


if __name__ == "__main__":
    main()
