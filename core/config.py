# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# --- PATH RESOLUTION ---
# __file__ is core/config.py. Parent is core/. Parent.parent is project-week5/
ROOT_DIR = Path(__file__).parent.parent

# Define all global project paths here
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank-additional-full.csv"
REFERENCE_STATS_PATH = DATA_DIR / "reference_stats.json"
ARTIFACT_PATH = DATA_DIR / "bank_pipeline.joblib"

# --- ENVIRONMENT SETTINGS ---
class Settings(BaseSettings):
    postgres_dsn: str
    redis_url: str
    mlflow_tracking_uri: str
    agent_webhook_url: str
    anthropic_api_key: str  

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()