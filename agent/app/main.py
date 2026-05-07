import sys
from pathlib import Path
from fastapi import FastAPI

# Standard path resolution for core/app imports
CURRENT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = CURRENT_DIR.parent
ROOT_DIR = SERVICE_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SERVICE_DIR))

from app.api import investigation, approval

app = FastAPI(title="Drift Triage Agent")

app.include_router(investigation.router)
app.include_router(approval.router)

@app.get("/health")
def health_check():
    return {"status": "online"}