from fastapi import APIRouter, Request
from app.schemas import PromoteResponse
from app.services import registry_svc

router = APIRouter(tags=["Model Registry"])

@router.post("/promote/{version}", response_model=PromoteResponse)
def promote_model(version: int, request: Request):
    """Programmatic gatekeeper to promote and hot-reload a model version to Production."""
    return registry_svc.evaluate_and_promote(version, request.app)