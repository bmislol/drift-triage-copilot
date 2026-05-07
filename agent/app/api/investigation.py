import uuid
from fastapi import APIRouter, HTTPException
from app.graph.workflow import app_graph
from app.schemas import DriftWebhook

router = APIRouter(prefix="/investigations", tags=["Investigations"])

@router.post("/webhook/drift")
async def handle_drift_webhook(payload: DriftWebhook):
    """Entry point: Starts a new LangGraph thread for a drift event."""
    # Every investigation needs a unique thread_id for Postgres persistence
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize the graph state with the drift event
    initial_state = {
        "drift_event": payload.drift_report,
        "active_version": payload.active_version,
        "human_approved": False
    }
    
    # Start the graph. It will run until it hits the 'action' interrupt.
    app_graph.invoke(initial_state, config=config)
    
    return {"thread_id": thread_id, "status": "started"}