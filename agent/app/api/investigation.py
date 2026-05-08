import uuid
from fastapi import APIRouter, HTTPException
from app.graph.workflow import app_graph
from app.schemas import DriftWebhook

from core.database import SessionLocal
from core.models import Investigation

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

    current_state = app_graph.get_state(config)
    print(f"🎯 Graph paused. Next node to execute: {current_state.next}")

    if current_state.values:
        db = SessionLocal()
        try:
            inv = Investigation(
                thread_id=thread_id,
                status="open",
                severity=current_state.values.get("severity", "unknown"),
                recommended_action=current_state.values.get("recommended_action", "unknown"),
                summary=f"Agent Reasoning: {current_state.values.get('reasoning', 'Awaiting approval.')}"
            )
            db.add(inv)
            db.commit()
        except Exception as e:
            print(f"DB Error: {e}")
            db.rollback()
        finally:
            db.close()
    
    return {"thread_id": thread_id, "status": "started"}