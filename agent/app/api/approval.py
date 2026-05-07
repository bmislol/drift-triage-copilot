from fastapi import APIRouter, HTTPException
from app.graph.workflow import app_graph
from app.schemas import ApprovalRequest

router = APIRouter(prefix="/approve", tags=["Human-in-the-Loop"])

@router.post("/{thread_id}")
async def approve_action(thread_id: str, request: ApprovalRequest):
    """Resumes a paused graph thread after human approval."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if the thread exists in Postgres
    state = app_graph.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Investigation thread not found.")
    
    # Update the state with the human's decision
    app_graph.update_state(config, {"human_approved": request.approved})
    
    # Resume execution. The graph will now move into the 'action' node.
    app_graph.invoke(None, config=config)
    
    return {"status": "resumed", "approved": request.approved}