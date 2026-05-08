from langchain_core.messages import HumanMessage
from app.graph.state import AgentState
from app.services.llm_svc import get_llm
from app.services.redis_svc import enqueue_job
from app.prompts.action import ACTION_PROMPT
import uuid
import json

def action_node(state: AgentState):
    """Dispatches the approved action and generates an execution log."""
    # 1. HIL Check: Only run if approved
    if not state.get("human_approved"):
        return {"summary": "Action blocked: Awaiting Human Approval."}

    llm = get_llm()
    action = state["recommended_action"]
    active_version = state.get("active_version", "1")

    # Calculate rollback target (Simple logic: N - 1)
    # We strip 'v' if it exists (e.g., "v2" -> 2)
    try:
        current_num = int(active_version.replace("v", ""))
        target_version = str(max(1, current_num - 1))
    except:
        target_version = "1"
    
    # 2. Use the LLM to 'finalize' the action summary
    formatted_prompt = ACTION_PROMPT.format(
        action=action,
        severity=state["severity"],
        reasoning=state["reasoning"]
    )
    
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    # 3. Mechanical Task: Enqueue to Redis
    job_id = f"{action}-{uuid.uuid4().hex[:6]}"
    
    payload = {
        "drift_event": state["drift_event"],
        "version": target_version 
    }
    
    success = enqueue_job(job_id, action, payload)

    print(f"📦 Tried to enqueue job {job_id} to Redis. Success: {success}")
    
    return {
        "job_id": job_id,
        "summary": response.content if success else "Failed to dispatch job to Redis."
    }