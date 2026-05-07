from typing import Annotated, TypedDict, Optional, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The raw drift payload from the model service
    drift_event: dict

    active_version: str
    
    # Triage results
    severity: str  # none, warning, critical
    reasoning: str
    
    # Action results
    recommended_action: str  # retrain, rollback, replay_test, none
    job_id: Optional[str]
    
    # Human-in-the-loop
    human_approved: bool
    
    # Final summary for the user
    summary: str

    # Saves to postgres
    next: str
    
    # Standard LangGraph message history
    messages: Annotated[list, add_messages]