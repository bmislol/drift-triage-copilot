from langchain_core.messages import HumanMessage
from app.graph.state import AgentState
from app.services.llm_svc import get_llm
from app.prompts.comms import COMMS_PROMPT
import json

def comms_node(state: AgentState):
    """Generates a human-readable summary of the agent's work."""
    llm = get_llm()
    
    # Provide the context of what happened in previous nodes
    context = {
        "severity": state["severity"],
        "reasoning": state["reasoning"],
        "action_taken": state["recommended_action"],
        "job_id": state.get("job_id", "N/A")
    }
    
    formatted_prompt = COMMS_PROMPT.format(context=json.dumps(context, indent=2))
    
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    return {"summary": response.content}