import json
from langchain_core.messages import HumanMessage
from app.graph.state import AgentState
from app.services.llm_svc import get_llm
from app.prompts.triage import TRIAGE_PROMPT

def triage_node(state: AgentState):
    """Analyzes drift and determines severity/action."""
    llm = get_llm()
    
    # Format the prompt with the drift event from state
    formatted_prompt = TRIAGE_PROMPT.format(
        drift_report=json.dumps(state["drift_event"], indent=2)
    )
    
    # We use .with_structured_output to force the LLM to return valid keys
    structured_llm = llm.with_structured_output({
        "title": "TriageResult",
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["none", "warning", "critical"]},
            "recommended_action": {"type": "string", "enum": ["retrain", "rollback", "replay_test", "none"]},
            "reasoning": {"type": "string"}
        },
        "required": ["severity", "recommended_action", "reasoning"]
    })
    
    response = structured_llm.invoke([HumanMessage(content=formatted_prompt)])
    
    print(f"🕵️ Triage Complete: {response['severity']} - Action: {response['recommended_action']}")
    
    return {
        "severity": response["severity"],
        "recommended_action": response["recommended_action"],
        "reasoning": response["reasoning"]
    }