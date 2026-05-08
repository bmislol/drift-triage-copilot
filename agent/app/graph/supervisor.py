import json
from langchain_core.messages import HumanMessage
from app.graph.state import AgentState
from app.services.llm_svc import get_llm
from app.prompts.supervisor import SUPERVISOR_PROMPT

def supervisor_node(state: AgentState):
    """Orchestrator that decides the next step in the investigation."""
    llm = get_llm()
    
    # We want the LLM to choose the next destination
    options = ["triage", "action", "comms", "FINISH"]
    
    formatted_prompt = SUPERVISOR_PROMPT.format(
        state=json.dumps({
            "severity": state.get("severity"),
            "action": state.get("recommended_action"),
            "human_approved": state.get("human_approved"),
            "job_id": state.get("job_id")
        }, indent=2),
        options=options
    )
    
    # Structured output ensures the LLM picks one of our valid nodes
    structured_llm = llm.with_structured_output({
        "title": "SupervisorDescision",
        "type": "object",
        "properties": {
            "next": {"type": "string", "enum": options}
        },
        "required": ["next"]
    })
    
    response = structured_llm.invoke([HumanMessage(content=formatted_prompt)])
    
    print(f"🧠 Supervisor decided to route to: {response['next']}")

    return {"messages": [HumanMessage(content=f"Supervisor decided: {response['next']}", name="Supervisor")], "next": response["next"]}