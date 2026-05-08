import json
from langchain_core.messages import HumanMessage
from app.graph.state import AgentState
from app.services.llm_svc import get_llm
from app.prompts.comms import COMMS_PROMPT
from langchain_core.runnables import RunnableConfig

# Import your database and models from the core package
from core.database import SessionLocal
from core.models import Investigation

def comms_node(state: AgentState, config: RunnableConfig):
    """Generates a final summary and saves the record to Postgres."""
    llm = get_llm()
    
    # 1. Generate the human-readable summary
    formatted_prompt = COMMS_PROMPT.format(
        drift_event=json.dumps(state.get("drift_event", {})),
        severity=state.get("severity"),
        action=state.get("recommended_action"),
        job_id=state.get("job_id")
    )
    
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    final_summary = response.content
    
    print(f"📝 Comms Summary: {final_summary}")
    
    # 2. Save the official record to the Postgres database
    # Extract the thread_id from the LangGraph config
    thread_id = config["configurable"]["thread_id"]
    
    db = SessionLocal()
    try:
        # Check if we already created a row for this, otherwise create new
        investigation = db.query(Investigation).filter(Investigation.thread_id == thread_id).first()
        
        if not investigation:
            investigation = Investigation(thread_id=thread_id)
            db.add(investigation)
            
        investigation.status = "resolved"
        investigation.severity = state.get("severity")
        investigation.recommended_action = state.get("recommended_action")
        investigation.summary = final_summary
        
        db.commit()
        print(f"💾 Investigation {thread_id} saved to Postgres.")
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()
        
    return {"summary": final_summary, "next": "FINISH"}