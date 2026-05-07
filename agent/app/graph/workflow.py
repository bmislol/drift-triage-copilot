from langgraph.graph import StateGraph, START, END
from app.graph.state import AgentState
from app.graph.nodes.triage import triage_node
from app.graph.nodes.action import action_node
from app.graph.nodes.comms import comms_node
from app.graph.supervisor import supervisor_node
from app.services.checkpoint_svc import get_checkpointer

def create_graph():
    # 1. Initialize the Graph with our shared State
    workflow = StateGraph(AgentState)
    
    # 2. Add our specialized nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("action", action_node)
    workflow.add_node("comms", comms_node)
    
    # 3. Define the edges - sub-agents always report back to the supervisor
    workflow.add_edge("triage", "supervisor")
    workflow.add_edge("action", "supervisor")
    workflow.add_edge("comms", END)
    
    # 4. Define the Supervisor's conditional routing
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "triage": "triage",
            "action": "action",
            "comms": "comms",
            "FINISH": END,
            END: END # for safety
        }
    )
    
    # 5. Set the entry point
    workflow.set_entry_point("supervisor")
    
    # 6. Initialize the checkpointer
    checkpointer = get_checkpointer()
    
    # Create internal LangGraph tables if they don't exist
    checkpointer.setup()
    
    # Compile with the checkpointer and HIL Interrupt
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["action"]
    )

# Singleton instance of the compiled graph
app_graph = create_graph()