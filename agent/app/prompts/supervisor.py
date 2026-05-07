# This variable name MUST match what you are trying to import in supervisor.py
SUPERVISOR_PROMPT = """
You are the Orchestrator. Given the current state of the investigation, 
decide which specialist should work next.

CURRENT STATE:
{state}

OPTIONS:
{options}

Pick the most logical next step.
"""