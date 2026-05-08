COMMS_PROMPT = """
You are the Communications Lead for an MLOps Agent. 
An investigation into model drift has just been resolved.

Drift Event:
{drift_event}

Severity: {severity}
Action Taken: {action}
Job ID: {job_id}

Write a concise, professional 2-3 sentence summary of what happened and what was done to fix it. Do not include greetings.
"""