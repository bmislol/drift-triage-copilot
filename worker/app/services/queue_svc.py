import json
import redis
import sys
from pathlib import Path

# Path hack to ensure we can import from core (similar to what we did in model_service)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from core.config import settings

# Initialize Redis client using the shared config
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

def enqueue_job(job_id: str, job_type: str, payload: dict) -> bool:
    """
    Pushes a job to q:main if it hasn't been queued recently.
    Returns True if queued, False if skipped due to idempotency.
    """
    idem_key = f"idempotency:{job_id}"
    
    # NX = Only set if it doesn't exist, EX = Expire in 86400 seconds (24 hours)
    is_new = redis_client.set(idem_key, "1", nx=True, ex=86400)
    
    if not is_new:
        print(f"⚠️ Job {job_id} skipped (idempotency key already exists).")
        return False
        
    job_data = {
        "job_id": job_id,
        "type": job_type,
        "payload": payload,
        "retry_count": 0
    }
    
    # Push the JSON string to the right end of the list
    redis_client.rpush("q:main", json.dumps(job_data))
    print(f"📥 Enqueued {job_type} job: {job_id}")
    return True

def pop_job(timeout: int = 0) -> dict | None:
    """
    Blocks until a job is available in q:main, then pops it from the left.
    """
    # BLPOP returns a tuple: (queue_name, data) or None if timeout is reached
    result = redis_client.blpop("q:main", timeout=timeout)
    if result:
        return json.loads(result[1])
    return None

def requeue_job(job_data: dict):
    """
    Pushes a failed job back to the queue with an incremented retry count.
    """
    job_data["retry_count"] += 1
    redis_client.rpush("q:main", json.dumps(job_data))
    print(f"🔄 Re-queued job {job_data['job_id']} (Attempt {job_data['retry_count']})")

def push_to_dlq(job_data: dict, error_msg: str):
    """
    Moves a permanently failed job to the Dead Letter Queue.
    """
    job_data["error_reason"] = error_msg
    redis_client.rpush("q:dlq", json.dumps(job_data))
    print(f"🪦 Job {job_data['job_id']} moved to DLQ. Reason: {error_msg}")