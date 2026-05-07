import redis
import json
from core.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)

def enqueue_job(job_id: str, job_type: str, payload: dict) -> bool:
    """Helper to push jobs to the Phase 4 worker queue."""
    # Check idempotency
    if not redis_client.set(f"idempotency:{job_id}", "1", nx=True, ex=86400):
        return False
        
    job_data = {"job_id": job_id, "type": job_type, "payload": payload, "retry_count": 0}
    redis_client.rpush("q:main", json.dumps(job_data))
    return True