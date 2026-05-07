import sys
import uuid
import json
import redis
from pathlib import Path

# Add root to path so we can import core
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import settings

def send_test_job(job_type: str, payload: dict):
    # Connect to Redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Create a unique job ID
    job_id = f"test-{job_type}-{uuid.uuid4().hex[:6]}"
    
    # Construct the JSON message matching your JobPayload schema
    message = {
        "job_id": job_id,
        "type": job_type,
        "payload": payload,
        "retry_count": 0
    }
    
    # Push to the right side of q:main (RPUSH)
    r.rpush("q:main", json.dumps(message))
    print(f"✅ Successfully pushed {job_type} job to q:main")
    print(f"   Job ID: {job_id}")

if __name__ == "__main__":
    # Test 1: A Replay Test (Safe, no side effects)
    send_test_job("replay_test", {"description": "Checking plumbing"})
    
    # Test 2: A Retrain (Optional)
    # send_test_job("retrain", {})