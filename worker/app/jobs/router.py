import traceback
from app.schemas import JobPayload
from app.services.queue_svc import requeue_job, push_to_dlq
from app.jobs.model_jobs import handle_retrain, handle_rollback, handle_replay_test

def execute_job(job_data: dict):
    """Routes the job to the correct handler and manages retries."""
    try:
        job = JobPayload(**job_data)
        print(f"🚀 Starting job {job.job_id} of type {job.type}")
        
        if job.type == "retrain":
            handle_retrain(job.payload)
        elif job.type == "rollback":
            handle_rollback(job.payload)
        elif job.type == "replay_test":
            handle_replay_test(job.payload)
        else:
            raise ValueError(f"Unknown job type: {job.type}")
            
        print(f"🎉 Job {job.job_id} completed successfully.")
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Job {job_data.get('job_id')} failed: {str(e)}")
        
        current_retries = job_data.get("retry_count", 0)
        if current_retries < 5:
            requeue_job(job_data)
        else:
            push_to_dlq(job_data, error_msg)