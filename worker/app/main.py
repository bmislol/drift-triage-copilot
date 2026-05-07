import sys
import time
from pathlib import Path

# Setup paths so we can import 'core' and 'app'
CURRENT_DIR = Path(__file__).resolve().parent      # worker/app
SERVICE_DIR = CURRENT_DIR.parent                   # worker/
ROOT_DIR = SERVICE_DIR.parent                      # drift-triage-copilot/

# Force BOTH into Python's system path so 'core' and 'app' can be found
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SERVICE_DIR))

from app.services.queue_svc import pop_job
from app.jobs.router import execute_job

def main():
    print("👷 Worker started. Listening to q:main...")
    
    while True:
        try:
            # Block until a job arrives (timeout=0 means wait forever)
            job_data = pop_job(timeout=0)
            
            if job_data:
                execute_job(job_data)
                
        except KeyboardInterrupt:
            print("\n🛑 Worker shutting down gracefully...")
            break
        except Exception as e:
            print(f"💥 Critical worker error: {e}")
            # Sleep briefly to prevent log-spam if Redis goes down temporarily
            time.sleep(5) 

if __name__ == "__main__":
    main()