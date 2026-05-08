import httpx
import random
import time

# Adjust this if your Model Service runs on a different port!
MODEL_API_URL = "http://localhost:8000/predict"

# A baseline user from the UCI dataset (all normal values)
base_client = {
    "age": 30, "job": "blue-collar", "marital": "married", "education": "basic.9y",
    "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "fri", "campaign": 1, "pdays": 999,
    "previous": 0, "poutcome": "nonexistent", "emp_var_rate": -1.8,
    "cons_price_idx": 92.893, "cons_conf_idx": -46.2, 
    "euribor3m": 1.299, # Baseline euribor3m
    "nr_employed": 5099.1
}

def seed_data():
    with httpx.Client() as client:
        print("🌱 Seeding normal predictions...")
        # Send 20 normal predictions
        for _ in range(20):
            client.post(MODEL_API_URL, json=base_client)
            
        print("🚨 INJECTING MACROECONOMIC DRIFT! (Shifting euribor3m and job)")
        # Send 50 heavily drifted predictions
        for _ in range(50):
            drifted_client = base_client.copy()
            
            # 1. Numeric Drift: Force the interest rate unnaturally high
            drifted_client["euribor3m"] = random.uniform(8.0, 12.0) 
            
            # 2. Categorical Drift: Change all jobs to "unknown"
            drifted_client["job"] = "unknown"
            
            try:
                client.post(MODEL_API_URL, json=drifted_client)
            except Exception as e:
                print(f"API Error: {e}")
                
            time.sleep(0.1) # Small delay to not overwhelm the local server

        print("✅ Done injecting drift.")

if __name__ == "__main__":
    seed_data()