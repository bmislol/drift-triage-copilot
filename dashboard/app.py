import sys
from pathlib import Path
import time
import random
import streamlit as st
import pandas as pd
import httpx

# Route to the core folder
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.database import SessionLocal
from core.models import Investigation, PredictionLog

st.set_page_config(page_title="Drift Triage Copilot", page_icon="🚨", layout="wide")

# --- API HELPERS ---
MODEL_API_URL = "http://localhost:8000/predict"
AGENT_API_URL = "http://localhost:8001/approve"

def fetch_db_records(model, limit=None):
    db = SessionLocal()
    try:
        query = db.query(model)
        if model == Investigation:
            query = query.order_by(Investigation.created_at.desc())
        else:
            query = query.order_by(model.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        return query.all()
    finally:
        db.close()

def approve_action(thread_id, approved=True):
    try:
        response = httpx.post(f"{AGENT_API_URL}/{thread_id}", json={"approved": approved}, timeout=10.0)
        if response.status_code == 200:
            st.success(f"Action {'Approved' if approved else 'Denied'} for thread {thread_id[:8]}")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Agent connection failed: {e}")

# --- NAVIGATION ---
st.sidebar.title("🌲 MLOps Copilot")
page = st.sidebar.radio(
    "Navigation", 
    [
        "📥 Inbox (Approvals)", 
        "🗄️ Investigation History", 
        "🔮 Manual Prediction", 
        "📊 Prediction Logs", 
        "🧪 Drift Simulator"
    ]
)

# ==========================================
# PAGE 1: INBOX
# ==========================================
if page == "📥 Inbox (Approvals)":
    st.title("📥 Pending Approvals")
    st.markdown("Review and authorize actions recommended by the Triage Agent.")
    
    investigations = fetch_db_records(Investigation)
    open_invs = [inv for inv in investigations if inv.status == "open"]
    
    if not open_invs:
        st.success("Inbox zero! No pending interventions.")
    else:
        for inv in open_invs:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"🔴 {inv.severity.upper()} Drift Event")
                    st.write(f"**Thread:** `{inv.thread_id}`")
                    st.write(f"**Agent Summary:** {inv.summary}")
                    st.write(f"**Proposed Action:** `{inv.recommended_action.upper()}`")
                with col2:
                    st.write("") # spacing
                    st.write("")
                    if st.button("✅ Approve Action", key=f"app_{inv.thread_id}", type="primary", use_container_width=True):
                        approve_action(inv.thread_id, True)
                        st.rerun()
                    if st.button("❌ Deny Action", key=f"den_{inv.thread_id}", use_container_width=True):
                        approve_action(inv.thread_id, False)
                        st.rerun()

# ==========================================
# PAGE 2: HISTORY
# ==========================================
elif page == "🗄️ Investigation History":
    st.title("🗄️ Resolved Investigations")
    investigations = fetch_db_records(Investigation)
    resolved_invs = [inv for inv in investigations if inv.status != "open"]
    
    if resolved_invs:
        df = pd.DataFrame([{
            "Time": inv.updated_at.strftime("%Y-%m-%d %H:%M") if inv.updated_at else "N/A",
            "Thread": inv.thread_id[:8],
            "Severity": inv.severity,
            "Action Taken": inv.recommended_action,
            "Summary": inv.summary
        } for inv in resolved_invs])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No past investigations.")

# ==========================================
# PAGE 3: MANUAL PREDICTION
# ==========================================
elif page == "🔮 Manual Prediction":
    st.title("🔮 Test Model Inference")
    st.markdown("Send a single record to the active production model.")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", value=30)
            job = st.selectbox("Job", ["blue-collar", "admin.", "technician", "management", "unknown"])
            marital = st.selectbox("Marital", ["married", "single", "divorced"])
        with col2:
            euribor3m = st.number_input("Euribor 3 Month Rate", value=1.299, format="%.3f")
            emp_var_rate = st.number_input("Emp Var Rate", value=-1.8)
            cons_price_idx = st.number_input("Consumer Price Index", value=92.89)
        with col3:
            education = st.selectbox("Education", ["basic.9y", "high.school", "university.degree"])
            contact = st.selectbox("Contact", ["cellular", "telephone"])
            default = st.selectbox("Default", ["no", "yes", "unknown"])
            
        submitted = st.form_submit_button("Run Prediction", type="primary")
        
        if submitted:
            # Construct standard payload (filled rest with baseline defaults)
            payload = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "housing": "yes", "loan": "no", "contact": contact,
                "month": "may", "day_of_week": "fri", "campaign": 1, "pdays": 999,
                "previous": 0, "poutcome": "nonexistent", "emp_var_rate": emp_var_rate,
                "cons_price_idx": cons_price_idx, "cons_conf_idx": -46.2, 
                "euribor3m": euribor3m, "nr_employed": 5099.1
            }
            try:
                res = httpx.post(MODEL_API_URL, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    prob = data['probability']
                    latency = data['latency_ms']
                    
                    # Nice, human-readable statements based on 0.5 threshold
                    if prob < 0.5:
                        st.success(f"🌟 **Good News!** It looks like this customer is happy and will stay. (Churn Risk: No) | ⚡ {latency:.1f}ms")
                    else:
                        st.warning(f"⚠️ **Attention Needed:** This customer is at high risk of churning. (Churn Risk: Yes) | ⚡ {latency:.1f}ms")
                else:
                    st.error("Model Service Error.")
            except Exception as e:
                st.error(f"Failed to reach model service: {e}")

# ==========================================
# PAGE 4: LOGS
# ==========================================
elif page == "📊 Prediction Logs":
    st.title("📊 Model Telemetry Logs")
    st.markdown("Raw inference logs pulled from Postgres.")
    
    logs = fetch_db_records(PredictionLog, limit=100)
    if logs:
        df = pd.DataFrame([{
            "Time": log.timestamp.strftime("%H:%M:%S"),
            "Version": log.model_version,
            "Pred": log.prediction,
            "Prob": round(log.probability, 3),
            "Euribor3m (Input)": log.input_data.get("euribor3m", "N/A") if log.input_data else "N/A",
            "Latency (ms)": round(log.latency_ms, 2)
        } for log in logs])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet.")

# ==========================================
# PAGE 5: DRIFT SIMULATOR
# ==========================================
elif page == "🧪 Drift Simulator":
    st.title("🧪 Trigger Macroeconomic Drift")
    st.markdown("Flood the model service with mutated data to trigger an automatic investigation.")
    
    if st.button("🚀 Execute Drift Attack", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        base_client = {
            "age": 30, "job": "blue-collar", "marital": "married", "education": "basic.9y",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "may", "day_of_week": "fri", "campaign": 1, "pdays": 999,
            "previous": 0, "poutcome": "nonexistent", "emp_var_rate": -1.8,
            "cons_price_idx": 92.893, "cons_conf_idx": -46.2, 
            "euribor3m": 1.299, "nr_employed": 5099.1
        }
        
        with httpx.Client() as client:
            status_text.text("Seeding normal baseline data...")
            for i in range(20):
                client.post(MODEL_API_URL, json=base_client)
                progress_bar.progress((i + 1) / 70)
                
            status_text.text("🚨 Injecting macroeconomic anomaly (Euribor spike)...")
            for i in range(50):
                drifted = base_client.copy()
                drifted["euribor3m"] = random.uniform(8.0, 12.0)
                drifted["job"] = "unknown"
                try:
                    client.post(MODEL_API_URL, json=drifted)
                except:
                    pass
                progress_bar.progress((20 + i + 1) / 70)
                time.sleep(0.05)
                
        status_text.text("✅ Attack complete. Check Inbox for agent triage!")
        st.balloons()