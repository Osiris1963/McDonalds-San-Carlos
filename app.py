# app.py
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

from data_processing import load_from_firestore
from forecasting import generate_forecast

# ---------------------------
# Page Config & Styles
# ---------------------------
st.set_page_config(page_title="Sales & Customers Forecaster", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* Tidy up the look a bit */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Firestore Connection
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_db_client(_service_json: dict | None):
    """
    Returns a Firestore client. If firebase_admin is already initialized, reuse it.
    Otherwise initialize from a provided service account dict or st.secrets["firebase"].
    """
    if firestore is None:
        return None

    if not firebase_admin._apps:
        try:
            if _service_json:
                cred = credentials.Certificate(_service_json)
            else:
                # Try secrets (if configured in Streamlit Cloud)
                cred = credentials.Certificate(st.secrets["firebase"])  # type: ignore
            firebase_admin.initialize_app(cred)
        except Exception:
            return None
    return firestore.client()

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("🔐 Firestore")
    auth_mode = st.radio("Auth source", ["Secrets", "Paste JSON"], horizontal=True)
    svc_json = None
    if auth_mode == "Paste JSON":
        svc_text = st.text_area("Service Account JSON", height=180, help="Paste the full service account JSON here.")
        if svc_text.strip():
            import json
            try:
                svc_json = json.loads(svc_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                svc_json = None

    db = get_db_client(svc_json)

    st.header("📦 Collections")
    history_col = st.text_input("History collection", value="history")
    events_col = st.text_input("Events collection (optional)", value="events")
    logs_col = st.text_input("Forecast logs collection", value="forecast_logs")

    st.header("⚙️ Forecast Settings")
    horizon = st.number_input("Days to forecast", min_value=1, max_value=30, value=15, step=1)
    do_log = st.checkbox("Log forecast to Firestore", value=False)

st.title("📈 Daily Sales & Customers Forecast")

if db is None:
    st.error("Could not connect to Firestore. Check your credentials.")
    st.stop()

# ---------------------------
# Load Data
# ---------------------------
with st.spinner("Loading history..."):
    hist_df = load_from_firestore(db, history_col)

if hist_df.empty:
    st.info("No historical data found in your history collection.")
    st.stop()

events_df = pd.DataFrame()
if events_col:
    with st.spinner("Loading events..."):
        events_df = load_from_firestore(db, events_col)
        # Only keep date column as a calendar of event days
        if not events_df.empty:
            events_df = events_df[["date"]].drop_duplicates()

st.subheader("Historical Snapshot")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Records", f"{len(hist_df):,}")
with c2:
    st.metric("From", hist_df["date"].min().strftime("%b %d, %Y"))
with c3:
    st.metric("To", hist_df["date"].max().strftime("%b %d, %Y"))
with c4:
    total_sales = float(hist_df["sales"].sum())
    st.metric("Total Sales", f"₱{total_sales:,.2f}")

st.dataframe(hist_df.tail(30), use_container_width=True)

# ---------------------------
# Forecast
# ---------------------------
st.subheader("Forecast Result")
with st.spinner("Training models and generating forecast..."):
    fc_df, model_c = generate_forecast(hist_df.rename(columns={"date": "date"}), events_df, periods=int(horizon))

if fc_df.empty:
    st.warning("No forecast could be generated.")
    st.stop()

# KPI row
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Forecast Days", len(fc_df))
with k2:
    st.metric("Avg Customers", f"{fc_df['forecast_customers'].mean():,.0f}")
with k3:
    st.metric("Avg ATV", f"₱{fc_df['forecast_atv'].mean():,.2f}")

st.write("### Table")
st.dataframe(fc_df, use_container_width=True)

# ---------------------------
# Optional Logging
# ---------------------------
def log_forecast(db_client, collection_name: str, forecast_df: pd.DataFrame) -> bool:
    try:
        batch = db_client.batch()
        ts = datetime.utcnow().isoformat()
        for _, row in forecast_df.iterrows():
            # deterministic doc id: YYYY-MM-DD
            doc_id = pd.to_datetime(row["ds"]).strftime("%Y-%m-%d")
            ref = db_client.collection(collection_name).document(doc_id)
            payload = {
                "ds": row["ds"].to_pydatetime() if hasattr(row["ds"], "to_pydatetime") else pd.to_datetime(row["ds"]),
                "forecast_sales": float(row["forecast_sales"]),
                "forecast_customers": int(row["forecast_customers"]),
                "forecast_atv": float(row["forecast_atv"]),
                "logged_at": ts,
            }
            batch.set(ref, payload, merge=True)
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast: {e}")
        return False

if do_log:
    ok = log_forecast(db, logs_col, fc_df)
    if ok:
        st.success(f"Logged {len(fc_df)} forecast rows to '{logs_col}'.")

st.caption("Tip: To fine-tune, add more history and label special event days in your events collection.")
