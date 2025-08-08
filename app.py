import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Firebase (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _FIREBASE_AVAILABLE = True
except Exception:
    _FIREBASE_AVAILABLE = False

from data_processing import (
    ensure_datetime_index,
    prepare_history,
    add_future_calendar,
    build_event_frame,
    pretty_money,
)
from forecasting import (
    forecast_customers_with_trend_correction,
    forecast_atv_direct,
    combine_sales_and_bands,
    backtest_metrics,
)

# --------------------
# STREAMLIT CONFIG
# --------------------
st.set_page_config(page_title="AI Sales Forecaster (2025)", layout="wide")
st.title("AI Sales & Customer Forecaster — 2025 Edition")
st.caption("Hybrid: ETS (trend-first) for Customers + Direct Multi-Horizon LightGBM for ATV. With events, PH holidays, and paydays.")

# --------------------
# OPTIONAL FIREBASE INIT (non-blocking, no banner)
# --------------------
def get_firestore_client():
    """
    Initialize Firestore if credentials exist.
    Priority:
      1) st.secrets['gcp_service_account']  (dict)
      2) st.secrets['service_account_file'] (path)
      3) ENV SERVICE_ACCOUNT_PATH           (path)
    Returns firestore.Client or None.
    """
    if not _FIREBASE_AVAILABLE:
        return None
    try:
        if not firebase_admin._apps:
            cred = None
            # 1) Dict credentials from Streamlit Secrets
            if "gcp_service_account" in st.secrets:
                cred_info = dict(st.secrets["gcp_service_account"])
                cred = credentials.Certificate(cred_info)
            # 2) File path from secrets
            elif "service_account_file" in st.secrets and os.path.exists(st.secrets["service_account_file"]):
                cred = credentials.Certificate(st.secrets["service_account_file"])
            else:
                # 3) ENV var (or default file) if exists
                sa_path = os.getenv("SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
                if os.path.exists(sa_path):
                    cred = credentials.Certificate(sa_path)

            if cred is None:
                return None  # credentials not provided; run app without Firestore

            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception:
        # Never break the UI if Firebase fails
        return None

db = get_firestore_client()

# Sidebar toggle to save to Firestore only if db is available
with st.sidebar:
    st.subheader("Storage")
    if db is not None:
        save_to_firestore = st.toggle("Save runs to Firestore", value=True, help="Turn off if you want local-only runs.")
        st.success("Firestore connected.")
    else:
        save_to_firestore = False
        st.info("Firestore not configured. App runs normally without saving.")

# --------------------
# HELP / DATA FORMAT (kept like before)
# --------------------
with st.expander("📘 Data format (CSV)"):
    st.markdown(
        """
        **Required columns** (case-insensitive):
        - `date` (YYYY-MM-DD or parseable date)
        - `customers` and `sales` (ATV auto-computed), **or** `customers` and `atv`  
        **Granularity:** Daily
        """
    )

# --------------------
# CONTROLS (same layout/feel)
# --------------------
uploaded = st.file_uploader("Upload historical CSV", type=["csv"])
default_horizon = 15
H = st.number_input("Forecast horizon (days)", min_value=7, max_value=35, value=default_horizon, step=1)

left, right = st.columns([1,1])
with left:
    st.subheader("🔧 Options")
    apply_weekday_caps = st.checkbox("Apply realistic weekday growth caps for Customers", value=True)
    decay_lambda = st.slider("Recent-trend decay (Customers)", min_value=0.75, max_value=0.99, value=0.9, step=0.01)
    atv_guardrail_factor = st.slider("ATV guardrail (MAD multiplier)", min_value=2.0, max_value=5.0, value=3.0, step=0.5)
    show_bands = st.checkbox("Show P10/P50/P90 bands", value=True)

with right:
    st.subheader("🎯 Events / Uplifts")
    start_for_future = st.date_input("Start date for future calendar", value=(datetime.utcnow() + timedelta(days=1)).date())
    events_editor = pd.DataFrame({
        "date": pd.date_range(start=start_for_future, periods=H, freq="D"),
        "uplift_customers_pct": np.zeros(H, dtype=float),
        "uplift_atv_pct": np.zeros(H, dtype=float),
        "notes": ["" for _ in range(H)]
    })
    events_df = st.data_editor(events_editor, key="events_editor", num_rows="fixed", use_container_width=True)

run_cols = st.columns([1,1,1])
go = run_cols[0].button("🚀 Run Forecast")
do_backtest = run_cols[1].button("🧪 Backtest (Rolling Origin)")
download_forecast = run_cols[2].button("💾 Download latest forecast (CSV)")

if "latest_forecast" not in st.session_state:
    st.session_state["latest_forecast"] = None

# --------------------
# FIRESTORE SAVE HELPERS (only used when toggle is on)
# --------------------
def save_forecast_to_firestore(df_forecast, run_id):
    if not (save_to_firestore and db is not None):
        return
    batch = db.batch()
    for date_idx, row in df_forecast.iterrows():
        doc_ref = db.collection("forecasts").document(f"{run_id}_{date_idx.date()}")
        batch.set(doc_ref, row.to_dict())
    batch.commit()

def save_backtest_to_firestore(metrics_df, run_id):
    if not (save_to_firestore and db is not None):
        return
    batch = db.batch()
    for _, row in metrics_df.iterrows():
        doc_ref = db.collection("backtest_metrics").document(f"{run_id}_{row['cutoff'].date()}")
        batch.set(doc_ref, row.to_dict())
    batch.commit()

# --------------------
# VALIDATION + PIPELINE
# --------------------
def validate_and_prepare(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")
    df = ensure_datetime_index(df)
    df = prepare_history(df)
    return df

def run_pipeline(df_hist: pd.DataFrame, H: int, events_future: pd.DataFrame):
    future_cal = add_future_calendar(df_hist, periods=H)
    ev = build_event_frame(events_future)

    cust_fc, cust_bands = forecast_customers_with_trend_correction(
        hist=df_hist,
        future_cal=future_cal,
        H=H,
        decay_lambda=decay_lambda,
        apply_weekday_caps=apply_weekday_caps,
        event_uplift_pct=ev["uplift_customers_pct"],
        return_bands=show_bands,
    )

    atv_fc, atv_bands = forecast_atv_direct(
        hist=df_hist,
        future_cal=future_cal,
        H=H,
        guardrail_mad_mult=atv_guardrail_factor,
        event_uplift_pct=ev["uplift_atv_pct"],
        return_bands=show_bands,
    )

    combined = combine_sales_and_bands(
        dates=future_cal.index,
        customers=cust_fc,
        customers_bands=cust_bands,
        atv=atv_fc,
        atv_bands=atv_bands,
        return_bands=show_bands,
    )
    return combined

# --------------------
# MAIN
# --------------------
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        hist = validate_and_prepare(df_raw)

        st.write("✅ Data preview (last 30 rows):")
        st.dataframe(hist.tail(30), use_container_width=True)

        if go:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fc = run_pipeline(hist, H, events_df)
            st.session_state["latest_forecast"] = fc
            save_forecast_to_firestore(fc, run_id)

        if do_backtest:
            with st.spinner("Running backtest..."):
                metrics = backtest_metrics(hist, horizon=H, folds=6)
            st.subheader("📊 Backtest Metrics (Rolling Origin)")
            st.write(metrics)
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_backtest_to_firestore(metrics, run_id)

        if st.session_state["latest_forecast"] is not None:
            st.subheader("🔮 Forecast")
            fc = st.session_state["latest_forecast"].copy()
            pretty = fc.copy()
            for c in pretty.columns:
                if c.startswith("atv") or c.startswith("sales"):
                    pretty[c] = pretty[c].apply(pretty_money)
                else:
                    # customers columns
                    pretty[c] = pretty[c].round(0).astype(int)
            st.dataframe(pretty, use_container_width=True, height=400)

            st.line_chart(fc[["customers_p50"]].rename(columns={"customers_p50":"customers"}))
            st.line_chart(fc[["atv_p50"]].rename(columns={"atv_p50":"ATV (₱)"}))
            st.line_chart(fc[["sales_p50"]].rename(columns={"sales_p50":"Sales (₱)"}))

            if download_forecast:
                buf = io.StringIO()
                fc.to_csv(buf, index=True)
                st.download_button(
                    "Download forecast.csv",
                    buf.getvalue(),
                    file_name="forecast.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV to begin.")
