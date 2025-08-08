# app.py — Firestore-first (no CSV), 1-click forecast → table, Edit Data tab, Insights tab

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Firestore
import firebase_admin
from firebase_admin import credentials, firestore

# Local modules
from data_processing import load_from_firestore, create_advanced_features
from forecasting import generate_forecast

# ----------------- CONFIG -----------------
st.set_page_config(page_title="AI Sales & Customer Forecaster — 2025", layout="wide")

HISTORY_COLLECTION = "historical_data"
EVENTS_COLLECTION = "future_activities"
FORECAST_LOG_COLLECTION = "forecast_log"

# ----------------- STYLES (kept simple) -----------------
st.markdown("""
<style>
    .stButton > button { border-radius: 8px; font-weight: 600; padding: 10px 16px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ----------------- FIRESTORE INIT (uses st.secrets.firebase_credentials) -----------------
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = {
                "type": st.secrets.firebase_credentials.type,
                "project_id": st.secrets.firebase_credentials.project_id,
                "private_key_id": st.secrets.firebase_credentials.private_key_id,
                "private_key": st.secrets.firebase_credentials.private_key.replace("\\n", "\n"),
                "client_email": st.secrets.firebase_credentials.client_email,
                "client_id": st.secrets.firebase_credentials.client_id,
                "auth_uri": st.secrets.firebase_credentials.auth_uri,
                "token_uri": st.secrets.firebase_credentials.token_uri,
                "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url,
                "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error. Check your Streamlit secrets → firebase_credentials. Details: {e}")
        return None

db = init_firestore()
if db is None:
    st.stop()

# ----------------- STATE -----------------
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "customer_model" not in st.session_state:
    st.session_state.customer_model = None
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()

# ----------------- TABS -----------------
tab_forecast, tab_edit, tab_insights = st.tabs(["📈 Forecast", "✏️ Edit Data", "🧠 Insights"])

# =========================== FORECAST TAB ===========================
with tab_forecast:
    st.header("📈 Forecast")

    # Load data straight from Firestore
    history_df = load_from_firestore(db, HISTORY_COLLECTION)
    events_df = load_from_firestore(db, EVENTS_COLLECTION)

    if history_df.empty:
        st.error(f"No documents found in '{HISTORY_COLLECTION}'. Add data in Firestore then refresh.")
        st.stop()

    st.caption("Preview of history (last 30 rows)")
    st.dataframe(history_df.sort_values("date").tail(30), use_container_width=True)

    # Controls
    left, right = st.columns([1,1])
    with left:
        periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=35, value=15, step=1)
    with right:
        run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    # Generate forecast
    if run_btn:
        with st.spinner("Training & forecasting..."):
            forecast_df, customer_model = generate_forecast(history_df, events_df, periods=int(periods))
            st.session_state.forecast_df = forecast_df
            st.session_state.customer_model = customer_model

        if not forecast_df.empty:
            # Save to forecast_log
            try:
                batch = db.batch()
                log_ref = db.collection(FORECAST_LOG_COLLECTION)
                generated_ts = pd.Timestamp.utcnow().to_pydatetime()
                for _, r in forecast_df.iterrows():
                    doc_id = pd.to_datetime(r["ds"]).strftime("%Y-%m-%d")
                    batch.set(
                        log_ref.document(doc_id),
                        {
                            "generated_on": generated_ts,
                            "forecast_for_date": r["ds"].to_pydatetime() if hasattr(r["ds"], "to_pydatetime") else pd.to_datetime(r["ds"]).to_pydatetime(),
                            "predicted_sales": float(r["forecast_sales"]),
                            "predicted_customers": int(r["forecast_customers"]),
                            "predicted_atv": float(r["forecast_atv"]),
                        },
                        merge=True
                    )
                batch.commit()
                st.success("Forecast generated and logged to Firestore.")
            except Exception as e:
                st.warning(f"Forecast generated but failed to log: {e}")

    # Show table
    if not st.session_state.forecast_df.empty:
        st.subheader("🔮 Forecast (table)")
        show = st.session_state.forecast_df.copy()
        show["forecast_customers"] = show["forecast_customers"].astype(int)
        show["forecast_atv"] = show["forecast_atv"].round(2)
        show["forecast_sales"] = show["forecast_sales"].round(2)
        st.dataframe(show, use_container_width=True, height=420)

        # download
        buf = io.StringIO()
        show.to_csv(buf, index=False)
        st.download_button("Download forecast.csv", buf.getvalue(), "forecast.csv", "text/csv")

# =========================== EDIT DATA TAB ===========================
with tab_edit:
    st.header("✏️ Edit Data (Firestore)")
    df_edit = load_from_firestore(db, HISTORY_COLLECTION)
    if df_edit.empty:
        st.info(f"No docs in '{HISTORY_COLLECTION}'.")
    else:
        # keep clean user-facing columns
        display_cols = [c for c in df_edit.columns if c not in ("doc_id",)]
        df_edit_view = df_edit[display_cols].sort_values("date", ascending=False)
        edited = st.data_editor(df_edit_view, use_container_width=True, num_rows="dynamic")

        if st.button("💾 Save Changes"):
            try:
                batch = db.batch()
                col_ref = db.collection(HISTORY_COLLECTION)
                for _, row in edited.iterrows():
                    if pd.isna(row.get("date")):
                        continue
                    # use yyyy-mm-dd as doc id
                    doc_id = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                    payload = {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    # cast numeric fields if present
                    for k in ("customers", "sales", "atv", "add_on_sales"):
                        if k in payload and payload[k] is not None:
                            try:
                                payload[k] = float(payload[k])
                            except Exception:
                                pass
                    payload["date"] = pd.to_datetime(payload["date"]).to_pydatetime()
                    batch.set(col_ref.document(doc_id), payload, merge=True)
                batch.commit()
                st.success("Saved to Firestore.")
            except Exception as e:
                st.error(f"Save failed: {e}")

# =========================== INSIGHTS TAB ===========================
with tab_insights:
    st.header("🧠 Forecast Insights")
    if st.session_state.customer_model is None or st.session_state.forecast_df.empty:
        st.info("Run a forecast first to see insights.")
    else:
        model = st.session_state.customer_model
        # Feature importance (works for LGBMRegressor)
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                # rebuild features from history (same feature builder)
                feats_df = create_advanced_features(
                    load_from_firestore(db, HISTORY_COLLECTION),
                    load_from_firestore(db, EVENTS_COLLECTION)
                )
                # columns used during training in forecasting.py
                FEATURE_COLS = [c for c in feats_df.columns if c not in ("date","customers","sales","atv")]
                imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False)
                st.subheader("ATV/Customers model — feature importance")
                st.dataframe(imp_df.head(40), use_container_width=True)
            else:
                st.info("Model does not expose feature_importances_.")
        except Exception as e:
            st.warning(f"Could not compute importances: {e}")
