"""
AI Sales & Customer Forecaster — 2025 Edition (Streamlit + Firestore)

How to connect Firestore (pick ONE):
1) Streamlit Cloud Secrets (recommended):
   - In your repo: .streamlit/secrets.toml  OR via the Streamlit UI "Secrets"
   - Paste your service account JSON under the key gcp_service_account:
     [gcp_service_account]
     type = "service_account"
     project_id = "YOUR_PROJECT_ID"
     private_key_id = "..."
     private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
     client_email = "firebase-adminsdk-xyz@YOUR_PROJECT_ID.iam.gserviceaccount.com"
     client_id = "..."
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xyz%40YOUR_PROJECT_ID.iam.gserviceaccount.com"

2) JSON file:
   - Upload serviceAccountKey.json to your deployment and set env var:
     SERVICE_ACCOUNT_PATH=serviceAccountKey.json
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# --- Optional Firebase ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _FB_OK = True
except Exception:
    _FB_OK = False

# Your local modules (from the files I already gave you)
from data_processing import (
    ensure_datetime_index, prepare_history, add_future_calendar,
    build_event_frame, pretty_money,
)
from forecasting import (
    forecast_customers_with_trend_correction, forecast_atv_direct,
    combine_sales_and_bands, backtest_metrics,
    _train_direct_horizon_models, _recent_trend_multiplier,  # used in Insights
)

# --------------------
# Streamlit Config
# --------------------
st.set_page_config(page_title="AI Sales & Customer Forecaster — 2025", layout="wide")
st.title("AI Sales & Customer Forecaster — 2025 Edition")
st.caption("Forecast tab → button → table. Edit Data pulls from your Firestore. Insights shows model signals.")

# --------------------
# Firestore (optional, non-blocking)
# --------------------
def get_firestore_client():
    """Initialize Firestore if credentials exist. Never break the UI."""
    if not _FB_OK:
        return None
    try:
        if not firebase_admin._apps:
            cred = None
            # 1) Dict credentials from Streamlit Secrets
            if "gcp_service_account" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["gcp_service_account"]))
            # 2) Path from secrets
            elif "service_account_file" in st.secrets and os.path.exists(st.secrets["service_account_file"]):
                cred = credentials.Certificate(st.secrets["service_account_file"])
            else:
                # 3) Env var or default file
                sa_path = os.getenv("SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
                if os.path.exists(sa_path):
                    cred = credentials.Certificate(sa_path)

            if cred is None:
                return None

            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception:
        return None

db = get_firestore_client()

# Sidebar controls for DB usage + collection names
with st.sidebar:
    st.header("Storage")
    hist_collection = st.text_input("History collection (read/write)", value="daily_history")
    fc_collection = st.text_input("Forecasts collection (write)", value="forecasts")
    bt_collection = st.text_input("Backtest collection (write)", value="backtest_metrics")

    if db is None:
        st.info("Firestore not configured. App runs without saving.")
    else:
        st.success("Firestore connected.")

    save_to_firestore = st.toggle("Save new runs to Firestore", value=(db is not None))

# --------------------
# Tabs (like before)
# --------------------
tab_forecast, tab_edit, tab_insights = st.tabs(["📈 Forecast", "✏️ Edit Data", "🧠 Insights"])

# Keep state
if "hist_data" not in st.session_state:
    st.session_state["hist_data"] = None
if "latest_forecast" not in st.session_state:
    st.session_state["latest_forecast"] = None

# --------------------
# FORECAST TAB
# --------------------
with tab_forecast:
    with st.expander("📘 Data format (CSV)"):
        st.markdown(
            """
            **Required (case-insensitive):**
            - `date`
            - `customers` and `sales` (ATV auto-computed), **or** `customers` and `atv`  
            **Granularity:** Daily
            """
        )

    uploaded = st.file_uploader("Upload CSV (optional, overrides DB)", type=["csv"], key="upload_csv")
    H = st.number_input("Forecast horizon (days)", 7, 35, 15, 1)

    left, right = st.columns([1,1])
    with left:
        st.subheader("Options")
        apply_caps = st.checkbox("Apply weekday growth caps (Customers)", True)
        decay_lambda = st.slider("Recent-trend decay (Customers)", 0.75, 0.99, 0.90, 0.01)
        atv_guard = st.slider("ATV guardrail (MAD×)", 2.0, 5.0, 3.0, 0.5)
        show_bands = st.checkbox("Show P10/P50/P90 bands", True)
    with right:
        st.subheader("Events / Uplifts")
        start_for_future = st.date_input("Start date", value=(datetime.utcnow()+timedelta(days=1)).date())
        ev_df = pd.DataFrame({
            "date": pd.date_range(start=start_for_future, periods=H, freq="D"),
            "uplift_customers_pct": np.zeros(H),
            "uplift_atv_pct": np.zeros(H),
            "notes": ["" for _ in range(H)],
        })
        events_df = st.data_editor(ev_df, num_rows="fixed", use_container_width=True, key="events_editor")

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    run_btn = col_btn1.button("🚀 Run Forecast", use_container_width=True, key="run_forecast_btn")
    backtest_btn = col_btn2.button("🧪 Backtest (Rolling Origin)", use_container_width=True)
    dl_btn = col_btn3.button("💾 Download latest forecast (CSV)", use_container_width=True, key="dl_btn")

    # Load history: uploaded CSV overrides DB
    hist = None
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            df_raw.columns = [c.strip().lower() for c in df_raw.columns]
            if "date" not in df_raw.columns:
                raise ValueError("CSV must include a 'date' column.")
            hist = ensure_datetime_index(df_raw)
            hist = prepare_history(hist)
        except Exception as e:
            st.error(f"CSV error: {e}")
            hist = None
    elif db is not None:
        # Pull from Firestore history collection (robust on date field type)
        try:
            try:
                q = db.collection(hist_collection).order_by("date")
                docs = list(q.stream())
            except Exception:
                docs = list(db.collection(hist_collection).stream())

            rows = []
            for d in docs:
                rec = d.to_dict() or {}
                rec_date = rec.get("date", d.id)

                # Accept Firestore Timestamp, datetime/date, or string
                if hasattr(rec_date, "to_datetime"):
                    rec_date = rec_date.to_datetime()
                if hasattr(rec_date, "isoformat"):
                    rec_date = rec_date.strftime("%Y-%m-%d")
                rec["date"] = str(rec_date)

                for k in ("customers", "sales", "atv"):
                    if k in rec:
                        try:
                            rec[k] = float(rec[k])
                        except Exception:
                            rec[k] = None
                rows.append(rec)

            if rows:
                hist = pd.DataFrame(rows)
                hist.columns = [c.lower() for c in hist.columns]
                hist = ensure_datetime_index(hist)
                hist = prepare_history(hist)
        except Exception as e:
            st.warning(f"Could not load from Firestore: {e}")

    if hist is not None:
        st.session_state["hist_data"] = hist
        st.write("✅ Data preview (last 30 rows)")
        st.dataframe(hist.tail(30), use_container_width=True)
    else:
        st.info("Upload a CSV or configure Firestore history collection.")

    def run_pipeline(df_hist: pd.DataFrame, H: int, events_future: pd.DataFrame) -> pd.DataFrame:
        future_cal = add_future_calendar(df_hist, periods=H)
        ev = build_event_frame(events_future)

        cust_fc, cust_bands = forecast_customers_with_trend_correction(
            hist=df_hist, future_cal=future_cal, H=H,
            decay_lambda=decay_lambda, apply_weekday_caps=apply_caps,
            event_uplift_pct=ev["uplift_customers_pct"], return_bands=show_bands,
        )
        atv_fc, atv_bands = forecast_atv_direct(
            hist=df_hist, future_cal=future_cal, H=H,
            guardrail_mad_mult=atv_guard, event_uplift_pct=ev["uplift_atv_pct"],
            return_bands=show_bands,
        )
        combined = combine_sales_and_bands(
            dates=future_cal.index, customers=cust_fc, customers_bands=cust_bands,
            atv=atv_fc, atv_bands=atv_bands, return_bands=show_bands,
        )
        return combined

    def save_forecast_to_firestore(df_forecast: pd.DataFrame, run_id: str):
        if not (save_to_firestore and db is not None):
            return
        batch = db.batch()
        for dt, row in df_forecast.iterrows():
            doc_ref = db.collection(fc_collection).document(f"{run_id}_{dt.date()}")
            batch.set(doc_ref, row.to_dict())
        batch.commit()

    if run_btn:
        if st.session_state["hist_data"] is None:
            st.error("No data. Upload CSV or configure Firestore history.")
        else:
            with st.spinner("Forecasting..."):
                fc = run_pipeline(st.session_state["hist_data"], H, events_df)
            st.session_state["latest_forecast"] = fc
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_forecast_to_firestore(fc, run_id)

    if backtest_btn and st.session_state["hist_data"] is not None:
        with st.spinner("Running backtest..."):
            metrics = backtest_metrics(st.session_state["hist_data"], horizon=H, folds=6)
        st.subheader("📊 Backtest Metrics (Rolling Origin)")
        st.dataframe(metrics, use_container_width=True)
        if save_to_firestore and db is not None:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            batch = db.batch()
            for _, r in metrics.iterrows():
                doc_ref = db.collection(bt_collection).document(f"{run_id}_{r['cutoff'].date()}")
                batch.set(doc_ref, r.to_dict())
            batch.commit()
            st.success("Backtest metrics saved to Firestore.")

    if st.session_state["latest_forecast"] is not None:
        st.subheader("🔮 Forecast (table)")
        fc = st.session_state["latest_forecast"].copy()
        pretty = fc.copy()
        for c in pretty.columns:
            if c.startswith(("atv", "sales")):
                pretty[c] = pretty[c].apply(pretty_money)
            else:
                pretty[c] = pretty[c].round(0).astype(int)
        st.dataframe(pretty, use_container_width=True, height=420)

        if dl_btn:
            buf = io.StringIO()
            st.session_state["latest_forecast"].to_csv(buf)
            st.download_button("Download forecast.csv", buf.getvalue(), "forecast.csv", "text/csv")

# --------------------
# EDIT DATA TAB (read/write Firestore)
# --------------------
with tab_edit:
    st.subheader("Edit History Data")
    if db is None:
        st.info("Firestore not configured. Configure credentials to use Edit Data.")
    else:
        try:
            try:
                q = db.collection(hist_collection).order_by("date")
                docs = list(q.stream())
            except Exception:
                docs = list(db.collection(hist_collection).stream())

            rows = []
            for d in docs:
                rec = d.to_dict() or {}
                rec_date = rec.get("date", d.id)
                if hasattr(rec_date, "to_datetime"):
                    rec_date = rec_date.to_datetime()
                if hasattr(rec_date, "isoformat"):
                    rec_date = rec_date.strftime("%Y-%m-%d")
                rec["date"] = str(rec_date)
                rows.append(rec)

            if not rows:
                st.warning(f"No documents found in '{hist_collection}'.")
            else:
                df_edit = pd.DataFrame(rows)
                df_edit.columns = [c.lower() for c in df_edit.columns]
                if "date" in df_edit.columns:
                    df_edit["date"] = pd.to_datetime(df_edit["date"], errors="coerce").dt.strftime("%Y-%m-%d")

                st.write("Tip: Keep columns: date, customers, sales, atv")
                edited = st.data_editor(df_edit, use_container_width=True, num_rows="dynamic", key="edit_data")

                if st.button("💾 Save Changes to Firestore"):
                    batch = db.batch()
                    for _, r in edited.iterrows():
                        if pd.isna(r.get("date")):
                            continue
                        doc_id = str(r["date"])
                        record = dict(r.dropna())
                        for k in ("customers", "sales", "atv"):
                            if k in record:
                                try:
                                    record[k] = float(record[k])
                                except Exception:
                                    record[k] = None
                        batch.set(db.collection(hist_collection).document(doc_id), record)
                    batch.commit()
                    st.success("Saved!")
        except Exception as e:
            st.error(f"Edit Data error: {e}")

# --------------------
# INSIGHTS TAB
# --------------------
with tab_insights:
    st.subheader("Model Insights")
    st.caption("ATV LightGBM feature importance (avg across horizons) and the recent-trend multiplier used for customers.")

    hist_for_insights = st.session_state.get("hist_data", None)
    H_ins = st.number_input("H (for insights training)", 7, 30, 15, 1, key="H_insights")

    if hist_for_insights is None:
        st.info("No history loaded yet. Upload CSV or load from Firestore on the Forecast tab.")
    else:
        try:
            # Customers trend multiplier
            mult = _recent_trend_multiplier(hist_for_insights, decay_lambda=0.90)
            st.metric("Recent-trend multiplier (Customers)", f"{mult:.3f}")

            # Train direct-horizon models (like in forecast) to get feature importances
            models, feats = _train_direct_horizon_models(hist_for_insights, H_ins)
            importances = []
            for m in models:
                if hasattr(m, "feature_importances_"):
                    importances.append(m.feature_importances_)
            if importances:
                imp_avg = np.mean(np.vstack(importances), axis=0)
                imp_df = pd.DataFrame({"feature": feats, "importance": imp_avg}).sort_values("importance", ascending=False)
                imp_df["importance"] = imp_df["importance"].round(3)
                st.write("ATV feature importance (avg across horizons):")
                st.dataframe(imp_df, use_container_width=True, height=360)
            else:
                st.info("No importances available (very short history?).")
        except Exception as e:
            st.warning(f"Could not compute insights: {e}")
