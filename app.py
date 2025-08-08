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

from data_processing import (
    ensure_datetime_index, prepare_history, add_future_calendar,
    build_event_frame, pretty_money,
)
from forecasting import (
    forecast_customers_with_trend_correction, forecast_atv_direct,
    combine_sales_and_bands, backtest_metrics,
    # we’ll import private helpers for Insights (safe)
)
# Import internal helpers for Insights (feature importance & multiplier)
from forecasting import _build_atv_feature_matrix  # type: ignore
from forecasting import _train_direct_horizon_models  # type: ignore
from forecasting import _recent_trend_multiplier  # type: ignore

# --------------------
# Streamlit Config
# --------------------
st.set_page_config(page_title="AI Sales & Customer Forecaster — 2025", layout="wide")

# --------------------
# Firestore (optional, non-blocking)
# --------------------
def get_firestore_client():
    if not _FB_OK:
        return None
    try:
        if not firebase_admin._apps:
            cred = None
            if "gcp_service_account" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["gcp_service_account"]))
            elif "service_account_file" in st.secrets and os.path.exists(st.secrets["service_account_file"]):
                cred = credentials.Certificate(st.secrets["service_account_file"])
            else:
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

# Sidebar controls for DB usage
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
st.title("AI Sales & Customer Forecaster — 2025 Edition")
st.caption("Forecast tab → button → table. Edit Data pulls from your database. Insights shows model signals.")

tab_forecast, tab_edit, tab_insights = st.tabs(["📈 Forecast", "✏️ Edit Data", "🧠 Insights"])

# Keep forecast controls on Forecast tab (familiar flow)
with tab_forecast:
    with st.expander("📘 Data format (CSV)"):
        st.markdown(
            """
            **Required (case-insensitive):**
            - `date`
            - `customers` and `sales` (ATV auto-computed) **or** `customers` and `atv`
            Daily granularity.
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
    dl_btn = col_btn3.button("💾 Download latest forecast (CSV)", use_container_width=True, key="dl_btn")

    if "hist_data" not in st.session_state:
        st.session_state["hist_data"] = None
    if "latest_forecast" not in st.session_state:
        st.session_state["latest_forecast"] = None

    # Load history: uploaded CSV beats DB; else try DB; else wait.
    hist = None
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        try:
            hist = ensure_datetime_index(df_raw)
            hist = prepare_history(hist)
        except Exception as e:
            st.error(f"CSV error: {e}")
            hist = None
    elif db is not None:
        # Pull from Firestore history collection (expects documents keyed by date or having a 'date' field)
        try:
            docs = db.collection(hist_collection).stream()
            rows = []
            for d in docs:
                rec = d.to_dict()
                # accept either 'date' field or use doc id
                rec_date = rec.get("date", d.id)
                rows.append(rec | {"date": rec_date})
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

    def run_pipeline(df_hist: pd.DataFrame, H: int, events_future: pd.DataFrame):
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
# EDIT DATA TAB (pulls from DB, editable, save back)
# --------------------
with tab_edit:
    st.subheader("Edit History Data")
    if db is None:
        st.info("Firestore not configured. Configure credentials to use Edit Data.")
    else:
        # Load
        docs = db.collection(hist_collection).stream()
        rows = []
        for d in docs:
            rec = d.to_dict()
            rec_date = rec.get("date", d.id)
            rows.append(rec | {"date": rec_date})
        if not rows:
            st.warning(f"No documents found in '{hist_collection}'.")
        else:
            df_edit = pd.DataFrame(rows)
            # normalize
            df_edit.columns = [c.lower() for c in df_edit.columns]
            # ensure date string
            if "date" in df_edit.columns:
                df_edit["date"] = pd.to_datetime(df_edit["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.write("Tip: Keep columns: date, customers, sales, atv")
            edited = st.data_editor(df_edit, use_container_width=True, num_rows="dynamic", key="edit_data")

            col_save1, col_save2 = st.columns([1,3])
            if col_save1.button("💾 Save Changes to Firestore"):
                # write back each row; document id = date
                batch = db.batch()
                for _, r in edited.iterrows():
                    if pd.isna(r.get("date", None)):
                        continue
                    doc_id = str(r["date"])
                    batch.set(db.collection(hist_collection).document(doc_id), dict(r.dropna()))
                batch.commit()
                st.success("Saved!")

# --------------------
# INSIGHTS TAB
# --------------------
with tab_insights:
    st.subheader("Model Insights")
    st.caption("LightGBM feature importance for ATV (averaged across horizons) and recent-trend multiplier used for Customers.")

    hist_for_insights = st.session_state.get("hist_data", None)
    H_ins = st.number_input("H (for insights training)", 7, 30, 15, 1, key="H_insights")

    if hist_for_insights is None:
        st.info("No history loaded yet. Upload CSV or load from Firestore on Forecast tab.")
    else:
        try:
            # Customers trend multiplier
            mult = _recent_trend_multiplier(hist_for_insights, decay_lambda=0.90)
            st.metric("Recent-trend multiplier (Customers)", f"{mult:.3f}")

            # Train direct horizon models (like in forecast) to get feature importances
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
