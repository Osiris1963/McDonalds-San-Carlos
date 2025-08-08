# Firestore-first (no CSV), upgraded logic, tabs: Forecast / Edit Data / Insights
# Secrets key supported: [firebase_credentials] (your format) or [gcp_service_account]

import io, os, json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _FB_OK = True
except Exception:
    _FB_OK = False

from data_processing import (
    load_history_from_firestore,
    load_events_from_firestore,
    ensure_datetime_index,
    prepare_history,
    add_future_calendar,
    build_event_frame_from_df,
    pretty_money,
)
from forecasting import (
    forecast_customers_with_trend_correction,
    forecast_atv_direct,
    combine_sales_and_bands,
    backtest_metrics,
    _train_direct_horizon_models,   # Insights
    _recent_trend_multiplier,       # Insights
)

st.set_page_config(page_title="AI Sales & Customer Forecaster — 2025", layout="wide")
st.title("AI Sales & Customer Forecaster — 2025 Edition")
st.caption("Firestore-first · ETS+damped trend + recent-trend (Customers) · Direct multi-horizon (ATV) · Events/PH holidays/paydays · Backtesting · Insights")

# ---------- Firestore init ----------
@st.cache_resource
def get_firestore_client():
    if not _FB_OK:
        return None
    try:
        if not firebase_admin._apps:
            cred = None
            for key in ("firebase_credentials", "gcp_service_account"):
                if key in st.secrets:
                    info = dict(st.secrets[key])
                    if "private_key" in info:
                        info["private_key"] = str(info["private_key"]).replace("\\n", "\n")
                    cred = credentials.Certificate(info); break
            if cred is None and "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
                info = json.loads(str(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]))
                if "private_key" in info:
                    info["private_key"] = info["private_key"].replace("\\n", "\n")
                cred = credentials.Certificate(info)
            if cred is None:
                path = st.secrets.get("service_account_file", None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or ("serviceAccountKey.json" if os.path.exists("serviceAccountKey.json") else None)
                if path: cred = credentials.Certificate(path)
            if cred is None: return None
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.sidebar.error(f"Firestore init error: {e}")
        return None

db = get_firestore_client()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Firestore Collections")
    HIST_COLLECTION = st.text_input("History (read/write)", value="historical_data")
    EVENTS_COLLECTION = st.text_input("Future events (read)", value="future_activities")
    FORECASTS_COLLECTION = st.text_input("Forecasts (write)", value="forecasts")
    BACKTEST_COLLECTION = st.text_input("Backtest metrics (write)", value="backtest_metrics")
    if db is None: st.error("Firestore not configured. Ensure secrets contain [firebase_credentials].")
    else: st.success("Firestore connected.")
    save_runs = st.toggle("Save new runs to Firestore", value=(db is not None))

if db is None:
    st.stop()

tab_forecast, tab_edit, tab_insights = st.tabs(["📈 Forecast", "✏️ Edit Data", "🧠 Insights"])

if "hist_data" not in st.session_state: st.session_state["hist_data"] = None
if "latest_forecast" not in st.session_state: st.session_state["latest_forecast"] = None

# ---------- Forecast tab ----------
with tab_forecast:
    st.subheader("Run Forecast")

    hist_raw = load_history_from_firestore(db, HIST_COLLECTION)
    if hist_raw.empty:
        st.error(f"No documents found in '{HIST_COLLECTION}'."); st.stop()

    hist = ensure_datetime_index(hist_raw)
    hist = prepare_history(hist)
    st.session_state["hist_data"] = hist

    st.caption("Recent history (last 30 rows)")
    st.dataframe(hist.tail(30), use_container_width=True)

    H = st.number_input("Forecast horizon (days)", 7, 35, 15, 1)

    left, right = st.columns([1,1])
    with left:
        st.subheader("Options")
        apply_caps = st.checkbox("Apply weekday growth caps (Customers)", True)
        decay_lambda = st.slider("Recent-trend decay (Customers)", 0.75, 0.99, 0.90, 0.01)
        atv_guard = st.slider("ATV guardrail (MAD×)", 2.0, 5.0, 3.0, 0.5)
        show_bands = st.checkbox("Show P10/P50/P90 bands", True)
    with right:
        st.subheader("Events/Uplifts")
        ev_db = load_events_from_firestore(db, EVENTS_COLLECTION, horizon=int(H), start_date=(hist.index.max() + pd.Timedelta(days=1)))
        ev_edit = st.data_editor(ev_db, use_container_width=True, num_rows="dynamic", key="ev_editor")

    run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    bt_btn  = st.button("🧪 Backtest (Rolling Origin)", use_container_width=True)

    def run_pipeline(df_hist: pd.DataFrame, H: int, events_future_df: pd.DataFrame) -> pd.DataFrame:
        future_cal = add_future_calendar(df_hist, periods=H)
        ev = build_event_frame_from_df(events_future_df, index=future_cal.index)
        cust_fc, cust_bands = forecast_customers_with_trend_correction(
            hist=df_hist, future_cal=future_cal, H=H,
            decay_lambda=decay_lambda, apply_weekday_caps=apply_caps,
            event_uplift_pct=ev.get("uplift_customers_pct"), return_bands=show_bands,
        )
        atv_fc, atv_bands = forecast_atv_direct(
            hist=df_hist, future_cal=future_cal, H=H,
            guardrail_mad_mult=atv_guard,
            event_uplift_pct=ev.get("uplift_atv_pct"), return_bands=show_bands,
        )
        return combine_sales_and_bands(
            dates=future_cal.index, customers=cust_fc, customers_bands=cust_bands,
            atv=atv_fc, atv_bands=atv_bands, return_bands=show_bands,
        )

    def save_forecast_to_firestore(df_fc: pd.DataFrame):
        if not (save_runs and db is not None) or df_fc is None or df_fc.empty: return
        batch = db.batch(); col = db.collection(FORECASTS_COLLECTION); run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for dt, row in df_fc.iterrows():
            payload = dict(row); payload["forecast_for_date"] = dt.to_pydatetime(); payload["generated_on"] = datetime.utcnow()
            batch.set(col.document(f"{run_id}_{dt.date()}"), payload, merge=True)
        batch.commit()

    if run_btn:
        with st.spinner("Forecasting..."):
            fc = run_pipeline(hist, int(H), ev_edit)
        st.session_state["latest_forecast"] = fc
        save_forecast_to_firestore(fc)

    if bt_btn:
        with st.spinner("Running backtest..."):
            metrics = backtest_metrics(hist, horizon=int(H), folds=6)
        st.subheader("📊 Backtest Metrics"); st.dataframe(metrics, use_container_width=True)
        if save_runs and db is not None and not metrics.empty:
            batch = db.batch(); col = db.collection(BACKTEST_COLLECTION); run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            for _, r in metrics.iterrows(): batch.set(col.document(f"{run_id}_{r['cutoff'].date()}"), r.to_dict(), merge=True)
            batch.commit(); st.success("Backtest metrics saved.")

    if st.session_state["latest_forecast"] is not None:
        st.subheader("🔮 Forecast (table)")
        pretty = st.session_state["latest_forecast"].copy()
        for c in pretty.columns:
            if c.startswith(("atv", "sales")): pretty[c] = pretty[c].apply(pretty_money)
            else: pretty[c] = pretty[c].round(0).astype(int)
        st.dataframe(pretty, use_container_width=True, height=420)
        buf = io.StringIO(); st.session_state["latest_forecast"].to_csv(buf); st.download_button("Download forecast.csv", buf.getvalue(), "forecast.csv", "text/csv")

# ---------- Edit tab ----------
with tab_edit:
    st.subheader("Edit History Data (Firestore)")
    df_edit = load_history_from_firestore(db, HIST_COLLECTION, raw=True)
    if df_edit.empty: st.info(f"No docs in '{HIST_COLLECTION}'.")
    else:
        if "date" in df_edit.columns: df_edit["date"] = pd.to_datetime(df_edit["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        edited = st.data_editor(df_edit, use_container_width=True, num_rows="dynamic", key="edit_history")
        if st.button("💾 Save Changes to Firestore"):
            batch = db.batch(); col = db.collection(HIST_COLLECTION)
            for _, r in edited.iterrows():
                if pd.isna(r.get("date")): continue
                doc_id = str(r["date"]); payload = dict(r.dropna())
                for k in ("customers","sales","atv","add_on_sales"):
                    if k in payload:
                        try: payload[k] = float(payload[k])
                        except Exception: pass
                try: payload["date"] = pd.to_datetime(payload["date"]).to_pydatetime()
                except Exception: pass
                batch.set(col.document(doc_id), payload, merge=True)
            batch.commit(); st.success("Saved.")

# ---------- Insights tab ----------
with tab_insights:
    st.subheader("Model Insights")
    hist_for_insights = st.session_state.get("hist_data")
    if hist_for_insights is None or hist_for_insights.empty: st.info("No history available.")
    else:
        st.caption("Recent-trend multiplier used for Customers (higher = stronger recent lift).")
        try:
            mult = _recent_trend_multiplier(hist_for_insights, decay_lambda=0.90)
            st.metric("Recent-trend multiplier", f"{mult:.3f}")
        except Exception as e:
            st.warning(f"Trend metric failed: {e}")
        H_ins = st.number_input("Horizons for importance (ATV)", 7, 30, 15, 1, key="H_insights")
        try:
            models, feats = _train_direct_horizon_models(hist_for_insights, int(H_ins))
            importances = [m.feature_importances_ for m in models if hasattr(m, "feature_importances_")]
            if importances:
                imp = np.mean(np.vstack(importances), axis=0)
                imp_df = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
                imp_df["importance"] = imp_df["importance"].round(3)
                st.dataframe(imp_df, use_container_width=True, height=360)
            else:
                st.info("No feature importances available (very short history?).")
        except Exception as e:
            st.warning(f"Insights failed: {e}")
