import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

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

st.set_page_config(page_title="AI Sales Forecaster (2025)", layout="wide")

st.title("AI Sales & Customer Forecaster — 2025 Edition")
st.caption("Hybrid: ETS (trend-first) for Customers + Direct Multi-Horizon LightGBM for ATV. With events, PH holidays, and paydays.")

with st.expander("📘 Data format (CSV)"):
    st.markdown(
        """
        **Required columns** (case-insensitive):
        - `date` (YYYY-MM-DD or any parseable date)
        - **Either** `sales` **or** both `customers` **and** `atv`  
          - If `sales` given but `customers`/`atv` not given, the app will derive `atv = sales / customers` **only if** customers provided.  
          - Best: provide **customers** and **sales**; ATV will be computed as `sales/customers`.
        - Optional: `weather`, `event_flag`, `notes` (ignored safely if present)

        **Granularity:** Daily.
        """
    )

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
    st.markdown("Provide **future** event uplifts (percent). Leave zeros if none.")
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

# Internal state
if "latest_forecast" not in st.session_state:
    st.session_state["latest_forecast"] = None

def validate_and_prepare(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")
    df = ensure_datetime_index(df)
    df = prepare_history(df)  # computes atv if possible, checks integrity, adds derived cols
    if df["customers"].isna().any():
        raise ValueError("Customers column is required or derivable. Provide 'customers' or both 'sales' & 'customers'.")
    if df["atv"].isna().any():
        raise ValueError("ATV could not be derived. Provide either 'sales' & 'customers' or 'atv' explicitly.")
    return df

def run_pipeline(df_hist: pd.DataFrame, H: int, events_future: pd.DataFrame):
    # Build future calendar with covariates + events
    future_cal = add_future_calendar(df_hist, periods=H)
    ev = build_event_frame(events_future)

    # Customers (trend is king): ETS baseline + recent-trend correction + (optional) weekday caps + events
    cust_fc, cust_bands = forecast_customers_with_trend_correction(
        hist=df_hist,
        future_cal=future_cal,
        H=H,
        decay_lambda=decay_lambda,
        apply_weekday_caps=apply_weekday_caps,
        event_uplift_pct=ev["uplift_customers_pct"],
        return_bands=show_bands,
    )

    # ATV (stable): Direct multi-horizon LightGBM + guardrails + events
    atv_fc, atv_bands = forecast_atv_direct(
        hist=df_hist,
        future_cal=future_cal,
        H=H,
        guardrail_mad_mult=atv_guardrail_factor,
        event_uplift_pct=ev["uplift_atv_pct"],
        return_bands=show_bands,
    )

    # Combine into Sales
    combined = combine_sales_and_bands(
        dates=future_cal.index,
        customers=cust_fc,
        customers_bands=cust_bands,
        atv=atv_fc,
        atv_bands=atv_bands,
        return_bands=show_bands,
    )
    return combined

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        hist = validate_and_prepare(df_raw)

        st.write("✅ Data preview (last 30 rows):")
        st.dataframe(hist.tail(30), use_container_width=True)

        if go:
            fc = run_pipeline(hist, H, events_df)
            st.session_state["latest_forecast"] = fc

        if do_backtest:
            with st.spinner("Running backtest..."):
                metrics = backtest_metrics(hist, horizon=H, folds=6)
            st.subheader("📊 Backtest Metrics (Rolling Origin)")
            st.write(metrics)

        if st.session_state["latest_forecast"] is not None:
            st.subheader("🔮 Forecast")
            fc = st.session_state["latest_forecast"].copy()
            show_cols = ["customers_p50", "atv_p50", "sales_p50"]
            if show_bands:
                show_cols = [
                    "customers_p10", "customers_p50", "customers_p90",
                    "atv_p10", "atv_p50", "atv_p90",
                    "sales_p10", "sales_p50", "sales_p90",
                ]
            pretty = fc.copy()
            for c in pretty.columns:
                if c.startswith("atv") or c.startswith("sales"):
                    pretty[c] = pretty[c].apply(pretty_money)
                else:
                    pretty[c] = pretty[c].round(0).astype(int)

            st.dataframe(pretty, use_container_width=True, height=400)

            st.line_chart(fc[["customers_p50"]].rename(columns={"customers_p50":"customers"}))
            st.line_chart(fc[["atv_p50"]].rename(columns={"atv_p50":"ATV (₱)"}))
            st.line_chart(fc[["sales_p50"]].rename(columns={"sales_p50":"Sales (₱)"}))

            if download_forecast:
                buf = io.StringIO()
                out = fc.copy()
                out.to_csv(buf, index=True)
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
