# app.py
import streamlit as st
import pandas as pd
from forecasting import generate_customer_forecast, generate_atv_forecast
from data_processing import load_from_firestore
# ... (Keep your imports and init_firestore logic from previous file)

db = init_firestore()

if db:
    # State management
    for k in ['cust_df', 'atv_df', 'bias']:
        if k not in st.session_state: st.session_state[k] = None

    with st.sidebar:
        st.header("Forecaster v7.5")
        if st.button("📊 Run Intelligent Forecast"):
            # Load all data
            hist = load_from_firestore(db, 'historical_data')
            logs = load_from_firestore(db, 'forecast_log') # Load past predictions
            events = load_from_firestore(db, 'future_activities')

            with st.spinner("Brain is thinking..."):
                # Step 1: Customer Forecast (Calls optimizer internally)
                c_df, model, bias = generate_customer_forecast(hist, events, forecast_log_df=logs)
                st.session_state.cust_df = c_df
                st.session_state.bias = bias
                
                # Step 2: ATV
                a_df, _ = generate_atv_forecast(hist, events)
                st.session_state.atv_df = a_df
            st.success("Forecast Complete")

    # Display results
    if st.session_state.cust_df is not None:
        b = st.session_state.bias
        st.subheader(f"System Effectiveness Adjustment: {b:.2%}")
        if b > 1.0:
            st.warning(f"Note: System is automatically boosting predictions because recent actuals were higher than predicted.")
        
        # Merge and show results
        final = pd.merge(st.session_state.cust_df, st.session_state.atv_df, on='ds')
        final['Total Sales'] = final['forecast_customers'] * final['forecast_atv']
        st.dataframe(final)
