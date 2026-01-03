# app.py (Modified Sidebar Section)
with st.sidebar:
    # ... [Keep Steps 1 and 2] ...
    
    st.subheader("Step 3: AI Optimizer")
    if st.button("🚀 Run Optimized Forecast", type="primary", use_container_width=True):
        historical_df = get_historical_data(db)
        events_df = get_events_data(db)
        
        if len(historical_df) < 30:
            st.error("Insufficient data.")
        else:
            with st.spinner("Self-correcting and learning from past mistakes..."):
                # Call the optimizer orchestrator
                final_df, metrics, model = run_optimized_forecast(db, historical_df, events_df)
                
                st.session_state.final_forecast_df = final_df
                st.session_state.customer_model = model
                st.session_state.optimizer_metrics = metrics
                
                # Save to database
                save_forecast_to_log(db, final_df)
                st.success(f"Optimized! System Accuracy: {metrics['accuracy']}%")

# [Optional: Add a Metric Display to the Dashboard tab]
if 'optimizer_metrics' in st.session_state:
    m = st.session_state.optimizer_metrics
    cols = st.columns(3)
    cols[0].metric("Model Accuracy", f"{m['accuracy']}%")
    cols[1].metric("Bias Correction", f"{m['bias_adjustment']}x")
    cols[2].metric("Status", m['status'])
