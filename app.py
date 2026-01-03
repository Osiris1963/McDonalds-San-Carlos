# Inside app.py (Step 3 Button Logic)

if st.button("Generate Final Forecast & Save", type="primary", use_container_width=True, disabled=is_disabled):
    with st.spinner("🤖 Applying Self-Learning Bias and Combining Models..."):
        historical_df = get_historical_data(db)
        events_df = get_events_data(db)
        
        # 1. Forecast Customers (Now with Bias Correction)
        cust_df, cust_model = generate_customer_forecast(historical_df, events_df, db)
        
        # 2. Forecast ATV
        atv_df, _ = generate_atv_forecast(historical_df, events_df)
        
        # 3. Merge
        final_df = pd.merge(cust_df, atv_df, on='ds')
        final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
        
        # Store in session
        st.session_state.customer_forecast_df = cust_df
        st.session_state.atv_forecast_df = atv_df
        st.session_state.final_forecast_df = final_df
        st.session_state.customer_model = cust_model

        if save_forecast_to_log(db, final_df):
            st.success("✅ Forecast Successful! Bias correction applied.")
        st.rerun()
