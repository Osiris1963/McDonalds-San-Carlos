# app.py
import streamlit as st
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

from data_processing import load_from_firestore
from forecasting import generate_forecast

# Page Config and CSS can remain exactly as you had them.
st.set_page_config(page_title="Sales Forecaster v5.1 Stable", layout="wide")
# ... your custom CSS function ...

# Firestore init and other helper functions can also remain the same.
@st.cache_resource
def init_firestore():
    # ... your existing function ...
    pass
# ... other helpers ...

# --- Main Application ---
# apply_custom_styling() # Call your CSS function
db = init_firestore()

if db:
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'model' not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        # ... Your sidebar widgets ...
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 90:
                st.error("Need at least 90 days of data for a reliable TFT forecast.")
            else:
                with st.spinner("🧠 Training Robust Forecasting Engine... (This may take a few minutes)"):
                    forecast_df, model = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.model = model
                
                if not forecast_df.empty:
                    st.success("Forecast Generated Successfully!")
                else:
                    st.error("Forecast generation failed.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab (SIMPLIFIED) ---
    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df = st.session_state.forecast_df
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['predicted_customers'], mode='lines+markers',
                line=dict(color='#c8102e', width=3), name='Predicted Customers'
            ))
            
            fig.update_layout(
                title="Customer Forecast", yaxis_title="Predicted Customers",
                font=dict(family="Poppins, sans-serif", color="white"),
                plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a'
            )
            st.plotly_chart(fig, use_container_width=True)

            display_df = df[['ds', 'predicted_customers', 'predicted_atv', 'predicted_sales']].rename(columns={
                'ds': 'Date', 'predicted_customers': 'Predicted Customers',
                'predicted_atv': 'Predicted Avg Sale (₱)', 'predicted_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    # --- Forecast Insights Tab (SIMPLIFIED) ---
    with tabs[1]:
        st.header("💡 Key Forecast Drivers")
        st.info("This shows the general importance of different features to the model's predictions.")
        
        if st.session_state.model:
            model = st.session_state.model
            # Use the model's built-in variable importance calculation
            try:
                # This requires a validation dataloader, which we created in forecasting.py
                val_dataloader = model.val_dataloader()
                importance = model.evaluate(val_dataloader, verbose=False)
                fig = model.plot_variable_importances(importance)
                fig.update_layout(
                    font=dict(color="white"),
                    plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a'
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate feature importance plot. Details: {e}")
        else:
            st.info("Generate a forecast to see the key drivers.")
    
    # --- Edit Data Tab can remain the same ---
    with tabs[2]:
        # ... your existing code ...
        pass
else:
    st.error("Could not connect to Firestore.")
