# app.py
import streamlit as st
import pandas as pd
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

# --- Import from our new, separated modules ---
from data_processing import load_from_firestore # This function stays the same
from forecasting import generate_forecast # This is our new TFT function

# --- Page Configuration and Styling (No changes needed) ---
st.set_page_config(
    page_title="Sales Forecaster v5.0 TFT",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide"
)

# --- Custom CSS (No changes needed) ---
def apply_custom_styling():
    # Your existing CSS is great, no changes needed.
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style> ... </style> 
    """, unsafe_allow_html=True) # Keep your existing CSS here

# --- Firestore Initialization & Data Saving (No changes needed) ---
@st.cache_resource
def init_firestore():
    # Your existing function is perfect.
    try:
        # ...
        return firestore.client()
    except Exception as e:
        # ...
        return None

def save_forecast_to_log(db_client, forecast_df):
    # Your existing function is fine.
    pass

def render_historical_record(row, db_client):
    # Your existing function is perfect.
    pass

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'model' not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        # ... (Your sidebar widgets can stay the same) ...
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 90: # TFT needs a bit more history
                st.error("Need at least 90 days of data for a reliable TFT forecast.")
            else:
                with st.spinner("🧠 Training Temporal Fusion Transformer... (This may take a few minutes)"):
                    # THIS IS THE NEW FUNCTION CALL
                    forecast_df, model = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.model = model
                
                if not forecast_df.empty:
                    st.success("Forecast Generated Successfully!")
                else:
                    st.error("Forecast generation failed.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab (UPGRADED) ---
    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df = st.session_state.forecast_df
            
            # Create the interactive uncertainty plot
            fig = go.Figure()
            # Upper bound (90% quantile)
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['customers_p90'], mode='lines',
                line=dict(width=0), showlegend=False
            ))
            # Lower bound (10% quantile), filled to upper bound
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['customers_p10'], mode='lines',
                line=dict(width=0), fillcolor='rgba(200, 16, 46, 0.2)',
                fill='tonexty', name='80% Confidence Interval'
            ))
            # Median forecast (p50)
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['predicted_customers'], mode='lines+markers',
                line=dict(color='#c8102e', width=3), name='Median Forecast (p50)'
            ))
            
            fig.update_layout(
                title="Customer Forecast with Uncertainty Range",
                yaxis_title="Predicted Customers",
                xaxis_title="Date",
                font=dict(family="Poppins, sans-serif", color="white"),
                plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a',
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display the data table as well
            st.dataframe(df[['ds', 'predicted_customers', 'predicted_atv', 'predicted_sales']].rename(columns={
                'ds': 'Date', 'predicted_customers': 'Median Predicted Customers',
                'predicted_atv': 'Predicted Avg Sale (₱)', 'predicted_sales': 'Median Predicted Sales (₱)'
            }).set_index('Date'), use_container_width=True)

        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    # --- Forecast Insights Tab (UPGRADED) ---
    with tabs[1]:
        st.header("💡 Key Forecast Drivers (TFT Interpretation)")
        st.info("This shows the most important factors the TFT model used. Higher values mean more impact.")
        
        if st.session_state.model:
            model = st.session_state.model
            # Use the model's built-in interpretation plot
            interpretation = model.interpret_output(model.predict(model.val_dataloader(), mode="raw", return_x=True)[0], reduction="sum")
            
            # The interpretation object is a dictionary of plots
            st.pyplot(interpretation['attention'][0])
            st.pyplot(interpretation['static_variables'][0])
            st.pyplot(interpretation['encoder_variables'][0])
            st.pyplot(interpretation['decoder_variables'][0])
        else:
            st.info("Generate a forecast to see the key drivers.")

    # --- Edit Data Tab (No changes needed) ---
    with tabs[2]:
        # Your existing code for this tab is excellent and doesn't need changes.
        st.header("✍️ Edit Historical Data")
        # ... Keep your existing implementation ...
