# app.py
import streamlit as st
import pandas as pd
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import load_from_firestore
# Import the new, separated functions
from forecasting import train_customer_model_lgbm, train_atv_model_sarima, generate_forecast_from_models

st.set_page_config(
    page_title="Sales Forecaster v5.0",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide"
)

# --- Custom CSS (omitted for brevity, assume it's the same) ---
st.markdown("""<style>... a very long style string ...</style>""", unsafe_allow_html=True) # Placeholder

@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials.to_dict()
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- Main Application ---
db = init_firestore()

if db:
    # Initialize session state for models and forecast
    if 'customer_model' not in st.session_state:
        st.session_state.customer_model = None
    if 'atv_model' not in st.session_state:
        st.session_state.atv_model = None
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png", width=100)
        st.title("AI Forecaster v5.0")
        st.info("Modular Engine: LGBM + SARIMA")

        st.markdown("---")
        st.header("1. Train AI Models")

        # Button to train the customer model
        if st.button("🧠 Train Customer Model (LGBM)"):
            with st.spinner("Loading data and training customer model..."):
                historical_df = load_from_firestore(db, 'historical_data')
                events_df = load_from_firestore(db, 'future_activities')
                if len(historical_df) > 30:
                    st.session_state.customer_model = train_customer_model_lgbm(historical_df, events_df)
                    st.success("Customer Model Trained!")
                else:
                    st.error("Need at least 30 days of data.")

        # Button to train the ATV model
        if st.button("📈 Train ATV Model (SARIMA)"):
            with st.spinner("Loading data and training ATV model..."):
                historical_df = load_from_firestore(db, 'historical_data')
                if len(historical_df) > 30:
                    st.session_state.atv_model = train_atv_model_sarima(historical_df)
                    st.success("ATV Model Trained!")
                else:
                    st.error("Need at least 30 days of data.")
        
        st.markdown("---")
        st.header("2. Generate Forecast")

        # Display status of trained models
        st.write("Customer Model:", "✅ Trained" if st.session_state.customer_model else "❌ Not Trained")
        st.write("ATV Model:", "✅ Trained" if st.session_state.atv_model else "❌ Not Trained")

        # Disable forecast button until both models are trained
        is_disabled = not (st.session_state.customer_model and st.session_state.atv_model)
        if st.button("🔮 Generate Forecast", type="primary", use_container_width=True, disabled=is_disabled):
            with st.spinner("Generating forecast from trained models..."):
                historical_df = load_from_firestore(db, 'historical_data')
                events_df = load_from_firestore(db, 'future_activities')
                st.session_state.forecast_df = generate_forecast_from_models(
                    st.session_state.customer_model,
                    st.session_state.atv_model,
                    historical_df,
                    events_df,
                    periods=15
                )
            st.success("Forecast Generated!")

    # --- Main Dashboard Area ---
    st.header("🔮 Forecast Dashboard")
    if not st.session_state.forecast_df.empty:
        df_to_show = st.session_state.forecast_df.rename(columns={
            'ds': 'Date', 'forecast_customers': 'Predicted Customers',
            'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)'
        }).set_index('Date')
        
        df_to_show['Predicted Sales (₱)'] = df_to_show['Predicted Sales (₱)'].apply(lambda x: f"₱{x:,.2f}")
        df_to_show['Predicted Avg Sale (₱)'] = df_to_show['Predicted Avg Sale (₱)'].apply(lambda x: f"₱{x:,.2f}")
        
        st.dataframe(df_to_show, use_container_width=True, height=560)
    else:
        st.info("Train both models, then click 'Generate Forecast' in the sidebar to begin.")

else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
