# app.py
import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

st.set_page_config(page_title="AI Sales Forecaster v8.0", layout="wide")

@st.cache_resource
def init_db():
    if not firebase_admin._apps:
        creds = st.secrets.firebase_credentials.to_dict()
        creds['private_key'] = creds['private_key'].replace('\\n', '\n')
        firebase_admin.initialize_app(credentials.Certificate(creds))
    return firestore.client()

db = init_db()

with st.sidebar:
    st.title("🚀 Smart Forecaster")
    st.info("Direct-Horizon Ensemble Logic")
    
    if st.button("Generate 15-Day Outlook", type="primary", use_container_width=True):
        hist_df = load_from_firestore(db, 'historical_data')
        ev_df = load_from_firestore(db, 'future_activities')
        
        if len(hist_df) > 365:
            with st.spinner("🤖 Training 15 Expert Models..."):
                c_df, _ = generate_customer_forecast(hist_df, ev_df)
                a_df, _ = generate_atv_forecast(hist_df, ev_df)
                
                final = pd.merge(c_df, a_df, on='ds')
                final['forecast_sales'] = final['forecast_customers'] * final['forecast_atv']
                st.session_state.final_df = final
                st.success("Forecast Complete!")
        else:
            st.warning("Need at least 1 year of data for SDLY Anchoring.")

st.header("🔮 15-Day Sales Projections")
if 'final_df' in st.session_state:
    df = st.session_state.final_df.copy()
    df['Date'] = df['ds'].dt.strftime('%Y-%m-%d')
    df['Sales'] = df['forecast_sales'].map("₱{:,.2f}".format)
    st.dataframe(df[['Date', 'Sales', 'forecast_customers', 'forecast_atv']].set_index('Date'), use_container_width=True)
else:
    st.info("Run the forecast from the sidebar to begin.")
