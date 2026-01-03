# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore
from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

# --- Page Configuration ---
st.set_page_config(page_title="AI Sales Forecaster v8.5", layout="wide")

# --- Firestore Initialization & Logging ---
@st.cache_resource
def init_db():
    if not firebase_admin._apps:
        creds = st.secrets.firebase_credentials.to_dict()
        creds['private_key'] = creds['private_key'].replace('\\n', '\n')
        firebase_admin.initialize_app(credentials.Certificate(creds))
    return firestore.client()

def save_forecast_to_log(db_client, forecast_df):
    """Writes the 15-day outlook to Firestore 'forecast_log' collection."""
    if db_client is None or forecast_df.empty:
        return False
    try:
        batch = db_client.batch()
        log_col = db_client.collection('forecast_log')
        # Use localized timestamp for the 'generated_on' field
        gen_time = pd.to_datetime('now') 

        for _, row in forecast_df.iterrows():
            # Format doc ID as the date of the forecast (e.g., 2026-01-15)
            doc_id = row['ds'].strftime('%Y-%m-%d')
            doc_ref = log_col.document(doc_id)
            
            data = {
                'generated_on': gen_time,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }
            batch.set(doc_ref, data, merge=True)
        
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Firestore Save Error: {e}")
        return False

# --- Main App Execution ---
db = init_db()

with st.sidebar:
    st.title("🚀 Smart Forecaster")
    st.info("Direct-Horizon Ensemble Logic")
    
    if st.button("Generate & Save 15-Day Outlook", type="primary", use_container_width=True):
        hist_df = load_from_firestore(db, 'historical_data')
        ev_df = load_from_firestore(db, 'future_activities')
        
        # We need 364 days of data for the 'Same-Day-Last-Year' anchor
        if len(hist_df) > 364:
            with st.spinner("🧠 Training 15 Expert Models & Saving to Cloud..."):
                # 1. Generate Customer Forecast (Direct Strategy)
                c_df, _ = generate_customer_forecast(hist_df, ev_df)
                
                # 2. Generate ATV Forecast (Prophet Seasonality)
                a_df, _ = generate_atv_forecast(hist_df, ev_df)
                
                # 3. Combine 
                final = pd.merge(c_df, a_df, on='ds')
                final['forecast_sales'] = final['forecast_customers'] * final['forecast_atv']
                
                # 4. Save to Firestore (The missing step)
                success = save_forecast_to_log(db, final)
                
                if success:
                    st.session_state.final_df = final
                    st.success("✅ Forecast saved to Firestore!")
                    time.sleep(1)
                    st.rerun()
        else:
            st.error("Insufficient Data: Need 365+ days for SDLY (Same-Day-Last-Year) logic.")

# --- Dashboard Display ---
st.header("🔮 15-Day Sales Projections")
if 'final_df' in st.session_state:
    df = st.session_state.final_df.copy()
    df['Date'] = df['ds'].dt.strftime('%Y-%m-%d')
    df['Sales'] = df['forecast_sales'].map("₱{:,.2f}".format)
    df['Avg Sale'] = df['forecast_atv'].map("₱{:,.2f}".format)
    
    st.dataframe(
        df[['Date', 'Sales', 'forecast_customers', 'Avg Sale']].rename(
            columns={'forecast_customers': 'Predicted Customers'}
        ).set_index('Date'), 
        use_container_width=True
    )
else:
    st.info("Run the forecast from the sidebar to begin and update the database.")
