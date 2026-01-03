# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import from our hybrid model modules ---
from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="AI Forecaster v7.0 (Self-Learning)",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide"
)

# --- Firestore Initialization ---
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

def save_forecast_to_log(db_client, forecast_df):
    if db_client is None or forecast_df.empty: return False
    try:
        batch = db_client.batch()
        log_col = db_client.collection('forecast_log')
        ts = pd.to_datetime('today', utc=True)
        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            batch.set(log_col.document(doc_id), {
                'generated_on': ts,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }, merge=True)
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

# --- Helper Functions ---
@st.cache_data
def get_historical_data(_db): return load_from_firestore(_db, 'historical_data')

@st.cache_data
def get_events_data(_db): return load_from_firestore(_db, 'future_activities')

# --- Main App Logic ---
db = init_firestore()

if db:
    # Initialize session states
    for key in ['cust_df', 'atv_df', 'final_df', 'model']:
        if key not in st.session_state: st.session_state[key] = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png", width=100)
        st.title("AI Forecaster v7.0")
        st.info("Engine: Self-Learning Poisson + Prophet")

        if st.button("🔄 Clear Cache & Reset"):
            st.cache_data.clear(); st.rerun()

        st.markdown("---")
        # Step 1 & 2 are now automated in the Final Step for a "One-Click" experience
        st.subheader("Autonomous Generation")
        
        if st.button("🚀 Generate & Save Final Forecast", type="primary", use_container_width=True):
            with st.spinner("🤖 Applying Self-Learning Bias and Training..."):
                hist_df = get_historical_data(db)
                ev_df = get_events_data(db)
                
                if len(hist_df) < 30:
                    st.error("Insufficient data (Need 30+ days)")
                else:
                    # Execute the unified Self-Learning pipeline
                    c_df, c_model = generate_customer_forecast(hist_df, ev_df, db)
                    a_df, _ = generate_atv_forecast(hist_df, ev_df)
                    
                    f_df = pd.merge(c_df, a_df, on='ds')
                    f_df['forecast_sales'] = f_df['forecast_customers'] * f_df['forecast_atv']
                    
                    st.session_state.cust_df = c_df
                    st.session_state.atv_df = a_df
                    st.session_state.final_df = f_df
                    st.session_state.model = c_model
                    
                    if save_forecast_to_log(db, f_df):
                        st.success("✅ Forecast Generated and Logged!")
                        time.sleep(1)
                        st.rerun()

    # --- Dashboard UI ---
    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Model Insights"])
    
    with tabs[0]:
        if st.session_state.final_df is not None:
            df = st.session_state.final_df.copy()
            st.subheader("Self-Corrected Sales Forecast")
            
            # Formatting for display
            df['Date'] = df['ds'].dt.strftime('%Y-%m-%d')
            df['Sales'] = df['forecast_sales'].map("₱{:,.2f}".format)
            df['Customers'] = df['forecast_customers']
            df['ATV'] = df['forecast_atv'].map("₱{:,.2f}".format)
            
            st.dataframe(df[['Date', 'Sales', 'Customers', 'ATV']].set_index('Date'), use_container_width=True, height=500)
        else:
            st.info("Click the button in the sidebar to run the self-learning forecast engine.")

    with tabs[1]:
        if st.session_state.model:
            st.subheader("Key Drivers of Customer Traffic")
            importance = pd.DataFrame({
                'feature': st.session_state.model.feature_name_,
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            st.bar_chart(importance.set_index('feature'))
