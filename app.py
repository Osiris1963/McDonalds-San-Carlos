# app.py
import streamlit as st
import pandas as pd
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

st.set_page_config(page_title="Forecaster v7.0 (Self-Learning)", layout="wide")

# --- Styling & Database ---
def apply_custom_styling():
    st.markdown("""
    <style>
        .main > div { background-color: #1a1a1a; color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #252525; }
        .stButton>button { background: linear-gradient(45deg, #c8102e, #e01a37); color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_firestore():
    if not firebase_admin._apps:
        try:
            creds_dict = st.secrets.firebase_credentials.to_dict()
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firestore Init Error: {e}")
            return None
    return firestore.client()

def get_forecast_logs(db_client):
    if db_client is None: return pd.DataFrame()
    docs = db_client.collection('forecast_log').stream()
    logs = [doc.to_dict() for doc in docs]
    return pd.DataFrame(logs) if logs else pd.DataFrame()

def save_forecast_to_log(db_client, forecast_df):
    if db_client is None or forecast_df.empty: return
    batch = db_client.batch()
    for _, row in forecast_df.iterrows():
        doc_id = row['ds'].strftime('%Y-%m-%d')
        ref = db_client.collection('forecast_log').document(doc_id)
        batch.set(ref, {
            'forecast_for_date': row['ds'],
            'predicted_sales': float(row['forecast_sales']),
            'predicted_customers': int(row['forecast_customers']),
            'predicted_atv': float(row['forecast_atv'])
        }, merge=True)
    batch.commit()

# --- Application ---
apply_custom_styling()
db = init_firestore()

if db:
    for key in ['customer_forecast_df', 'atv_forecast_df', 'final_forecast_df', 'customer_model', 'accuracy_bias']:
        if key not in st.session_state: st.session_state[key] = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png", width=100)
        st.title("AI Forecaster v7.0")
        
        if st.button("📊 Step 1: Forecast Customers", use_container_width=True):
            hist_df = load_from_firestore(db, 'historical_data')
            log_df = get_forecast_logs(db)
            event_df = load_from_firestore(db, 'future_activities')
            
            with st.spinner("🧠 Analyzing effectiveness & Relearning..."):
                cust_df, model, bias = generate_customer_forecast(hist_df, event_df, forecast_log_df=log_df)
                st.session_state.customer_forecast_df = cust_df
                st.session_state.customer_model = model
                st.session_state.accuracy_bias = bias
            st.success("Customer Intelligence Updated!")

        if st.button("📈 Step 2: Forecast ATV", use_container_width=True):
            hist_df = load_from_firestore(db, 'historical_data')
            event_df = load_from_firestore(db, 'future_activities')
            atv_df, _ = generate_atv_forecast(hist_df, event_df)
            st.session_state.atv_forecast_df = atv_df
            st.success("ATV Intelligence Updated!")

        ready = st.session_state.customer_forecast_df is not None and st.session_state.atv_forecast_df is not None
        if st.button("🚀 Finalize & Save", type="primary", use_container_width=True, disabled=not ready):
            final_df = pd.merge(st.session_state.customer_forecast_df, st.session_state.atv_forecast_df, on='ds')
            final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
            st.session_state.final_forecast_df = final_df
            save_forecast_to_log(db, final_df)
            st.balloons()

    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Model Insights"])
    
    with tabs[0]:
        if st.session_state.accuracy_bias:
            b = st.session_state.accuracy_bias
            if b > 1:
                st.warning(f"🤖 **Self-Correction:** Increasing forecast by {((b-1)*100):.1f}% based on recent actuals.")
            else:
                st.info(f"🤖 **Self-Correction:** Lowering forecast by {((1-b)*100):.1f}% based on recent actuals.")
        
        if st.session_state.final_forecast_df is not None:
            df_disp = st.session_state.final_forecast_df.copy()
            df_disp.columns = ['Date', 'Customers', 'ATV (₱)', 'Sales (₱)']
            st.dataframe(df_disp.set_index('Date').style.format({"ATV (₱)": "{:,.2f}", "Sales (₱)": "{:,.2f}"}), use_container_width=True)

    with tabs[1]:
        if st.session_state.customer_model:
            importance = pd.DataFrame({
                'Feature': st.session_state.customer_model.feature_name_, 
                'Importance': st.session_state.customer_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            st.subheader("Top Drivers of Customer Traffic")
            st.bar_chart(importance.set_index('Feature'))
