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

# --- Import from our hybrid model modules ---
from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v7.0 (Self-Learning)",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; }
        .block-container { padding: 2.5rem 2rem !important; }
        [data-testid="stSidebar"] { background-color: #252525; border-right: 1px solid #444; }
        .stButton > button {
            border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out;
            border: none; padding: 10px 16px;
        }
        .stButton:has(button:contains("Generate")), .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button, .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px); box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px; background-color: transparent; color: #d3d3d3;
            padding: 8px 14px; font-weight: 600; font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: #ffffff; }
        .st-expander {
            border: 1px solid #444 !important; box-shadow: none; border-radius: 10px;
            background-color: #252525; margin-bottom: 0.5rem;
        }
        .st-expander header { font-size: 0.9rem; font-weight: 600; color: #d3d3d3; }
        .stPlotlyChart { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization & Data Functions ---
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
        log_collection_ref = db_client.collection('forecast_log')
        generated_on_ts = pd.to_datetime('today', utc=True)

        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            log_doc_ref = log_collection_ref.document(doc_id)
            log_data = {
                'generated_on': generated_on_ts,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }
            batch.set(log_doc_ref, log_data, merge=True)
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast: {e}")
        return False

@st.cache_data
def get_historical_data(_db_conn):
    return load_from_firestore(_db_conn, 'historical_data')

@st.cache_data
def get_events_data(_db_conn):
    return load_from_firestore(_db_conn, 'future_activities')

def get_forecast_logs(db_client):
    """SENIOR DEV: Pulls past logs to allow the model to relearn from errors."""
    if db_client is None: return pd.DataFrame()
    try:
        docs = db_client.collection('forecast_log').stream()
        logs = [doc.to_dict() for doc in docs]
        if not logs: return pd.DataFrame()
        df = pd.DataFrame(logs)
        df['forecast_for_date'] = pd.to_datetime(df['forecast_for_date']).dt.normalize()
        return df
    except:
        return pd.DataFrame()

def render_historical_record(row, db_client):
    if 'doc_id' not in row or pd.isna(row['doc_id']): return
    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        day_type_options = ["Normal Day", "Not Normal Day"]
        current_day_type = row.get('day_type', day_type_options[0])
        try:
            current_index = day_type_options.index(current_day_type)
        except ValueError:
            current_index = 0
        
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
            if st.form_submit_button("💾 Update Record", use_container_width=True):
                db_client.collection('historical_data').document(row['doc_id']).update({'day_type': updated_day_type})
                st.success(f"Updated {date_str}!")
                st.cache_data.clear()
                time.sleep(0.5); st.rerun()

# --- Main Application Logic ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state for self-learning metrics
    for key in ['customer_forecast_df', 'atv_forecast_df', 'final_forecast_df', 'customer_model', 'accuracy_bias']:
        if key not in st.session_state: st.session_state[key] = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v7.0")
        st.info("System Status: Online & Learning")

        if st.button("🔄 Full System Reset"):
            st.cache_data.clear(); st.cache_resource.clear()
            for key in ['customer_forecast_df', 'atv_forecast_df', 'final_forecast_df', 'customer_model', 'accuracy_bias']:
                st.session_state[key] = None
            st.rerun()

        st.markdown("---")
        # Step 1: Customer Forecast with Relearning
        if st.button("📊 Step 1: Forecast Customers", use_container_width=True):
            hist_df = get_historical_data(db)
            event_df = get_events_data(db)
            log_df = get_forecast_logs(db) # Pull logs to check past effectiveness
            
            if len(hist_df) < 30:
                st.error("Insufficient data (Need 30+ days)")
            else:
                with st.spinner("🧠 Calculating effectiveness & Relearning..."):
                    # The forecasting module now uses the log_df to adjust predictions
                    cust_df, model, bias = generate_customer_forecast(hist_df, event_df, forecast_log_df=log_df)
                    st.session_state.customer_forecast_df = cust_df
                    st.session_state.customer_model = model
                    st.session_state.accuracy_bias = bias
                st.success("Customer Model Optimized!")

        # Step 2: ATV Forecast
        if st.button("📈 Step 2: Forecast ATV", use_container_width=True):
            hist_df = get_historical_data(db)
            event_df = get_events_data(db)
            with st.spinner("⏳ Training ATV Model..."):
                atv_df, _ = generate_atv_forecast(hist_df, event_df)
                st.session_state.atv_forecast_df = atv_df
            st.success("ATV Model Ready!")

        st.markdown("---")
        # Step 3: Combined Output
        ready = st.session_state.customer_forecast_df is not None and st.session_state.atv_forecast_df is not None
        if st.button("🚀 Generate Final Forecast", type="primary", use_container_width=True, disabled=not ready):
            with st.spinner("Finalizing calculations..."):
                final_df = pd.merge(st.session_state.customer_forecast_df, st.session_state.atv_forecast_df, on='ds')
                final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
                st.session_state.final_forecast_df = final_df
                save_forecast_to_log(db, final_df)
            st.success("Forecast Logged & Saved!")

    # --- Tabs Layout ---
    tabs = st.tabs(["🔮 Dashboard", "💡 Intelligence", "✍️ Data Management"])

    with tabs[0]:
        st.header("🔮 Sales Forecast Dashboard")
        if st.session_state.accuracy_bias:
            bias = st.session_state.accuracy_bias
            if bias > 1.02:
                st.warning(f"🤖 **Self-Correction Active:** Model detected a recent under-prediction. Nudging results UP by {((bias-1)*100):.1f}% to match actual trends.")
            elif bias < 0.98:
                st.info(f"🤖 **Self-Correction Active:** Model detected a recent over-prediction. Adjusting results DOWN by {((1-bias)*100):.1f}% for better accuracy.")

        if st.session_state.final_forecast_df is not None:
            df = st.session_state.final_forecast_df.copy()
            df.columns = ['Date', 'Customers', 'ATV (₱)', 'Total Sales (₱)']
            st.dataframe(df.set_index('Date').style.format({"ATV (₱)": "₱{:,.2f}", "Total Sales (₱)": "₱{:,.2f}"}), use_container_width=True, height=500)
        else:
            st.info("Please complete the sidebar steps to view the combined forecast.")

    with tabs[1]:
        st.header("💡 Model Intelligence")
        if st.session_state.customer_model:
            model = st.session_state.customer_model
            importance = pd.DataFrame({'Feature': model.feature_name_, 'Value': model.feature_importances_}).sort_values('Value', ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Value', y='Feature', data=importance, palette='magma', ax=ax)
            plt.title("Top Drivers for Customer Traffic")
            st.pyplot(fig)
        else:
            st.info("Run Step 1 to see model drivers.")

    with tabs[2]:
        st.header("✍️ Manage Historical Records")
        hist_edit = get_historical_data(db)
        if not hist_edit.empty:
            for _, row in hist_edit.sort_values('date', ascending=False).head(15).iterrows():
                render_historical_record(row, db)
else:
    st.error("Connection Failed. Verify Firestore secrets.")
