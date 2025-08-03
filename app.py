# app.py
import streamlit as st
import pandas as pd
import time
from datetime import date
import firebase_admin
from firebase_admin import credentials, firestore
import os

# --- Import from our new, separated modules ---
from data_processing import load_from_firestore
from forecasting import load_tft_model, generate_forecast, get_interpretation_plot

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="AI Sales Forecaster v4.0 (TFT)",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Same as before) ---
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
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- Model Loading ---
@st.cache_resource
def get_model():
    """Loads the pre-trained TFT model once and caches it."""
    with st.spinner("🧠 Loading AI model..."):
        model = load_tft_model("best_model.ckpt")
    return model

# --- Data Editing and Logging (Mostly unchanged) ---
def save_forecast_to_log(db_client, forecast_df):
    """Saves the generated forecast to the 'forecast_log' collection in Firestore."""
    if db_client is None or forecast_df.empty:
        st.warning("Database client not available or forecast is empty. Skipping log.")
        return False
    try:
        batch = db_client.batch()
        log_collection_ref = db_client.collection('forecast_log')
        generated_on_ts = pd.to_datetime('today').normalize()
        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            log_doc_ref = log_collection_ref.document(doc_id)
            log_data = {
                'generated_on': generated_on_ts, 'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }
            batch.set(log_doc_ref, log_data)
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast to database: {e}")
        return False

def render_historical_record(row, db_client):
    """Renders an editable historical data record."""
    date_str = row['date'].strftime('%B %d, %Y')
    with st.expander(f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}"):
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            updated_day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"], 
                                            index=0 if row.get('day_type') == "Normal Day" else 1,
                                            key=f"day_type_{row['doc_id']}")
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                db_client.collection('historical_data').document(row['doc_id']).update({'day_type': updated_day_type})
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()
model = get_model()

if db:
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'raw_predictions' not in st.session_state:
        st.session_state.raw_predictions = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v4.0")
        st.info("Unified TFT Architecture")

        if not os.path.exists("best_model.ckpt"):
            st.error("FATAL: `best_model.ckpt` not found. Please run `train_model.py` first.")
        else:
            if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
                historical_df = load_from_firestore(db, 'historical_data')
                events_df = load_from_firestore(db, 'future_activities')
                
                if len(historical_df) < 60:
                    st.error("Need at least 60 days of historical data for the model to work.")
                else:
                    with st.spinner("🧠 AI is predicting future outcomes..."):
                        forecast_df, raw_preds = generate_forecast(model, historical_df, events_df, periods=15)
                        st.session_state.forecast_df = forecast_df
                        st.session_state.raw_predictions = raw_preds
                    
                    if not forecast_df.empty:
                        with st.spinner("📡 Logging forecast to database..."):
                            save_successful = save_forecast_to_log(db, forecast_df)
                        if save_successful: st.success("Forecast Generated and Logged!")
                        else: st.warning("Forecast generated but failed to log.")
                    else:
                        st.error("Forecast generation failed.")

    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"])

    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df_to_show = st.session_state.forecast_df.rename(columns={
                'ds': 'Date', 'forecast_customers': 'Predicted Customers',
                'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)'
            })
            st.dataframe(df_to_show.set_index('Date'), use_container_width=True, height=560)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    with tabs[1]:
        st.header("💡 Forecast Insights")
        st.info("This shows the model's internal reasoning. It highlights the most important factors (past sales, events, day of week) that influenced the final prediction.")
        if st.session_state.raw_predictions:
            with st.spinner("Analyzing model's attention..."):
                interpretation_fig = get_interpretation_plot(model, st.session_state.raw_predictions)
                st.pyplot(interpretation_fig)
        else:
            st.info("Generate a forecast to see the breakdown of its components.")
            
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        historical_df = load_from_firestore(db, 'historical_data')
        if not historical_df.empty:
            for _, row in historical_df.sort_values(by="date", ascending=False).head(30).iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore.")
