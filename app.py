# app.py

import streamlit as st
import pandas as pd
import time
from datetime import date, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import torch

# --- Import from our new, separated modules ---
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling (Unchanged) ---
st.set_page_config(
    page_title="Sales Forecaster v4.0 (TFT)",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Firestore Initialization (Unchanged) ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials.to_dict()
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

# --- Functions for DB interaction (Largely Unchanged) ---
def save_forecast_to_log(db_client, forecast_df):
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
                'generated_on': generated_on_ts,
                'forecast_for_date': row['ds'],
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
    if 'doc_id' not in row or pd.isna(row['doc_id']): return
    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    with st.expander(expander_title):
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            day_type_options = ["Normal Day", "Not Normal Day"]
            current_day_type = row.get('day_type', 'Normal Day')
            current_index = day_type_options.index(current_day_type) if current_day_type in day_type_options else 0
            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                db_client.collection('historical_data').document(row['doc_id']).update({'day_type': updated_day_type})
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear(); time.sleep(1); st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state for new model
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'tft_model' not in st.session_state:
        st.session_state.tft_model = None
    if 'tft_dataset' not in st.session_state:
        st.session_state.tft_dataset = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v4.0")
        st.info("Architecture: Temporal Fusion Transformer")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.success("Data refreshed."); time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 75: # TFT needs more data than old models
                st.error("Need at least 75 days of historical data for the deep learning model.")
            else:
                forecast_df = pd.DataFrame()
                with st.spinner("🧠 Training Deep Learning Model (TFT)... This may take a few minutes."):
                    try:
                        forecast_df, tft_model, tft_dataset = generate_forecast(historical_df, events_df, periods=15)
                        st.session_state.forecast_df = forecast_df
                        st.session_state.tft_model = tft_model
                        st.session_state.tft_dataset = tft_dataset
                        st.success("Model trained successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")

                if not st.session_state.forecast_df.empty:
                    with st.spinner("📡 Logging forecast to database..."):
                        save_successful = save_forecast_to_log(db, st.session_state.forecast_df)
                    if save_successful:
                        st.success("Forecast Generated and Logged!")
                    else:
                        st.warning("Forecast generated but failed to log.")
                else:
                    st.error("Forecast generation failed. Check data for unusual patterns.")

    # --- Main Panel Tabs ---
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
        st.header("💡 Forecast Insights (from Temporal Fusion Transformer)")
        if st.session_state.tft_model and st.session_state.tft_dataset:
            st.info("These plots show the model's internal logic. They help us understand *why* it's making certain predictions.")
            model = st.session_state.tft_model
            dataset = st.session_state.tft_dataset
            
            # Create a dataloader for interpretation
            interp_dataloader = dataset.to_dataloader(train=False, batch_size=1)
            raw_predictions, x = model.predict(interp_dataloader, mode="raw", return_x=True)

            # Plot Interpretation
            st.subheader("Model Interpretation: Feature Importance")
            try:
                interpretation = model.interpret_output(raw_predictions, reduction="sum")
                fig = model.plot_interpretation(interpretation)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate interpretation plot. Error: {e}")

        else:
            st.info("Generate a forecast to see model insights.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Correct the 'Day Type' for past dates to improve future forecasts.")
        historical_df = load_from_firestore(db, 'historical_data')
        if not historical_df.empty:
            recent_df = historical_df.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Fatal: Could not connect to Firestore. Check configuration and network.")
