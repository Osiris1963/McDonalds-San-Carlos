# app.py
import streamlit as st
import pandas as pd
import time
import os
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import load_from_firestore
from forecasting import generate_direct_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v6.0 (Robust)",
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
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using structured Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            creds_dict = {
              "type": st.secrets.firebase_credentials.type,
              "project_id": st.secrets.firebase_credentials.project_id,
              "private_key_id": st.secrets.firebase_credentials.private_key_id,
              "private_key": st.secrets.firebase_credentials.private_key.replace('\\n', '\n'),
              "client_email": st.secrets.firebase_credentials.client_email,
              "client_id": st.secrets.firebase_credentials.client_id,
              "auth_uri": st.secrets.firebase_credentials.auth_uri,
              "token_uri": st.secrets.firebase_credentials.token_uri,
              "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url,
              "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url,
              "universe_domain": st.secrets.firebase_credentials.universe_domain
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

# --- Data Caching ---
@st.cache_data(ttl=1800)
def get_data(_db_client):
    historical_df = load_from_firestore(_db_client, 'historical_data')
    events_df = load_from_firestore(_db_client, 'future_activities')
    return historical_df, events_df

def save_forecast_to_log(db_client, forecast_df):
    """Saves the generated forecast to the 'forecast_log' collection in Firestore."""
    if db_client is None or forecast_df.empty:
        st.warning("Database client not available or forecast is empty. Skipping log.")
        return False
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
        st.error(f"Error logging forecast to database: {e}")
        return False

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v6.0")
        st.info("Direct Multi-Model Engine - Robust Build")
        
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df, events_df = get_data(db)

            if len(historical_df) < 60:
                st.error("Need at least 60 days of historical data for a reliable forecast.")
            else:
                with st.spinner("🧠 Training Robust Multi-Model Engine..."):
                    forecast_df = generate_direct_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                
                if not forecast_df.empty:
                    with st.spinner("📡 Logging forecast to database..."):
                        save_successful = save_forecast_to_log(db, forecast_df)
                    if save_successful:
                        st.success("Forecast Generated and Logged!")
                    else:
                        st.warning("Forecast generated but failed to log.")
                else:
                    st.error("Forecast generation failed.")

    tab_list = ["🔮 Forecast Dashboard", "📊 Forecast Charts", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab ---
    with tabs[0]:
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
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    # --- Forecast Charts Tab ---
    with tabs[1]:
        st.header("📊 Forecast Charts")
        st.info("Visualizing the predicted trends for the next 15 days.")
        
        if not st.session_state.forecast_df.empty:
            df_chart = st.session_state.forecast_df.copy()
            df_chart['ds'] = pd.to_datetime(df_chart['ds'])

            plt.style.use('dark_background')
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            
            ax1.plot(df_chart['ds'], df_chart['forecast_sales'], color='#c8102e', marker='o', linestyle='-')
            ax1.set_title('Predicted Sales (₱)', fontsize=16, color='white')
            ax2.plot(df_chart['ds'], df_chart['forecast_customers'], color='#ffc72c', marker='o', linestyle='-')
            ax2.set_title('Predicted Customers', fontsize=16, color='white')
            ax3.plot(df_chart['ds'], df_chart['forecast_atv'], color='#4caf50', marker='o', linestyle='-')
            ax3.set_title('Predicted Average Transaction Value (ATV)', fontsize=16, color='white')

            for ax in [ax1, ax2, ax3]:
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(axis='y', colors='white')
            ax3.tick_params(axis='x', colors='white', rotation=45)

            fig.tight_layout(pad=3.0)
            st.pyplot(fig)
        else:
            st.info("Generate a forecast to see the trend charts.")

    # --- Edit Data Tab ---
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can correct the 'Day Type' for past dates.")
        
        historical_df_edit, _ = get_data(db)
        
        if not historical_df_edit.empty:
            recent_df = historical_df_edit.sort_values(by="date", ascending=False).head(30)
            st.warning("Note: Editing data will not reflect in forecasts until the next time you click 'Generate Forecast'.")
            for _, row in recent_df.iterrows():
                if 'doc_id' in row and row['doc_id'] and not pd.isna(row['doc_id']):
                    date_str = row['date'].strftime('%B %d, %Y')
                    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
                    with st.expander(expander_title):
                        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
                            day_type_options = ["Normal Day", "Not Normal Day"]
                            current_day_type = row.get('day_type', 'Normal Day')
                            current_index = day_type_options.index(current_day_type) if current_day_type in day_type_options else 0
                            
                            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
                            
                            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                                update_data = {'day_type': updated_day_type}
                                db.collection('historical_data').document(row['doc_id']).update(update_data)
                                st.success(f"Record for {date_str} updated!")
                                st.cache_data.clear()
                                time.sleep(1); st.rerun()
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
