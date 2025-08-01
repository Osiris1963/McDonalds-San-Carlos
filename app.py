import streamlit as st
import pandas as pd
import time
from datetime import date, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objs as go

# --- Import from our new, separated modules ---
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v2.2",
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
    """Initializes a connection to Firestore using Streamlit Secrets."""
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
              "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

# --- RE-ENGINEERED FOR ORDERED LOGGING ---
def save_forecast_to_log(db_client, forecast_df):
    """
    Saves the generated forecast to the 'forecast_log' collection in Firestore.
    This version uses the forecast date (YYYY-MM-DD) as the document ID to ensure
    the collection is always stored in chronological order.
    """
    if db_client is None or forecast_df.empty:
        st.warning("Database client not available or forecast is empty. Skipping log.")
        return False

    try:
        batch = db_client.batch()
        log_collection_ref = db_client.collection('forecast_log')
        generated_on_ts = pd.to_datetime('today').normalize()

        for _, row in forecast_df.iterrows():
            # Create a deterministic, sortable document ID from the forecast date.
            doc_id = row['ds'].strftime('%Y-%m-%d')
            log_doc_ref = log_collection_ref.document(doc_id)
            
            log_data = {
                'generated_on': generated_on_ts,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }
            # Use set() with a specific doc_ref to create or overwrite.
            batch.set(log_doc_ref, log_data)
        
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast to database: {e}")
        return False

# --- UI Component Rendering Functions ---
def render_historical_record(row, db_client):
    """Renders an editable historical data record."""
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        return

    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        if day_type == 'Not Normal Day':
            st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")

        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            day_type_options = ["Normal Day", "Not Normal Day"]
            current_day_type = row.get('day_type', 'Normal Day')
            if current_day_type not in day_type_options:
                current_day_type = day_type_options[0] 
            current_index = day_type_options.index(current_day_type)
            
            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
            
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                update_data = {'day_type': updated_day_type}
                db_client.collection('historical_data').document(row['doc_id']).update(update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'prophet_model' not in st.session_state:
        st.session_state.prophet_model = None

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v2.2")
        st.info("Re-architected for precision and ordered logging.")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data refreshed. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 50:
                st.error("Need at least 50 days of data for a reliable forecast.")
            else:
                forecast_df = pd.DataFrame() # Ensure variable exists
                with st.spinner("🧠 Running advanced ensemble forecast..."):
                    forecast_df, prophet_model = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.prophet_model = prophet_model
                
                if not forecast_df.empty:
                    with st.spinner("📡 Logging forecast to database for performance tracking..."):
                        save_successful = save_forecast_to_log(db, forecast_df)
                    
                    if save_successful:
                        st.success("Forecast Generated and Logged Successfully!")
                    else:
                        st.warning("Forecast was generated but failed to log to the database.")
                else:
                    st.error("Forecast generation failed. Could not create or log forecast.")


    # --- Main Content Area with Tabs ---
    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab ---
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

    # --- Forecast Insights Tab ---
    with tabs[1]:
        st.header("💡 Forecast Insights")
        if st.session_state.prophet_model:
            future = st.session_state.prophet_model.make_future_dataframe(periods=15)
            forecast_components = st.session_state.prophet_model.predict(future)
            st.info("This chart shows the foundational drivers from the Prophet model for customer forecasts.")
            fig = st.session_state.prophet_model.plot_components(forecast_components)
            st.pyplot(fig)
        else:
            st.info("Generate a forecast to see the breakdown of its components.")

    # --- Edit Data Tab ---
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can correct the 'Day Type' for past dates if an unusual event occurred.")
        historical_df = load_from_firestore(db, 'historical_data')
        if not historical_df.empty:
            recent_df = historical_df.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
