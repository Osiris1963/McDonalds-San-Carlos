# app.py
# The Streamlit user interface for the AI Forecaster.

import streamlit as st
import pandas as pd
import time
from datetime import date, timedelta
import firebase_admin
from firebase_admin import credentials, firestore

# Import from our new, separated modules
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v4.0",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (No changes needed here) ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; color: #FFFFFF; }
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
        .stDataFrame { border: none; }
        .stDataFrame > div { background-color: #252525; }
        .stDataFrame thead th { background-color: #333; color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization (No changes needed) ---
@st.cache_resource
def init_firestore():
    try:
        # Check if already initialized to prevent Streamlit's rerun error
        if not firebase_admin._apps:
            # Use Streamlit secrets for credentials
            creds_dict = st.secrets["firebase_credentials"]
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- Main Application Logic ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state variables
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'prophet_model' not in st.session_state:
        st.session_state.prophet_model = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png", width=100)
        st.title("AI Forecaster v4.0")
        st.info("Dual-Model Architecture")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 30:
                st.error("Need at least 30 days of historical data for a reliable forecast.")
            else:
                with st.spinner("🧠 Running dual-model forecast... (LGBM + Prophet)"):
                    forecast_df, prophet_model = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.prophet_model = prophet_model # Save the Prophet model for insights
                
                if not forecast_df.empty:
                    st.success("Forecast Generated Successfully!")
                else:
                    st.error("Forecast generation failed. Check data for errors.")

    # --- Main Page Tabs ---
    tab_list = ["🔮 Forecast Dashboard", "💡 ATV Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df_to_show = st.session_state.forecast_df.rename(columns={
                'ds': 'Date', 
                'forecast_customers': 'Predicted Customers',
                'forecast_atv': 'Predicted Avg Sale (₱)', 
                'forecast_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            
            # Style the dataframe for better readability
            st.dataframe(
                df_to_show.style.format({
                    'Predicted Customers': '{:,.0f}',
                    'Predicted Avg Sale (₱)': '₱{:,.2f}',
                    'Predicted Sales (₱)': '₱{:,.2f}'
                }),
                use_container_width=True,
                height=560
            )
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    with tabs[1]:
        st.header("💡 ATV Model Insights (from Prophet)")
        st.info("This chart shows the components of the ATV forecast: the overall trend, weekly patterns, and the impact of special events like paydays.")
        if st.session_state.prophet_model:
            fig = st.session_state.prophet_model.plot_components(
                st.session_state.prophet_model.predict(st.session_state.prophet_model.history)
            )
            # Customize plot for dark theme
            for ax in fig.get_axes():
                ax.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
            fig.set_facecolor('#1a1a1a')
            st.pyplot(fig)
        else:
            st.info("Generate a forecast to see the breakdown of the ATV model's components.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        historical_df = load_from_firestore(db, 'historical_data')
        if not historical_df.empty:
            st.info("No editing functionality is implemented in this version.")
        else:
            st.warning("No historical data found in Firestore.")
else:
    st.error("Could not connect to Firestore. Please check your Streamlit secrets configuration.")
