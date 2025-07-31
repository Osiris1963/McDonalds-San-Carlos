import streamlit as st
import pandas as pd
from prophet import Prophet
import logging
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import bcrypt
import time
import plotly.graph_objs as go
import numpy as np

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(
    page_title="McDonald's San Carlos 688",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Hide the default sidebar and hamburger menu */
    [data-testid="stSidebar"], [data-testid="main-menu-button"] {
        display: none;
    }
    /* Main Font & Colors */
    html, body, [class*="st-"], .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>div {
        font-family: 'Poppins', sans-serif;
    }
    .main > div {
        background-color: #F7F7F7;
    }
    /* Custom Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
    }
    .nav-title img {
        height: 30px;
        margin-right: 10px;
    }
    .nav-links {
        display: flex;
        gap: 5px;
    }
    .nav-button {
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        background-color: transparent;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .nav-button:hover {
        background-color: #f0f0f0;
    }
    .nav-button.active {
        background-color: #DA291C;
        color: white;
    }
    .logout-button {
        background-color: #4B5563;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to the Firestore database."""
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- User Management Functions ---
def get_user(db_client, username):
    """Retrieves a user document from Firestore by username."""
    users_ref = db_client.collection('users')
    query = users_ref.where('username', '==', username).limit(1)
    results = list(query.stream())
    return results[0] if results else None

def verify_password(plain_password, hashed_password):
    """Verifies a plain text password against a bcrypt hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- Data Loading ---
@st.cache_data(ttl="10m")
def load_all_data(_db_client):
    """Loads all necessary collections from Firestore into pandas DataFrames."""
    collections = ["historical_data", "future_activities", "future_events", "users", "forecast_log", "forecast_insights"]
    data = {}
    for collection in collections:
        docs = _db_client.collection(collection).stream()
        records = [doc.to_dict() for doc in docs]
        
        df = pd.DataFrame(records)
        
        # Standardize date columns for consistent processing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values(by='date', ascending=False).reset_index(drop=True)
        elif 'ds' in df.columns:
             df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
             df = df.sort_values(by='ds', ascending=False).reset_index(drop=True)
             
        data[collection] = df
    return data

# --- NEW DAY-SPECIFIC FORECASTING ENGINE ---
@st.cache_data(ttl="1h")
def train_and_forecast_day_specific(_historical_df, target_col, horizon):
    """
    Trains a separate Prophet model for each day of the week and forecasts.
    This "apples-to-apples" method improves accuracy by isolating daily patterns.
    """
    # Ensure dataframe has the correct columns and format
    historical_df = _historical_df[['date', target_col]].copy()
    historical_df.rename(columns={'date': 'ds', target_col: 'y'}, inplace=True)
    historical_df['day_of_week'] = historical_df['ds'].dt.dayofweek # Monday=0, Sunday=6

    models = {}

    # Train a model for each day of the week
    for i in range(7):
        day_specific_df = historical_df[historical_df['day_of_week'] == i]
        
        if len(day_specific_df) > 2: # Prophet needs at least 2 data points
            # The seasonality_mode is set to 'multiplicative' as sales often exhibit
            # multiplicative rather than additive patterns (e.g., holidays cause a percentage increase).
            m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            m.add_country_holidays(country_name='PH')
            m.fit(day_specific_df[['ds', 'y']])
            models[i] = m

    # Create future dataframe and predict
    future_dates = pd.date_range(start=historical_df['ds'].max() + timedelta(days=1), periods=horizon)
    forecast_list = []

    for date in future_dates:
        day_of_week = date.dayofweek
        model = models.get(day_of_week)
        
        if model:
            future_df = pd.DataFrame({'ds': [date]})
            prediction = model.predict(future_df)
            forecast_list.append(prediction[['ds', 'yhat']].iloc[0])

    if not forecast_list:
        return pd.DataFrame(columns=['ds', 'yhat']), None

    forecast_df = pd.DataFrame(forecast_list)
    return forecast_df, models # Returning models can be useful for plotting components

# --- UI Rendering Functions (Placeholders) ---
# NOTE: The UI rendering functions are kept as placeholders as requested.
# The focus is on the backend logic. You would fill these with your
# specific Streamlit components (charts, tables, etc.).

def render_dashboard(data):
    st.header("📈 Sales Forecast Dashboard")
    st.info("The core logic has been updated to a day-specific forecasting model for higher accuracy.")
    
    historical_df = data.get("historical_data")
    if historical_df is not None and not historical_df.empty:
        target_col = st.selectbox("Select Target Variable for Forecasting", options=['sales', 'transactions', 'average_sale'], index=0)
        horizon = st.slider("Select Forecast Horizon (Days)", min_value=7, max_value=30, value=14)

        if st.button(f"Generate {horizon}-Day Forecast for {target_col.replace('_', ' ').title()}"):
            with st.spinner("Training 7 specialized models and predicting..."):
                forecast_df, _ = train_and_forecast_day_specific(historical_df, target_col, horizon)
            
            if not forecast_df.empty:
                st.subheader("Forecast Results")
                
                # Plotting forecast
                fig = go.Figure()
                # Historical Data
                fig.add_trace(go.Scatter(x=historical_df['date'], y=historical_df[target_col], mode='lines', name='Historical Data'))
                # Forecasted Data
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name='Forecasted Data', line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title=f"Forecast for {target_col.replace('_', ' ').title()}",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    font=dict(family="Poppins, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'}).set_index('Date'))
            else:
                st.warning("Not enough historical data to generate a forecast.")
    else:
        st.warning("No historical data found. Please add data via the 'Add Data' page.")

def render_insights(data):
    st.header("💡 Forecast Insights")
    st.write("This page will provide deeper insights into the generated forecasts.")

def render_evaluator(data):
    st.header("⚖️ Model Evaluator")
    st.write("This page will allow for the evaluation of model performance over time.")
    
def render_add_data(db):
    st.header("➕ Add or Upload Data")
    st.write("Use this section to add new historical sales data or other relevant information.")

def render_activities(db, data):
    st.header("🎉 Events & Activities")
    st.write("Manage future events and promotional activities here.")

def render_historical(db, data):
    st.header("📚 Historical Data")
    st.write("View and manage all historical data records.")
    if "historical_data" in data and not data["historical_data"].empty:
        st.dataframe(data["historical_data"], use_container_width=True)

def render_admin(db, data):
    st.header("⚙️ Admin Panel")
    st.write("Manage users and system settings.")

# --- Main Application ---
db = init_firestore()

# Initialize session state variables if they don't exist
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = False
    st.session_state.username = None
    st.session_state.access_level = 3 # Default access level (read-only)
    st.session_state.current_view = "dashboard"

# --- Login Interface ---
if not st.session_state.authentication_status:
    st.title("McDonald's San Carlos 688 Forecasting Suite")
    username = st.text_input("Username (Email)")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if db:
            user_record = get_user(db, username)
            if user_record and verify_password(password, user_record.to_dict().get('password', '')):
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.session_state.access_level = user_record.to_dict().get('access_level', 3)
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        else:
            st.error("Database connection failed. Please check credentials.")
# --- Main Application Interface ---
else:
    # --- Custom Navigation Bar ---
    st.markdown("""
        <div class="nav-container">
            <div class="nav-title">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png" alt="Logo">
                <span>McDonald's San Carlos 688</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Load Data ---
    data = load_all_data(db) if db else {}

    # --- Navigation Tabs ---
    tabs = ["Dashboard", "Insights", "Evaluator", "Add Data", "Activities", "History"]
    if st.session_state.access_level == 1: # Only show Admin tab to level 1 users
        tabs.append("Admin")
    
    # Use st.columns to create a horizontal button layout
    cols = st.columns(len(tabs) + 1)
    for i, tab in enumerate(tabs):
        if cols[i].button(tab, key=f"nav_{tab}", use_container_width=True):
            # Convert tab name to a simple key for session state
            st.session_state.current_view = tab.lower().replace(" ", "")
    
    # Logout button at the end
    if cols[-1].button("Logout", key="logout", use_container_width=True):
        st.session_state.authentication_status = False
        st.session_state.username = None
        st.session_state.access_level = 3
        st.session_state.current_view = "dashboard"
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Display Current View Based on Session State ---
    view = st.session_state.current_view
    if view == "dashboard":
        render_dashboard(data)
    elif view == "insights":
        render_insights(data)
    elif view == "evaluator":
        render_evaluator(data)
    elif view == "adddata":
        render_add_data(db)
    elif view == "activities":
        render_activities(db, data)
    elif view == "history":
        render_historical(db, data)
    elif view == "admin" and st.session_state.access_level == 1:
        render_admin(db, data)
    else:
        # Default to dashboard if view is somehow invalid
        render_dashboard(data)
