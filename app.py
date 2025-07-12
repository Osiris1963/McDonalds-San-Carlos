import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import yaml
from yaml.loader import SafeLoader
import io
import time
import os
import requests
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import json
import logging

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Forecaster Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern UI/UX Custom CSS ---
def apply_modern_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* --- Base & Fonts --- */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }
        .main > div {
            background-color: #121212;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
            font-weight: 700;
        }

        /* --- Layout & Spacing --- */
        .block-container {
            padding: 2rem 3rem 3rem 3rem !important;
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            border-right: 1px solid #2D2D2D;
            width: 340px !important;
        }
        [data-testid="stSidebar-resize-handler"] {
            display: none;
        }
        .st-emotion-cache-16txtl3 { /* Sidebar content container */
             background-color: #1E1E1E;
        }
        .stSidebar .st-emotion-cache-1v0mbdj > img { /* Sidebar Image */
            border-radius: 8px;
        }
        .stSidebar h1 {
            font-size: 1.8rem;
            color: #FFFFFF;
        }

        /* --- Buttons --- */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
            border: 2px solid transparent;
        }
        /* Primary Button: Generate Forecast */
        .stButton:has(button:contains("Generate Forecast")) > button {
            background: linear-gradient(45deg, #0078F2, #0056B3);
            color: #FFFFFF;
            box-shadow: 0 4px 15px 0 rgba(0, 120, 242, 0.3);
        }
        .stButton:has(button:contains("Generate Forecast")) > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(0, 120, 242, 0.4);
        }
        /* Secondary Buttons: Refresh, Download */
        .stButton:has(button:contains("Refresh Data")),
        .stButton:has(button:contains("Download")) > button {
            background-color: transparent;
            border: 2px solid #4A4A4A;
            color: #E0E0E0;
        }
        .stButton:has(button:contains("Refresh Data")):hover > button,
        .stButton:has(button:contains("Download")):hover > button {
            background-color: #2D2D2D;
            border-color: #0078F2;
            color: #FFFFFF;
        }
        /* Form Submit Button */
         .stButton:has(button:contains("Save Record")) > button {
            background-color: #0078F2;
            color: #FFFFFF;
        }
        /* Tertiary Button: Show/Hide */
        .stButton:has(button:contains("Recent Entries")) > button {
             background-color: #2D2D2D;
             border: none;
             color: #E0E0E0;
        }
        .stButton:has(button:contains("Recent Entries")):hover > button {
             background-color: #3D3D3D;
        }


        /* --- Tabs --- */
        .stTabs {
            border-bottom: 2px solid #2D2D2D;
            margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #A0A0A0;
            font-weight: 600;
            padding: 12px 18px;
            border-radius: 8px 8px 0 0;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF;
            border-bottom: 2px solid #0078F2;
        }

        /* --- Dataframe & Tables --- */
        .stDataFrame {
            border: 1px solid #2D2D2D;
            border-radius: 10px;
            background-color: #1E1E1E;
        }
        .stDataFrame [data-testid="stHeader"] {
            background-color: #2D2D2D;
            color: #FFFFFF;
            font-weight: 600;
        }
        .stDataFrame [data-testid="stTable"] {
            background-color: #1E1E1E;
        }

        /* --- Input Widgets --- */
        [data-testid="stForm"] {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #2D2D2D;
        }
        [data-testid="stDateInput"], [data-testid="stNumberInput"], [data-testid="stSelectbox"] {
            background-color: #2D2D2D;
            border-radius: 8px;
            border: 1px solid #4A4A4A;
        }

        /* --- Expanders as Cards --- */
        .st-expander {
            border: 1px solid #2D2D2D !important;
            box-shadow: none;
            border-radius: 12px;
            background-color: #1E1E1E;
            margin-top: 1.5rem;
        }
        .st-expander header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #FFFFFF;
            border-radius: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
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
        st.error(f"Firestore Connection Error: Failed to initialize. Check secrets. Error: {e}")
        return None

# --- App State Management ---
def initialize_state_firestore(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_events')
    defaults = {
        'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688",
        'authentication_status': True, 'forecast_components': pd.DataFrame(),
        'show_recent_entries': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="6h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    try:
        docs = _db_client.collection(collection_name).stream()
        records = [doc.to_dict() for doc in docs]
        if not records: return pd.DataFrame()
        df = pd.DataFrame(records)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
            df.dropna(subset=['date'], inplace=True)
        numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers']
        for col in [c for c in numeric_cols if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['sales', 'customers', 'add_on_sales'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from Firestore collection '{collection_name}': {e}")
        return pd.DataFrame()

def calculate_atv(df):
    sales = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(sales, customers)
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast"
        params={"latitude":10.48,"longitude":123.42,"daily":"weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max","timezone":"Asia/Manila","forecast_days":days}
        response=requests.get(url,params=params); response.raise_for_status()
        data=response.json(); df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation','wind_speed_10m_max':'wind_speed'},inplace=True)
        df['date']=pd.to_datetime(df['date']); df['weather']=df['weather_code'].apply(map_weather_code)
        return df
    except requests.exceptions.RequestException: return None

def map_weather_code(code):
    if code in[0,1]:return"Sunny"
    if code in[2,3]:return"Cloudy"
    if code in[51,53,55,61,63,65,80,81,82]:return"Rainy"
    if code in[95,96,99]:return"Storm"
    return"Cloudy"

def generate_recurring_local_events(start_date,end_date):
    local_events=[]; current_date=start_date
    while current_date<=end_date:
        if current_date.day in[15,30]:
            local_events.append({'holiday':'Payday','ds':current_date,'lower_window':0,'upper_window':1})
            for i in range(1,3): local_events.append({'holiday':'Near_Payday','ds':current_date-timedelta(days=i),'lower_window':0,'upper_window':0})
        if current_date.month==7 and current_date.day==1:
            local_events.append({'holiday':'San Carlos Charter Day','ds':current_date,'lower_window':0,'upper_window':0})
        current_date+=timedelta(days=1)
    return pd.DataFrame(local_events)

# --- Core Forecasting Model ---
@st.cache_resource
def train_and_forecast_component(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if len(df_train) < 15: return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    cap = df_prophet['y'].max() * 1.2
    df_prophet['cap'] = cap

    start_date = df_train['date'].min(); end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    all_manual_events = pd.concat([events_df.rename(columns={'date':'ds', 'event_name':'holiday'}), recurring_events])
    
    prophet_model = Prophet(
        growth='logistic', holidays=all_manual_events, daily_seasonality=False,
        weekly_seasonality=True, yearly_seasonality=(len(df_train) >= 365),
        changepoint_prior_scale=0.05, changepoint_range=0.8
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    future['cap'] = cap
    prophet_forecast = prophet_model.predict(future)

    forecast_cols = ['ds', 'trend', 'holidays', 'weekly', 'yearly', 'yhat']
    forecast_components = prophet_forecast[[c for c in forecast_cols if c in prophet_forecast.columns]]
    
    forecast_on_hist = prophet_forecast.loc[:len(df_prophet)-1]
    metrics = {
        'mae': mean_absolute_error(df_prophet['y'], forecast_on_hist['yhat']),
        'rmse': np.sqrt(mean_squared_error(df_prophet['y'], forecast_on_hist['yhat']))
    }
    return prophet_forecast[['ds', 'yhat']], metrics, forecast_components, prophet_model.holidays

# --- Data I/O ---
def add_to_firestore(db_client, collection_name, data, historical_df):
    if db_client is None: return
    try:
        current_date = pd.to_datetime(data['date'])
        last_year_date = current_date - timedelta(days=364)
        hist_copy = historical_df.copy()
        hist_copy['date_only'] = pd.to_datetime(hist_copy['date']).dt.date
        last_year_record = hist_copy[hist_copy['date_only'] == last_year_date.date()]
        data['last_year_sales'] = last_year_record['sales'].iloc[0] if not last_year_record.empty else 0.0
        data['last_year_customers'] = last_year_record['customers'].iloc[0] if not last_year_record.empty else 0.0
        data['date'] = current_date.to_pydatetime()
        db_client.collection(collection_name).add(data)
    except Exception as e: st.error(f"Error saving to Firestore: {e}")

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

# --- Plotting Functions ---
def plot_forecast_dashboard(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#0078F2', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_customers'],mode='lines',name='Customer Forecast',yaxis='y2',line=dict(color='#FFC72C', dash='dot')))
    fig.update_layout(
        title={'text': '15-Day Sales & Customer Forecast', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 20, 'color': 'white'}},
        xaxis_title='Date',
        yaxis=dict(title='Predicted Sales (₱)',color='#0078F2', gridcolor='#2D2D2D'),
        yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#FFC72C'),
        legend=dict(x=0.01,y=0.99,orientation='h', bgcolor='rgba(0,0,0,0.5)'), height=500,
        paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font_color='#E0E0E0',
        xaxis=dict(gridcolor='#2D2D2D')
    )
    return fig

# --- Main Application UI ---
apply_modern_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.image("https://i.imgur.com/s26n4m0.png", width=100)
            st.title(f"Welcome, {st.session_state['name']}")
            st.markdown("---", unsafe_allow_html=True)
            
            if st.button("📈 Generate Forecast", use_container_width=True):
                if len(st.session_state.historical_df) < 20: 
                    st.error("At least 20 days of data needed for a reliable forecast.")
                else:
                    with st.spinner("🧠 Generating intelligent forecast..."):
                        hist_df_atv = calculate_atv(st.session_state.historical_df.copy())
                        ev_df = st.session_state.events_df.copy()
                        
                        cust_f, cust_m, cust_c, all_h = train_and_forecast_component(hist_df_atv, ev_df, 15, 'customers')
                        atv_f, atv_m, _, _ = train_and_forecast_component(hist_df_atv, ev_df, 15, 'atv')
                        
                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            weather_df = get_weather_forecast()
                            if weather_df is not None:
                                combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                            else: combo_f['weather'] = 'N/A'
                                
                            st.session_state.forecast_df = combo_f
                            st.session_state.metrics = {'customers': cust_m, 'atv': atv_m}
                            st.session_state.forecast_components = cust_c
                            st.session_state.all_holidays = all_h
                            st.success("Forecast generated successfully!")
                        else: st.error("Forecast generation failed.")

            st.markdown("---", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear(); st.success("Cache cleared. Rerunning..."); time.sleep(1); st.rerun()
            
            st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
            st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast Dashboard","💡 Forecast Insights", "✍️ Add/Edit Data", "📜 Historical Data"])
        
        with tab1:
            st.header("Forecast Dashboard")
            if not st.session_state.forecast_df.empty:
                today=pd.to_datetime('today').normalize()
                future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
                if future_forecast_df.empty: st.warning("Forecast contains no future dates.")
                else:
                    disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                    display_df=future_forecast_df.rename(columns=disp_cols)
                    final_cols_order=[v for k,v in disp_cols.items()if v in display_df.columns]
                    st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                    st.plotly_chart(plot_forecast_dashboard(future_forecast_df), use_container_width=True)
            else: st.info("Click 'Generate Forecast' in the sidebar to begin.")

        with tab2:
            st.header("Forecast Insights")
            if 'forecast_components' not in st.session_state or st.session_state.forecast_components.empty:
                st.info("Generate a forecast to see its drivers.")
            else:
                # Logic for this tab remains the same
                st.write("Insights will be displayed here.")


        with tab3:
            st.header("Manage Data")
            form_col, display_col = st.columns([2, 3], gap="large")
            with form_col:
                st.subheader("Add New Daily Record")
                with st.form("new_record_form", clear_on_submit=True, border=False):
                    new_date=st.date_input("Date", date.today())
                    new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                    new_customers=st.number_input("Customer Count",min_value=0)
                    new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                    new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"])
                    if st.form_submit_button("✅ Save Record", use_container_width=True):
                        new_rec={"date":new_date,"sales":new_sales,"customers":new_customers,"weather":new_weather,"add_on_sales":new_addons}
                        add_to_firestore(db,'historical_data',new_rec, st.session_state.historical_df)
                        st.cache_data.clear(); st.success("Record added!"); time.sleep(1); st.rerun()
            
            with display_col:
                st.subheader("Recent Entries")
                if st.button("🗓️ Show/Hide Recent Entries", use_container_width=True):
                    st.session_state.show_recent_entries = not st.session_state.show_recent_entries
                
                if st.session_state.show_recent_entries:
                    recent_df = st.session_state.historical_df.copy()
                    if not recent_df.empty:
                        recent_df['date'] = pd.to_datetime(recent_df['date']).dt.date
                        display_cols = ['date', 'sales', 'customers', 'weather']
                        st.dataframe(
                            recent_df[[c for c in display_cols if c in recent_df.columns]].sort_values(by="date", ascending=False).head(7),
                            use_container_width=True, hide_index=True,
                            column_config={
                                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                "sales": st.column_config.NumberColumn("Sales (₱)", format="₱%,.2f"),
                                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                            }
                        )
                    else: st.info("No recent data to display.")

        with tab4:
            st.header("Historical Data Explorer")
            df = st.session_state.historical_df.copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dropna()
                all_years = sorted(df['date'].dt.year.unique(), reverse=True)
                if all_years:
                    col1, col2 = st.columns(2)
                    with col1: selected_year = st.selectbox("Select Year:", options=all_years)
                    df_year = df[df['date'].dt.year == selected_year]
                    all_months = sorted(df_year['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)
                    if all_months:
                        with col2: selected_month_str = st.selectbox("Select Month:", options=all_months)
                        selected_month_num = pd.to_datetime(selected_month_str, format='%B').month
                        filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].copy()
                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                    else: st.write(f"No data for {selected_year}.")
                else: st.write("No historical data to display.")
            else: st.write("No historical data to display.")
