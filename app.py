import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import time
import requests
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import logging

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(
    page_title="Component-Based Sales Forecaster",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom McDonald's Inspired CSS ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* --- Main Font & Colors --- */
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
        .main > div {
            background-color: #1a1a1a;
        }
        .block-container {
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        [data-testid="stSidebar"] {
            background-color: #252525;
            border-right: 1px solid #444;
            width: 320px !important;
        }
        .stButton > button {
            border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out;
            border: none; padding: 10px 16px;
        }
        .stButton:has(button:contains("Generate")),
        .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px); box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px; background-color: transparent; color: #d3d3d3;
            padding: 8px 14px; font-weight: 600; font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #c8102e; color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = {
              "type": st.secrets.firebase_credentials.type, "project_id": st.secrets.firebase_credentials.project_id,
              "private_key_id": st.secrets.firebase_credentials.private_key_id, "private_key": st.secrets.firebase_credentials.private_key.replace('\\n', '\n'),
              "client_email": st.secrets.firebase_credentials.client_email, "client_id": st.secrets.firebase_credentials.client_id,
              "auth_uri": st.secrets.firebase_credentials.auth_uri, "token_uri": st.secrets.firebase_credentials.token_uri,
              "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url, "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- App State Management ---
def initialize_state(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {'forecast_df': pd.DataFrame(), 'metrics': {}, 'forecast_components': pd.DataFrame()}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl="1h", show_spinner="Loading historical data...")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    docs = _db_client.collection(collection_name).stream()
    df = pd.DataFrame([doc.to_dict() for doc in docs])
    if df.empty: return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    df.dropna(subset=['date'], inplace=True)
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(df['base_sales'], df['customers'])
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600, show_spinner="Fetching weather forecast...")
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast"
        params={"latitude":10.48, "longitude":123.42, "daily":"weather_code", "timezone":"Asia/Manila", "forecast_days":days}
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date'},inplace=True); df['date']=pd.to_datetime(df['date'])
        df['weather'] = df['weather_code'].apply(lambda x: "Rainy" if x > 50 else ("Cloudy" if x > 1 else "Sunny"))
        return df[['date', 'weather']]
    except Exception as e:
        st.error(f"Could not fetch weather data: {e}"); return None

def generate_recurring_local_events(start_date,end_date):
    events=[];current_date=start_date
    while current_date<=end_date:
        if current_date.day in[15,30]:events.append({'holiday':'Payday','ds':current_date,'lower_window':0,'upper_window':1})
        if current_date.month==7 and current_date.day==1:events.append({'holiday':'San Carlos Charter Day','ds':current_date,'lower_window':0,'upper_window':0})
        current_date+=timedelta(days=1)
    return pd.DataFrame(events)

# --- RE-ARCHITECTED: Component-Based Forecasting Engine ---
@st.cache_resource(show_spinner="Generating component-based forecast...")
def generate_component_forecast(_historical_df, _events_df, periods, target_col):
    df = _historical_df.copy()
    if df.empty or len(df) < 90: # Reduced data requirement
        st.error(f"Not enough data for {target_col}. Need at least 90 days of historical data.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 1. Create Future DataFrame ---
    last_date = df['date'].max()
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, periods + 1)])
    future_df = pd.DataFrame({'date': future_dates})

    # --- 2. Component: Trend Baseline (from Prophet) ---
    prophet_df = df[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
    all_events = pd.concat([generate_recurring_local_events(df['date'].min(), future_df['date'].max()), _events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})])

    # Model for trend and holidays
    m_trend = Prophet(holidays=all_events, seasonality_mode='multiplicative')
    m_trend.add_country_holidays(country_name='PH')
    m_trend.fit(prophet_df)
    
    full_forecast = m_trend.predict(future_df[['date']].rename(columns={'date':'ds'}))
    future_merged = pd.merge(future_df, full_forecast[['ds', 'trend', 'holidays']], left_on='date', right_on='ds', how='left')
    future_merged.rename(columns={'trend': 'baseline'}, inplace=True)

    # --- 3. Component: Day-of-Week Power ---
    today = pd.to_datetime(date.today())
    df['dayofweek'] = df['date'].dt.dayofweek
    recent_df = df[df['date'] > today - timedelta(days=28)] # Use last 4 weeks
    
    if not recent_df.empty:
        overall_mean = recent_df[target_col].mean()
        dow_factors = recent_df.groupby('dayofweek')[target_col].mean() / overall_mean
    else: # Fallback if no recent data
        dow_factors = df.groupby('dayofweek')[target_col].mean() / df[target_col].mean()

    future_merged['dayofweek'] = future_merged['date'].dt.dayofweek
    future_merged['dow_factor'] = future_merged['dayofweek'].map(dow_factors).fillna(1.0)
    
    # --- 4. Component: Weather Impact ---
    weather_df = get_weather_forecast(periods)
    if weather_df is not None:
        future_merged = pd.merge(future_merged, weather_df, on='date', how='left')
        
        overall_mean_all_time = df[target_col].mean()
        weather_factors = df.groupby('weather')[target_col].mean() / overall_mean_all_time
        future_merged['weather_factor'] = future_merged['weather'].map(weather_factors).fillna(1.0)
    else:
        future_merged['weather_factor'] = 1.0

    # --- 5. Combine Components for Final Forecast ---
    future_merged['holiday_factor'] = future_merged['holidays'].fillna(0) + 1 # Convert additive to multiplicative factor

    future_merged['yhat'] = future_merged['baseline'] * future_merged['dow_factor'] * future_merged['weather_factor'] * future_merged['holiday_factor']
    future_merged['yhat'] = future_merged['yhat'].clip(lower=0)

    # Prepare components for breakdown plot
    components = future_merged[['date', 'baseline']].rename(columns={'date':'ds'})
    components['day_of_week'] = (future_merged['dow_factor'] - 1) * future_merged['baseline']
    components['weather'] = (future_merged['weather_factor'] - 1) * (future_merged['baseline'] + components['day_of_week'])
    components['holidays'] = (future_merged['holiday_factor'] - 1) * (future_merged['baseline'] + components['day_of_week'] + components['weather'])
    components['yhat'] = future_merged['yhat']

    return future_merged[['date', 'yhat']].rename(columns={'date':'ds'}), components

# --- Plotting and UI Functions ---
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_forecast_chart(df, title):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')))
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis=dict(title='Predicted Sales (₱)'), yaxis2=dict(title='Predicted Customers',overlaying='y',side='right'), legend=dict(x=0.01,y=0.99,orientation='h'), height=500, paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
    return fig

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash')))
    fig.update_layout(title=dict(text=title), xaxis_title='Date', yaxis_title=y_axis_title, legend=dict(font=dict(color='white')), height=450, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white')
    return fig

def plot_forecast_breakdown(components,selected_date):
    day_data=components[components['ds']==selected_date].iloc[0]
    x_data = ['Seasonally-Adjusted Trend', 'Day of Week Power', 'Weather Impact', 'Holidays/Events', 'Final Forecast']
    y_data = [day_data.get('baseline', 0), day_data.get('day_of_week', 0), day_data.get('weather', 0), day_data.get('holidays', 0), day_data.get('yhat', 0)]
    measure_data = ["absolute", "relative", "relative", "relative", "total"]
    
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"₱{v:,.0f}" if i != len(y_data)-1 else f"₱{v:,.0f}" for i, v in enumerate(y_data)],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}))
    fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A, %B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white')
    return fig,day_data

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()

if db:
    initialize_state(db)

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("Sales Forecaster")
        st.markdown("---")
        st.info("Component-Based Forecasting Engine")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.success("Cache cleared."); time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 90:
                st.error("Please provide at least 90 days of data for reliable forecasting.")
            else:
                with st.spinner("🧠 Engineering Forecast Components..."):
                    hist_df = st.session_state.historical_df.copy()
                    hist_df = calculate_atv(hist_df)
                    ev_df = st.session_state.events_df.copy()
                    
                    FORECAST_HORIZON = 15
                    
                    cust_f, cust_comp = generate_component_forecast(hist_df, ev_df, FORECAST_HORIZON, 'customers')
                    atv_f, _ = generate_component_forecast(hist_df, ev_df, FORECAST_HORIZON, 'atv')

                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                        
                        weather_df = get_weather_forecast(FORECAST_HORIZON)
                        if weather_df is not None:
                            combo_f = pd.merge(combo_f, weather_df, left_on='ds', right_on='date', how='left').drop(columns=['date'])
                        
                        st.session_state.forecast_df = combo_f
                        st.session_state.forecast_components = cust_comp
                        st.success("Component forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    # --- Main Content Tabs ---
    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data"])
    
    with tabs[0]: # Forecast Dashboard
        if not st.session_state.forecast_df.empty:
            today=pd.to_datetime('today').normalize()
            future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
            disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
            display_df=future_forecast_df.rename(columns=disp_cols)
            final_cols_order=[v for k,v in disp_cols.items()if v in display_df.columns]
            st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
            st.markdown("#### Forecast Visualization");st.plotly_chart(plot_forecast_chart(future_forecast_df, '15-Day Sales & Customer Forecast'), use_container_width=True)
        else: st.info("Click 'Generate Forecast' to begin.")
    
    with tabs[1]: # Forecast Insights
        st.info("This breakdown shows how each component contributes to the final customer forecast.")
        if st.session_state.forecast_components.empty:
            st.info("Generate a forecast first to see its component breakdown.")
        else:
            components=st.session_state.forecast_components.copy()
            components['date_str']=components['ds'].dt.strftime('%A, %B %d, %Y')
            selected_date_str=st.selectbox("Select a day to analyze:",options=components['date_str'])
            selected_date=components[components['date_str']==selected_date_str]['ds'].iloc[0]
            breakdown_fig, _ = plot_forecast_breakdown(components,selected_date)
            st.plotly_chart(breakdown_fig,use_container_width=True)

    with tabs[2]: # Forecast Evaluator
        st.header("📈 Forecast Evaluator")
        # Placeholder for evaluator logic if needed in the future
        st.info("The Forecast Evaluator tab is under construction with the new component model.")

    with tabs[3]: # Add/Edit Data
        st.subheader("✍️ Add New Daily Record")
        with st.form("new_record_form",clear_on_submit=True, border=True):
            new_date=st.date_input("Date", date.today())
            c1, c2, c3 = st.columns(3)
            new_sales=c1.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
            new_customers=c2.number_input("Customer Count",min_value=0)
            new_addons=c3.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
            
            if st.form_submit_button("✅ Save Record", use_container_width=True):
                new_rec={"date":new_date, "sales":new_sales, "customers":new_customers, "add_on_sales":new_addons}
                # Firestore logic would go here
                st.success("Record added (simulation)!");
else:
    st.error("Failed to connect to Firestore. Please check your configuration and network.")
