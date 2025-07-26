import streamlit as st
import pandas as pd
from prophet import Prophet
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Suppress informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Forecaster",
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
        [data-testid="stSidebar-resize-handler"] {
            display: none;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
            border: none;
            padding: 10px 16px;
        }
        .stButton:has(button:contains("Generate")),
        .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37);
            color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        .stButton:has(button:contains("Refresh")),
        .stButton:has(button:contains("View All")),
        .stButton:has(button:contains("Back to Overview")) > button {
            border: 2px solid #c8102e;
            background: transparent;
            color: #c8102e;
        }
        .stButton:has(button:contains("Refresh")):hover > button,
        .stButton:has(button:contains("View All")):hover > button,
        .stButton:has(button:contains("Back to Overview")):hover > button {
            background: #c8102e;
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background-color: transparent;
            color: #d3d3d3;
            padding: 8px 14px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #c8102e;
            color: #ffffff;
        }
        .st-expander {
            border: 1px solid #444 !important;
            box-shadow: none;
            border-radius: 10px;
            background-color: #252525;
            margin-bottom: 0.5rem;
        }
        .st-expander header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #d3d3d3;
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
        st.error(f"Firestore Connection Error: Failed to initialize Firebase. Please check your Streamlit Secrets configuration. Error details: {e}")
        return None

# --- App State Management ---
def initialize_state_firestore(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {
        'forecast_df': pd.DataFrame(), 
        'metrics': {}, 
        'name': "Store 688", 
        'authentication_status': True,
        'access_level': 1,
        'username': "Admin",
        'forecast_components': pd.DataFrame(), 
        'migration_done': False,
        'show_recent_entries': False,
        'show_all_activities': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="1h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    
    docs = _db_client.collection(collection_name).stream()
    records = [doc.to_dict() for doc in docs]
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = df['date'].dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'potential_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

def remove_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    original_rows = len(df)
    cleaned_df = df[df[column] <= upper_bound].copy()
    removed_rows = original_rows - len(cleaned_df)
    return cleaned_df, removed_rows, upper_bound

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    sales = pd.to_numeric(df['base_sales'], errors='coerce').fillna(0)
    customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'): atv = np.divide(sales, customers)
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast"
        params={"latitude":10.48, "longitude":123.42, "daily":"weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max", "timezone":"Asia/Manila", "forecast_days":days}
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation','wind_speed_10m_max':'wind_speed'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather']=df['weather_code'].apply(map_weather_code)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch weather data. Please try again later. Error: {e}")
        return None

def map_weather_code(code):
    if code in [0, 1]: return "Sunny"
    if code == 2: return "Partly Cloudy"
    if code == 3: return "Cloudy"
    if code in [45, 48]: return "Foggy"
    if code in [51, 53, 55, 61, 63, 65]: return "Rainy"
    if code in [80, 81, 82]: return "Rain Showers"
    if code in [95, 96, 99]: return "Thunderstorm"
    return "Cloudy"

def generate_recurring_local_events(start_date,end_date):
    local_events=[];current_date=start_date
    while current_date<=end_date:
        if current_date.day in[15,30]:local_events.append({'holiday':'Payday','ds':current_date,'lower_window':0,'upper_window':1});[local_events.append({'holiday':'Near_Payday','ds':current_date-timedelta(days=i),'lower_window':0,'upper_window':0})for i in range(1,3)]
        if current_date.month==7 and current_date.day==1:local_events.append({'holiday':'San Carlos Charter Day','ds':current_date,'lower_window':0,'upper_window':0})
        current_date+=timedelta(days=1)
    return pd.DataFrame(local_events)

# --- Core Forecasting Models ---

@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    
    if df_train.empty or len(df_train) < 15:
        return pd.DataFrame(), None

    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)

    manual_events_renamed = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_manual_events = pd.concat([manual_events_renamed, recurring_events])
    all_manual_events.dropna(subset=['ds', 'holiday'], inplace=True)
    
    use_yearly_seasonality = len(df_train) >= 365

    prophet_model = Prophet(
        growth='linear',
        holidays=all_manual_events,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=use_yearly_seasonality, 
        changepoint_prior_scale=0.15, 
        changepoint_range=0.9,
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    
    return forecast[['ds', 'yhat']], prophet_model

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource
def train_and_forecast_lstm(historical_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)

    if df_train.empty or len(df_train) < 15: 
        st.warning(f"Not enough data for LSTM model on {target_col}. Need at least 15 data points.")
        return pd.DataFrame()

    data = df_train[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) >= 50:
        SEQUENCE_LENGTH = 7
    elif len(scaled_data) >= 30:
        SEQUENCE_LENGTH = 5
    else:
        SEQUENCE_LENGTH = 3

    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    if len(X) == 0:
        st.warning(f"Could not create training sequences for LSTM on {target_col}.")
        return pd.DataFrame()

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    if len(X) >= 50: 
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(units=50),
            Dense(units=25),
            Dense(units=1)
        ])
    else: 
        model = Sequential([
            LSTM(units=30, input_shape=(X.shape[1], 1)),
            Dense(units=15),
            Dense(units=1)
        ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    
    forecasted_values = []
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    current_sequence = last_sequence.reshape((1, SEQUENCE_LENGTH, 1))

    for _ in range(periods):
        prediction = model.predict(current_sequence, verbose=0)
        forecasted_values.append(prediction[0,0])
        new_sequence = np.append(current_sequence[0, 1:], prediction, axis=0)
        current_sequence = new_sequence.reshape((1, SEQUENCE_LENGTH, 1))

    forecast_scaled = np.array(forecasted_values).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled)

    future_dates = pd.to_datetime(df_train['date'].iloc[-1]) + pd.to_timedelta(np.arange(1, periods + 1), 'D')
    
    in_sample_preds_scaled = model.predict(X, verbose=0)
    in_sample_preds = scaler.inverse_transform(in_sample_preds_scaled)
    
    in_sample_dates = df_train['date'].iloc[SEQUENCE_LENGTH:]
    
    in_sample_df = pd.DataFrame({'ds': in_sample_dates, 'yhat': in_sample_preds.flatten()})
    future_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast.flatten()})

    full_forecast_df = pd.concat([in_sample_df, future_df], ignore_index=True)
    return full_forecast_df

def train_and_forecast_stacked_ensemble(prophet_f, model_f, historical_target, target_col_name):
    base_forecasts = pd.merge(prophet_f[['ds', 'yhat']], model_f[['ds', 'yhat']], on='ds', suffixes=('_prophet', '_model'))
    training_data = pd.merge(base_forecasts, historical_target[['date', target_col_name]], left_on='ds', right_on='date')
    
    X_meta = training_data[['yhat_prophet', 'yhat_model']]
    y_meta = training_data[target_col_name]
    
    if len(X_meta) < 20:
        st.warning(f"Not enough historical data to train a stacking model for {target_col_name}. Falling back to simple averaging.")
        combined = base_forecasts.copy()
        combined['yhat'] = (combined['yhat_prophet'] + combined['yhat_model']) / 2
        return combined[['ds', 'yhat']]

    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)
    
    X_future_meta = base_forecasts[['yhat_prophet', 'yhat_model']]
    stacked_prediction = meta_model.predict(X_future_meta)
    
    final_forecast = base_forecasts[['ds']].copy()
    final_forecast['yhat'] = stacked_prediction
    
    return final_forecast

# ... [All helper functions from add_to_firestore to render_historical_record are unchanged] ...

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
        st.info("Forecasting with a Deep Learning AI Ensemble")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.cache_resource.clear(); st.success("Data cache cleared."); time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            # --- FIXED: Automatically clear cache to ensure latest data is always used ---
            st.cache_data.clear()
            st.cache_resource.clear()

            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                with st.spinner("🧠 Initializing Deep Learning AI Forecast..."):
                    base_df = st.session_state.historical_df.copy()
                    base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                    cleaned_df, removed_count, upper_bound = remove_outliers_iqr(base_df, column='base_sales')
                    if removed_count > 0: st.warning(f"Removed {removed_count} outlier day(s) with base sales over ₱{upper_bound:,.2f}.")
                    hist_df_with_atv = calculate_atv(cleaned_df)
                    ev_df = st.session_state.events_df.copy()
                    FORECAST_HORIZON = 15
                    
                    with st.spinner("Running Prophet & Adaptive Deep Learning (LSTM) base models..."):
                        prophet_atv_f, _ = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                        lstm_atv_f = train_and_forecast_lstm(hist_df_with_atv, FORECAST_HORIZON, 'atv')
                        
                        prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                        lstm_cust_f = train_and_forecast_lstm(hist_df_with_atv, FORECAST_HORIZON, 'customers')

                    with st.spinner("Stacking models for final prediction..."):
                        atv_f = train_and_forecast_stacked_ensemble(prophet_atv_f, lstm_atv_f, hist_df_with_atv, 'atv')
                        cust_f = train_and_forecast_stacked_ensemble(prophet_cust_f, lstm_cust_f, hist_df_with_atv, 'customers')

                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                        
                        with st.spinner("🛰️ Fetching live weather..."):
                            weather_df = get_weather_forecast()
                        if weather_df is not None:
                            combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                        else:
                            combo_f['weather'] = 'Not Available'
                            
                        st.session_state.forecast_df = combo_f
                        
                        try:
                            with st.spinner("📝 Saving forecast log for future accuracy tracking..."):
                                today_date = pd.to_datetime('today').normalize()
                                future_forecasts_to_log = combo_f[combo_f['ds'] > today_date]
                                for _, row in future_forecasts_to_log.iterrows():
                                    log_entry = {"generated_on": today_date, "forecast_for_date": pd.to_datetime(row['ds']), "predicted_sales": row['forecast_sales'], "predicted_customers": row['forecast_customers']}
                                    doc_id = f"{today_date.strftime('%Y-%m-%d')}_{pd.to_datetime(row['ds']).strftime('%Y-%m-%d')}"
                                    db.collection('forecast_log').document(doc_id).set(log_entry)
                            st.info("Forecast log saved successfully.")
                        except Exception as e:
                            st.error(f"Failed to save forecast log: {e}")
                        
                        if prophet_model_cust:
                            base_prophet_full_future = prophet_model_cust.make_future_dataframe(periods=FORECAST_HORIZON)
                            prophet_forecast_components = prophet_model_cust.predict(base_prophet_full_future)
                            st.session_state.forecast_components = prophet_forecast_components
                            st.session_state.all_holidays = prophet_model_cust.holidays
                        
                        st.success("Deep Learning AI forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed. One or more components could not be built.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data", "📅 Future Activities", "📜 Historical Data"]
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        if not st.session_state.forecast_df.empty:
            today=pd.to_datetime('today').normalize();future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
            if future_forecast_df.empty:st.warning("Forecast contains no future dates.")
            else:
                disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns};display_df=future_forecast_df.rename(columns=existing_disp_cols);final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
                st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                st.markdown("#### Forecast Visualization");fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
            with st.expander("🔬 View Full Model Diagnostic Chart"):
                st.info("This chart shows how the final model forecast compares against historical data.");
                d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);
                hist_atv=calculate_atv(st.session_state.historical_df.copy())
                customer_plot_df = st.session_state.forecast_df.rename(columns={'forecast_customers': 'yhat'})
                atv_plot_df = st.session_state.forecast_df.rename(columns={'forecast_atv': 'yhat'})
                with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv, customer_plot_df, st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv, atv_plot_df, st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    
    with tabs[1]:
        st.header("💡 Forecast Insights")
        st.info("The breakdown below is generated by the Prophet model component, showing the foundational drivers like trend and seasonality before the final stacking process with the LSTM model.")
        if st.session_state.forecast_components.empty:
            st.info("Click 'Generate Forecast' to see a breakdown of what drives the daily predictions.")
        else:
            future_components = st.session_state.forecast_components[st.session_state.forecast_components['ds'] >= pd.to_datetime('today').normalize()].copy()
            if not future_components.empty:
                cust_forecast_final = st.session_state.forecast_df[['ds', 'forecast_customers']].rename(columns={'forecast_customers': 'final_yhat'})
                future_components = pd.merge(future_components, cust_forecast_final, on='ds', how='left')
                future_components['yhat'] = future_components['final_yhat'].fillna(future_components['yhat'])
                future_components['date_str'] = future_components['ds'].dt.strftime('%A, %B %d, %Y')
                
                selected_date_str = st.selectbox("Select a day to analyze its forecast drivers:", options=future_components['date_str'])
                selected_date = pd.to_datetime(future_components[future_components['date_str'] == selected_date_str]['ds'].iloc[0])
                
                breakdown_fig, day_data = plot_forecast_breakdown(future_components, selected_date, st.session_state.all_holidays)
                st.plotly_chart(breakdown_fig, use_container_width=True)
                st.markdown("---")
                st.subheader("Prophet Insight Summary")
                st.markdown(generate_insight_summary(day_data, selected_date))
            else:
                st.warning("No future dates available in the forecast components to analyze.")
    
    # ... [The remaining tabs (2 through 5) and their render functions are unchanged] ...
    with tabs[2]:
        # ... Renders the Forecast Evaluator tab ...
        pass
    with tabs[3]:
        # ... Renders the Add/Edit Data tab ...
        pass
    with tabs[4]:
        # ... Renders the Future Activities tab ...
        pass
    with tabs[5]:
        # ... Renders the Historical Data tab ...
        pass
