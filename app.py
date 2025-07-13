import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # For Stacking ATV
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
import bcrypt

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
            background-color: #1a1a1a; /* Original dark background */
        }
        
        /* --- Clean Layout Adjustments --- */
        .block-container {
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #252525; /* Original sidebar color */
            border-right: 1px solid #444;
            width: 320px !important;
        }
        [data-testid="stSidebar-resize-handler"] {
            display: none;
        }
        
        /* --- Primary & Secondary Buttons --- */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
            border: none;
            padding: 10px 16px;
        }
        /* Primary Action Button Style (e.g., Generate Forecast, Save) */
        .stButton:has(button:contains("Generate")),
        .stButton:has(button:contains("Save")),
        .stButton:has(button:contains("Evaluate")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); /* Red gradient */
            color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button,
        .stButton:has(button:contains("Evaluate")):hover > button {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        /* Secondary Action Button (e.g., Refresh, View All) */
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

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background-color: transparent;
            color: #d3d3d3;
            padding: 8px 14px; /* Compact padding */
            font-weight: 600;
            font-size: 0.9rem; /* Smaller font for tabs */
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #c8102e;
            color: #ffffff;
        }

        /* --- Expanders for Editing --- */
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
    if 'historical_forecasts' not in st.session_state: st.session_state.historical_forecasts = load_from_firestore(db_client, 'historical_forecasts')
    defaults = {
        'forecast_df': pd.DataFrame(),
        'metrics': {},
        'name': "Store 688",
        'authentication_status': False,
        'access_level': 0,
        'username': None,
        'forecast_components': pd.DataFrame(),
        'migration_done': False,
        'show_recent_entries': False,
        'show_all_activities': False,
        'evaluation_df': pd.DataFrame()
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- User Management Functions ---
def get_user(db_client, username):
    users_ref = db_client.collection('users')
    query = users_ref.where('username', '==', username).limit(1)
    results = list(query.stream())
    if results:
        return results[0]
    return None

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="1h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()

    docs = _db_client.collection(collection_name).stream()
    records = []
    for doc in docs:
        record = doc.to_dict()
        record['doc_id'] = doc.id
        records.append(record)

    if not records: return pd.DataFrame()

    df = pd.DataFrame(records)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'potential_sales', 'forecast_sales', 'forecast_customers']
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
        params={
            "latitude":10.48, "longitude":123.42,
            "daily":"weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max",
            "timezone":"Asia/Manila", "forecast_days":days,
        }
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation','wind_speed_10m_max':'wind_speed'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather']=df['weather_code'].apply(map_weather_code)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch weather data. Error: {e}")
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
def create_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df = df.sort_values('date')
    df['sales_lag_7'] = df['sales'].shift(7)
    df['customers_lag_7'] = df['customers'].shift(7)
    df['sales_rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    df['customers_rolling_mean_7'] = df['customers'].shift(1).rolling(window=7, min_periods=1).mean()
    df['sales_rolling_std_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).std()
    return df

@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    if df_train.empty or len(df_train) < 15: return pd.DataFrame(), None
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    manual_events_renamed = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_manual_events = pd.concat([manual_events_renamed, recurring_events])
    all_manual_events.dropna(subset=['ds', 'holiday'], inplace=True)
    use_yearly_seasonality = len(df_train) >= 365
    prophet_model = Prophet(growth='linear', holidays=all_manual_events, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=use_yearly_seasonality, changepoint_prior_scale=0.05, changepoint_range=0.8)
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    return forecast[['ds', 'yhat']], prophet_model

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    if df_train.empty: return pd.DataFrame()
    last_date = df_train['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    future_df_template = pd.DataFrame({'date': future_dates})
    full_df = pd.concat([df_train, future_df_template], ignore_index=True)
    full_df_featured = create_advanced_features(full_df)
    df_featured = full_df_featured[full_df_featured['date'] <= last_date].copy()
    future_df_featured = full_df_featured[full_df_featured['date'] > last_date].copy()
    df_featured.dropna(inplace=True)
    features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7']
    features = [f for f in features if f in df_featured.columns]
    target = target_col
    X = df_featured[features]; y = df_featured[target]
    if X.empty or len(X) < 10: st.warning("Not enough data for XGBoost model after feature engineering."); return pd.DataFrame()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    def objective(trial):
        params = {'objective': 'reg:squarederror', 'n_estimators': trial.suggest_int('n_estimators', 500, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'random_state': 42, 'early_stopping_rounds': 50}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test); mae = mean_absolute_error(y_test, preds); return mae
    study = optuna.create_study(direction='minimize'); study.optimize(objective, n_trials=50)
    best_params = study.best_params; best_params.pop('early_stopping_rounds', None)
    final_model = xgb.XGBRegressor(**best_params); final_model.fit(X, y)
    X_future = future_df_featured[features]; future_predictions = final_model.predict(X_future)
    full_prediction_df = df_featured[['date']].copy(); full_prediction_df['yhat'] = final_model.predict(X)
    future_df_template['yhat'] = future_predictions
    final_df = pd.concat([full_prediction_df, future_df_template], ignore_index=True).rename(columns={'date': 'ds'})
    return final_df

# --- Plotting & Data I/O ---
def add_to_firestore(db_client, collection_name, data, historical_df):
    if db_client is None: return
    if 'date' in data and pd.notna(data['date']):
        current_date = pd.to_datetime(data['date'])
        last_year_date = current_date - timedelta(days=364)
        hist_copy = historical_df.copy()
        hist_copy['date_only'] = pd.to_datetime(hist_copy['date']).dt.date
        last_year_record = hist_copy[hist_copy['date_only'] == last_year_date.date()]
        if not last_year_record.empty:
            data['last_year_sales'] = last_year_record['sales'].iloc[0]
            data['last_year_customers'] = last_year_record['customers'].iloc[0]
        else: data['last_year_sales'] = 0.0; data['last_year_customers'] = 0.0
        data['date'] = current_date.to_pydatetime()
    else: return
    all_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'weather']
    for col in all_cols:
        if col in data and data[col] is not None:
            if col not in ['weather']: data[col] = float(pd.to_numeric(data[col], errors='coerce'))
        else:
            if col not in ['weather']: data[col] = 0.0
            else: data[col] = "N/A"
    db_client.collection(collection_name).add(data)

def update_historical_record_in_firestore(db_client, doc_id, data):
    if db_client: db_client.collection('historical_data').document(doc_id).update(data)

def update_activity_in_firestore(db_client, doc_id, data):
    if db_client:
        if 'potential_sales' in data: data['potential_sales'] = float(data['potential_sales'])
        db_client.collection('future_activities').document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client: db_client.collection(collection_name).document(doc_id).delete()

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_evaluation_chart(df, y_true, y_pred, title, y_axis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df[y_true], mode='lines+markers', name='Actual', line=dict(color='royalblue', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df['date'], y=df[y_pred], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash', width=2), marker=dict(symbol='x', size=8, color='#d62728')))
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'color': 'black'}},
        xaxis_title='Date', yaxis_title=y_axis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black')),
        plot_bgcolor='white', paper_bgcolor='white', font_color='black',
        xaxis=dict(title_font_color='black', tickfont=dict(color='black'), gridcolor='#dddddd', linecolor='black'),
        yaxis=dict(title_font_color='black', tickfont=dict(color='black'), gridcolor='#dddddd', linecolor='black')
    )
    return fig

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']): x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']): x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']): holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"; x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')
    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data.get('weekly', 0),'Time of Year':day_data.get('yearly', 0),'Holidays/Events':day_data.get('holidays', 0)}
    significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data.get('trend', 0):.0f} customers**.\n\n"
    if not significant_effects: summary += f"The final forecast of **{day_data.get('yhat', 0):.0f} customers** is driven primarily by this trend."; return summary
    pos_drivers={k:v for k,v in significant_effects.items()if v>0};neg_drivers={k:v for k,v in significant_effects.items()if v<0}
    if pos_drivers:biggest_pos_driver=max(pos_drivers,key=pos_drivers.get);summary+=f"📈 Main positive driver is **{biggest_pos_driver}**,adding an estimated **{pos_drivers[biggest_pos_driver]:.0f} customers**.\n"
    if neg_drivers:biggest_neg_driver=min(neg_drivers,key=neg_drivers.get);summary+=f"📉 Main negative driver is **{biggest_neg_driver}**,reducing by **{abs(neg_drivers[biggest_neg_driver]):.0f} customers**.\n"
    summary+=f"\nAfter all factors,the final forecast is **{day_data.get('yhat', 0):.0f} customers**.";return summary

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)

    if not st.session_state["authentication_status"]:
        st.markdown("<style>div[data-testid='stHorizontalBlock'] { margin-top: 5%; }</style>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.5, 1, 1.5])
        with col2:
            with st.container(border=True):
                st.title("Login")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", use_container_width=True):
                    user = get_user(db, username)
                    if user:
                        user_data = user.to_dict()
                        if verify_password(password, user_data['password']):
                            st.session_state['authentication_status'] = True
                            st.session_state['username'] = username
                            st.session_state['access_level'] = user_data['access_level']
                            st.rerun()
                        else: st.error("Incorrect password")
                    else: st.error("User not found")
    else:
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
            st.info("Forecasting with a Hybrid Ensemble Model")

            if st.button("🔄 Refresh Data from Firestore"):
                st.cache_data.clear(); st.cache_resource.clear()
                st.success("Data cache cleared. Rerunning.")
                time.sleep(1); st.rerun()

            if st.button("📈 Generate Forecast", use_container_width=True):
                if len(st.session_state.historical_df) < 50:
                    st.error("Please provide at least 50 days of data for reliable forecasting.")
                else:
                    with st.spinner("🧠 Initializing Hybrid Forecast..."):
                        base_df = st.session_state.historical_df.copy()
                        base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                        cleaned_df, removed_count, upper_bound = remove_outliers_iqr(base_df, column='base_sales')
                        if removed_count > 0: st.warning(f"Removed {removed_count} outlier day(s) with base sales over ₱{upper_bound:,.2f}.")
                        hist_df_with_atv = calculate_atv(cleaned_df); ev_df = st.session_state.events_df.copy()
                        FORECAST_HORIZON = 15; cust_f = pd.DataFrame(); atv_f = pd.DataFrame()

                        with st.spinner("Forecasting Customers..."):
                            prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                            xgb_cust_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                            if not prophet_cust_f.empty and not xgb_cust_f.empty: cust_f = pd.merge(prophet_cust_f, xgb_cust_f, on='ds', suffixes=('_prophet', '_xgb')); cust_f['yhat'] = (cust_f['yhat_prophet'] + cust_f['yhat_xgb']) / 2
                            else: st.error("Failed to generate customer forecast.")

                        with st.spinner("Forecasting Average Sale..."):
                            VALIDATION_PERIOD = 30; train_df = hist_df_with_atv.iloc[:-VALIDATION_PERIOD]; validation_df = hist_df_with_atv.iloc[-VALIDATION_PERIOD:]
                            prophet_atv_val, _ = train_and_forecast_prophet(train_df, ev_df, VALIDATION_PERIOD, 'atv')
                            xgb_atv_val = train_and_forecast_xgboost_tuned(train_df, ev_df, VALIDATION_PERIOD, 'atv')
                            meta_model_atv = None
                            if not prophet_atv_val.empty and not xgb_atv_val.empty:
                                validation_preds_atv = pd.merge(prophet_atv_val[['ds', 'yhat']], xgb_atv_val[['ds', 'yhat']], on='ds', suffixes=('_prophet', '_xgb'))
                                validation_data_atv = pd.merge(validation_preds_atv, validation_df[['date', 'atv']], left_on='ds', right_on='date')
                                meta_model_atv = LinearRegression(); X_meta_atv = validation_data_atv[['yhat_prophet', 'yhat_xgb']]; y_meta_atv = validation_data_atv['atv']; meta_model_atv.fit(X_meta_atv, y_meta_atv)
                            if meta_model_atv:
                                prophet_atv_f, _ = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                                xgb_atv_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                                if not prophet_atv_f.empty and not xgb_atv_f.empty: atv_f = pd.merge(prophet_atv_f, xgb_atv_f, on='ds', suffixes=('_prophet', '_xgb')); final_X_meta_atv = atv_f[['yhat_prophet', 'yhat_xgb']]; atv_f['yhat'] = meta_model_atv.predict(final_X_meta_atv)
                                else: st.error("Failed to generate ATV forecast.")
                            else: st.error("Failed to train ATV meta-model.")

                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_customers'}), atv_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            st.session_state.forecast_df = combo_f

                            historical_part_of_forecast = combo_f[combo_f['ds'] < pd.to_datetime('today').normalize()]
                            for _, row in historical_part_of_forecast.iterrows():
                                doc_ref = db.collection('historical_forecasts').document(row['ds'].strftime('%Y-%m-%d'))
                                doc_ref.set({
                                    'date': row['ds'],
                                    'forecast_sales': row['forecast_sales'],
                                    'forecast_customers': row['forecast_customers']
                                })
                            st.session_state.historical_forecasts = load_from_firestore(db, 'historical_forecasts')

                            if prophet_model_cust: st.session_state.forecast_components = prophet_model_cust.predict(prophet_cust_f[['ds']]); st.session_state.all_holidays = prophet_model_cust.holidays
                            st.success("Hybrid ensemble forecast generated and saved!")
                        else: st.error("Forecast generation failed.")

            st.markdown("---")
            st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
            st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)
            if st.button("Logout"): st.session_state['authentication_status'] = False; st.session_state['username'] = None; st.session_state['access_level'] = 0; st.rerun()

        tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data", "📅 Future Activities", "📜 Historical Data"]
        if st.session_state['access_level'] == 1: tab_list.append("👥 User Interface")

        tabs = st.tabs(tab_list)

        with tabs[0]: # Forecast Dashboard
            if not st.session_state.forecast_df.empty:
                forecast_df = st.session_state.forecast_df.copy()
                with st.spinner("🛰️ Fetching live weather..."): weather_df = get_weather_forecast()
                if weather_df is not None: forecast_df = pd.merge(forecast_df, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                else: forecast_df['weather'] = 'Not Available'

                future_forecast_df = forecast_df[forecast_df['ds'] >= pd.to_datetime('today').normalize()].copy()
                if not future_forecast_df.empty:
                    st.markdown("#### Forecasted Values")
                    disp_cols = {
                        'ds': 'Date', 'forecast_customers': 'Predicted Customers',
                        'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)',
                        'weather': 'Predicted Weather'
                    }
                    cols_to_show = [col for col in disp_cols.keys() if col in future_forecast_df.columns]
                    display_df = future_forecast_df[cols_to_show].rename(columns=disp_cols)

                    if 'Date' in display_df.columns:
                        display_df = display_df.set_index('Date')

                    st.dataframe(
                        display_df.style.format({
                            'Predicted Customers': '{:,.0f}',
                            'Predicted Avg Sale (₱)': '₱{:,.2f}',
                            'Predicted Sales (₱)': '₱{:,.2f}'
                        }),
                        use_container_width=True, height=560
                    )

                    fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
                else: st.warning("Forecast contains no future dates.")
            else: st.info("Click 'Generate Forecast' in the sidebar to begin.")

        with tabs[1]: # Forecast Insights
            st.info("The breakdown below is generated by the Prophet model component of the forecast.")
            if 'forecast_components' not in st.session_state or st.session_state.forecast_components.empty: st.info("Generate a forecast first to see insights.")
            else:
                future_components=st.session_state.forecast_components[st.session_state.forecast_components['ds']>=pd.to_datetime('today').normalize()].copy()
                if not future_components.empty:
                    future_components['date_str']=future_components['ds'].dt.strftime('%A, %B %d, %Y')
                    selected_date_str=st.selectbox("Select a day to analyze:",options=future_components['date_str'])
                    selected_date=future_components[future_components['date_str']==selected_date_str]['ds'].iloc[0]
                    breakdown_fig,day_data=plot_forecast_breakdown(st.session_state.forecast_components,selected_date,st.session_state.all_holidays)
                    st.plotly_chart(breakdown_fig,use_container_width=True);st.markdown("---");st.subheader("Insight Summary");st.markdown(generate_insight_summary(day_data,selected_date))
                else: st.warning("No future dates available in the forecast components to analyze.")

        with tabs[2]: # Forecast Evaluator
            st.header("📈 Forecast vs. Actual Performance")
            st.info("This tab compares saved historical forecasts against actual data. Use the 'Generate Forecast' button to update the saved forecasts.")

            hist_df = st.session_state.historical_df.copy()
            hist_forecast_df = st.session_state.historical_forecasts.copy()

            if hist_df.empty or hist_forecast_df.empty:
                st.warning("Not enough historical actuals or saved forecasts to evaluate. Please add data and generate a forecast.")
            else:
                # FIX: Ensure data types of the merge key ('date') are identical
                hist_df['date'] = pd.to_datetime(hist_df['date'])
                hist_forecast_df['date'] = pd.to_datetime(hist_forecast_df['date'])

                eval_df = pd.merge(hist_df, hist_forecast_df, on='date', how='inner')

                if 'add_on_sales' in eval_df.columns and 'forecast_sales' in eval_df.columns:
                    eval_df['forecast_total_sales'] = eval_df['forecast_sales'] + eval_df['add_on_sales']
                else:
                    eval_df['forecast_total_sales'] = eval_df.get('forecast_sales', 0)

                eval_df = eval_df.sort_values('date', ascending=False).head(30)

                if eval_df.empty:
                    st.warning("No overlapping data found between historical records and saved forecasts.")
                else:
                    st.markdown("---")
                    st.subheader(f"Accuracy Metrics for the Last {len(eval_df)} Days")

                    sales_mae = mean_absolute_error(eval_df['sales'], eval_df['forecast_total_sales'])
                    sales_mape = mean_absolute_percentage_error(eval_df['sales'], eval_df['forecast_total_sales'])
                    cust_mae = mean_absolute_error(eval_df['customers'], eval_df['forecast_customers'])
                    cust_mape = mean_absolute_percentage_error(eval_df['customers'], eval_df['forecast_customers'])

                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(label="Sales MAPE (Accuracy)", value=f"{100 - sales_mape * 100:.2f}%")
                    m_col1.metric(label="Sales MAE (Avg Error)", value=f"₱{sales_mae:,.2f}")
                    m_col2.metric(label="Customer MAPE (Accuracy)", value=f"{100 - cust_mape * 100:.2f}%")
                    m_col2.metric(label="Customer MAE (Avg Error)", value=f"{cust_mae:,.0f} customers")

                    st.markdown("---")
                    st.subheader("Comparison Charts")
                    sales_fig = plot_evaluation_chart(eval_df, 'sales', 'forecast_total_sales', 'Actual Sales vs. Forecast Sales', 'Sales (₱)')
                    st.plotly_chart(sales_fig, use_container_width=True)
                    cust_fig = plot_evaluation_chart(eval_df, 'customers', 'forecast_customers', 'Actual Customers vs. Forecast Customers', 'Number of Customers')
                    st.plotly_chart(cust_fig, use_container_width=True)
        
        with tabs[3]: # Add/Edit Data
            if st.session_state['access_level'] <= 2:
                # Add original code for this tab here
                pass
            else:
                st.warning("You do not have permission to add or edit data.")
        
        with tabs[4]: # Future Activities
            # Add original code for this tab here
            pass

        with tabs[5]: # Historical Data
            # Add original code for this tab here
            pass
        
        if st.session_state['access_level'] == 1:
            with tabs[6]: # User Interface
                # Add original code for this tab here
                pass
