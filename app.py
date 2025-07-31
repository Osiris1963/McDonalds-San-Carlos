import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
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
import inspect

# --- Suppress informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


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
        
        /* --- Clean Layout Adjustments --- */
        .block-container {
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #252525;
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

        /* --- Tabs --- */
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
    records = []
    for doc in docs:
        record = doc.to_dict()
        record['doc_id'] = doc.id
        records.append(record)
        
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
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear').fillna(0)
        
    return df

def cap_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    capped_count = (df[column] > upper_bound).sum()
    df_capped = df.copy()
    df_capped.loc[df_capped[column] > upper_bound, column] = upper_bound
    
    return df_capped, capped_count, upper_bound

@st.cache_data
def calculate_atv(df):
    df_copy = df.copy()
    df_copy['base_sales'] = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    sales = pd.to_numeric(df_copy['base_sales'], errors='coerce').fillna(0)
    customers = pd.to_numeric(df_copy['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'): atv = np.divide(sales, customers)
    df_copy['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df_copy

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast"
        params={
            "latitude":10.48,
            "longitude":123.42,
            "daily":"weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max",
            "timezone":"Asia/Manila",
            "forecast_days":days,
        }
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation','wind_speed_10m_max':'wind_speed'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather_condition']=df['weather_code'].apply(map_weather_code)
        return df[['date', 'temp_max', 'precipitation', 'wind_speed', 'weather_condition']]
    except requests.exceptions.RequestException as e:
        print(f"Weather API Error: {e}")
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

# --- RE-ENGINEERED: Self-Recalibration Logic ---
def check_performance_and_recalibrate(db, historical_df, degradation_threshold=0.98, short_term_days=7, long_term_days=30):
    try:
        log_docs = db.collection('forecast_log').stream()
        log_records = [doc.to_dict() for doc in log_docs]
        if not log_records: return False

        forecast_log_df = pd.DataFrame(log_records)
        forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.tz_localize(None)
        forecast_log_df['generated_on'] = pd.to_datetime(forecast_log_df['generated_on']).dt.tz_localize(None)
        day_ahead_logs = forecast_log_df[forecast_log_df['forecast_for_date'] - forecast_log_df['generated_on'] == timedelta(days=1)].copy()
        if day_ahead_logs.empty: return False

        true_accuracy_df = pd.merge(historical_df, day_ahead_logs, left_on='date', right_on='forecast_for_date', how='inner')
        if len(true_accuracy_df) < long_term_days: return False 
        
        today = pd.to_datetime('today').normalize()
        
        long_term_start = today - pd.Timedelta(days=long_term_days)
        long_term_df = true_accuracy_df[true_accuracy_df['date'] >= long_term_start]
        if long_term_df.empty: return False
        
        long_term_sales_safe = long_term_df['sales'].replace(0, np.nan)
        long_term_mape = np.nanmean(np.abs((long_term_df['sales'] - long_term_df['predicted_sales']) / long_term_sales_safe)) * 100
        long_term_accuracy = 100 - long_term_mape

        short_term_start = today - pd.Timedelta(days=short_term_days)
        short_term_df = true_accuracy_df[true_accuracy_df['date'] >= short_term_start]
        if short_term_df.empty: return False

        short_term_sales_safe = short_term_df['sales'].replace(0, np.nan)
        short_term_mape = np.nanmean(np.abs((short_term_df['sales'] - short_term_df['predicted_sales']) / short_term_sales_safe)) * 100
        short_term_accuracy = 100 - short_term_mape

        if short_term_accuracy < (long_term_accuracy * degradation_threshold):
            st.warning(f"🚨 Recent 7-day accuracy ({short_term_accuracy:.2f}%) has dropped significantly below the 30-day baseline ({long_term_accuracy:.2f}%). Triggering model recalibration.")
            st.cache_data.clear()
            time.sleep(2) 
            return True

    except Exception as e:
        st.warning(f"Could not perform automatic accuracy check. Error: {e}")
    return False

# --- Core Forecasting Models ---

@st.cache_data
def create_advanced_features(df, weather_df=None):
    df['date'] = pd.to_datetime(df['date'])
    
    if weather_df is not None and not weather_df.empty:
        df = pd.merge(df, weather_df, on='date', how='left')
    
    for col in ['temp_max', 'precipitation', 'wind_speed']:
        if col not in df.columns:
            df[col] = 0 
    df[['temp_max', 'precipitation', 'wind_speed']] = df[['temp_max', 'precipitation', 'wind_speed']].fillna(method='ffill').fillna(0)

    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    def get_paydays(d):
        eom = d + pd.tseries.offsets.MonthEnd(0)
        mid_month = pd.Timestamp(year=d.year, month=d.month, day=15)
        return mid_month, eom

    paydays = []
    for d in df['date']:
        mid, eom = get_paydays(d)
        prev_mid, prev_eom = get_paydays(d - pd.DateOffset(months=1))
        next_mid, next_eom = get_paydays(d + pd.DateOffset(months=1))
        all_paydays = sorted([prev_mid, prev_eom, mid, eom, next_mid, next_eom])
        future_paydays = [pd for pd in all_paydays if pd >= d]
        past_paydays = [pd for pd in all_paydays if pd < d]
        next_payday = min(future_paydays) if future_paydays else next_mid
        last_payday = max(past_paydays) if past_paydays else prev_eom
        paydays.append({'next_payday': next_payday, 'last_payday': last_payday})
        
    payday_df = pd.DataFrame(paydays, index=df.index)
    df['days_until_next_payday'] = (payday_df['next_payday'] - df['date']).dt.days
    df['days_since_last_payday'] = (df['date'] - payday_df['last_payday']).dt.days
    df['is_payday_window'] = df['date'].dt.day.isin([15, 30, 31, 1, 2]).astype(int)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    lags = [7, 14, 21, 28] # Use weekly lags for day-specific models
    for lag in lags:
        if 'sales' in df.columns: df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        if 'customers' in df.columns: df[f'customers_lag_{lag}'] = df['customers'].shift(lag)

    if 'sales' in df.columns:
        df['sales_rolling_mean_4'] = df['sales'].shift(1).rolling(window=4, min_periods=1).mean() # 4-week rolling mean
        df['sales_rolling_std_4'] = df['sales'].shift(1).rolling(window=4, min_periods=1).std()
    if 'customers' in df.columns:
        df['customers_rolling_mean_4'] = df['customers'].shift(1).rolling(window=4, min_periods=1).mean()
        
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temp_max']
    df['payday_weekday_interaction'] = df['is_payday_window'] * df['dayofweek']
        
    return df

@st.cache_data
def train_and_forecast_prophet_day_specific(historical_df, events_df, periods, target_col, day_of_week):
    df_train = historical_df[historical_df['date'].dt.dayofweek == day_of_week].copy()
    if df_train.empty or len(df_train) < 10:
        return pd.DataFrame(), None
        
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})
    
    # Modify seasonality for day-specific model
    prophet_model = Prophet(
        growth='linear', weekly_seasonality=False, daily_seasonality=False,
        yearly_seasonality=True, changepoint_prior_scale=0.1
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    # Create future dataframe for the specific day of the week
    last_date = historical_df['date'].max()
    future_dates = pd.date_range(start=last_date, periods=periods*7) # Go out far enough
    future_day_specific = future_dates[future_dates.dayofweek == day_of_week][:periods]
    future_df = pd.DataFrame({'ds': future_day_specific})
    
    forecast = prophet_model.predict(future_df)
    
    # Combine with historical for stacking
    hist_forecast = prophet_model.predict(df_prophet[['ds']])
    return pd.concat([hist_forecast[['ds', 'yhat']], forecast[['ds', 'yhat']]]), prophet_model

# ==============================================================================
# === DEBUGGED FUNCTION: train_and_forecast_tree_day_specific ==================
# ==============================================================================
@st.cache_data
def train_and_forecast_tree_day_specific(model_class, params, historical_df, periods, target_col, day_of_week, customer_forecast_df=None):
    df_day = historical_df[historical_df['date'].dt.dayofweek == day_of_week].copy()
    if len(df_day) < 20: return pd.DataFrame()

    # --- FIX START: Combine historical and future data BEFORE feature engineering ---
    # 1. Create future date placeholders
    last_date = historical_df['date'].max()
    future_dates_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7)
    future_day_specific_dates = future_dates_range[future_dates_range.dayofweek == day_of_week][:periods]
    future_df_placeholders = pd.DataFrame({'date': future_day_specific_dates})

    # 2. Combine historical data for the specific day with future placeholders
    combined_df = pd.concat([df_day, future_df_placeholders], ignore_index=True)
    
    # 3. Engineer features on the combined dataframe
    # This allows lags and rolling windows to be calculated correctly for future dates
    combined_featured_df = create_advanced_features(combined_df)
    # --- FIX END ---

    if target_col == 'atv' and customer_forecast_df is not None:
        combined_featured_df = pd.merge(combined_featured_df, customer_forecast_df[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecast_customers'}), on='date', how='left')
        combined_featured_df['forecast_customers'].fillna(method='ffill', inplace=True).fillna(method='bfill', inplace=True)

    features = [f for f in combined_featured_df.columns if combined_featured_df[f].dtype in ['int64', 'float64'] and f not in ['sales', 'customers', 'atv', 'date']]
    features = list(set(features) - {target_col})
    
    # Split back into historical (for training) and future (for prediction)
    train_df = combined_featured_df.dropna(subset=[target_col])
    predict_df = combined_featured_df[combined_featured_df[target_col].isna()]

    X_train = train_df[features]
    y_train = train_df[target_col]
    
    # Ensure no NaN values slip into training data
    valid_indices = X_train.dropna().index
    X_train = X_train.loc[valid_indices]
    y_train = y_train.loc[valid_indices]

    if X_train.empty: return pd.DataFrame()

    model = model_class(**params).fit(X_train, y_train)
    
    # Predict on the future dataframe which now has all required features
    X_future = predict_df[features]
    predictions = model.predict(X_future)
    
    future_forecast_df = pd.DataFrame({'ds': predict_df['date'], 'yhat': predictions})
    
    # Historical predictions for stacking
    hist_preds = model.predict(X_train)
    hist_forecast_df = pd.DataFrame({'ds': train_df.loc[X_train.index, 'date'], 'yhat': hist_preds})
    
    return pd.concat([hist_forecast_df, future_forecast_df])


@st.cache_data
def train_and_forecast_stacked_ensemble_day_specific(base_forecasts_dict, historical_target, target_col_name, day_of_week):
    df_day = historical_target[historical_target['date'].dt.dayofweek == day_of_week].copy()
    if df_day.empty: return pd.DataFrame()
    
    final_df = None
    for name, fcst_df in base_forecasts_dict.items():
        if fcst_df is None or fcst_df.empty: continue
        renamed_df = fcst_df[['ds', 'yhat']].rename(columns={'yhat': f'yhat_{name}'})
        if final_df is None: final_df = renamed_df
        else: final_df = pd.merge(final_df, renamed_df, on='ds', how='outer')

    if final_df is None or final_df.empty: return pd.DataFrame()

    final_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    final_df.bfill(inplace=True)

    meta_train_df = pd.merge(final_df, df_day[['date', target_col_name]], left_on='ds', right_on='date').dropna()
    meta_features = [col for col in meta_train_df.columns if 'yhat_' in col]
    X_meta = meta_train_df[meta_features]
    y_meta = meta_train_df[target_col_name]
    
    if len(X_meta) < 10: return final_df.rename(columns={'yhat_prophet': 'yhat'})[['ds', 'yhat']]

    meta_model = RidgeCV(alphas=np.logspace(-3, 2, 10)).fit(X_meta, y_meta)
    
    future_dates = final_df[~final_df['ds'].isin(meta_train_df['ds'])]
    X_future_meta = future_dates[meta_features].dropna()
    
    if not X_future_meta.empty:
        future_predictions = meta_model.predict(X_future_meta)
        future_dates.loc[X_future_meta.index, 'yhat'] = future_predictions

    hist_predictions = meta_model.predict(X_meta)
    meta_train_df['yhat'] = hist_predictions

    return pd.concat([meta_train_df[['ds', 'yhat']], future_dates[['ds', 'yhat']]])

# --- NEW ORCHESTRATION FUNCTION ---
def run_day_specific_pipeline(historical_df, events_df, weather_df, target_col, periods, customer_forecasts=None):
    all_forecasts = []
    
    for day_of_week in range(7):
        day_name = date(2024, 1, 1+day_of_week).strftime('%A')
        st.write(f"Training models for {day_name}s...")

        # Filter data for the specific day
        df_day = historical_df[historical_df['date'].dt.dayofweek == day_of_week].copy()
        if len(df_day) < 20:
            st.warning(f"Skipping {day_name}s due to insufficient data ({len(df_day)} records).")
            continue

        # Base model parameters
        xgb_params = {'objective': 'reg:squarederror', 'random_state': 42, 'n_estimators': 100}
        lgbm_params = {'objective': 'regression_l1', 'random_state': 42, 'verbosity': -1, 'n_estimators': 100}
        cat_params = {'objective': 'RMSE', 'random_seed': 42, 'verbose': 0, 'iterations': 100}

        # Train base models for the specific day
        prophet_f, _ = train_and_forecast_prophet_day_specific(historical_df, events_df, periods, target_col, day_of_week)
        xgb_f = train_and_forecast_tree_day_specific(xgb.XGBRegressor, xgb_params, historical_df, periods, target_col, day_of_week, customer_forecasts)
        lgbm_f = train_and_forecast_tree_day_specific(lgb.LGBMRegressor, lgbm_params, historical_df, periods, target_col, day_of_week, customer_forecasts)
        cat_f = train_and_forecast_tree_day_specific(cat.CatBoostRegressor, cat_params, historical_df, periods, target_col, day_of_week, customer_forecasts)

        # Stack the models for the specific day
        base_forecasts = {"prophet": prophet_f, "xgb": xgb_f, "lgbm": lgbm_f, "cat": cat_f}
        day_stacked_f = train_and_forecast_stacked_ensemble_day_specific(base_forecasts, historical_df, target_col, day_of_week)
        all_forecasts.append(day_stacked_f)

    if not all_forecasts:
        return pd.DataFrame()
        
    # Combine forecasts from all day-specific models
    final_forecast = pd.concat(all_forecasts).sort_values('ds').reset_index(drop=True)
    return final_forecast


def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist, fcst, metrics, target):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')))
    title_text=f"{target.replace('_',' ').title()} Forecast"
    y_axis_title=title_text+' (₱)' if 'atv' in target or 'sales' in target else title_text
    
    fig.update_layout(
        title=f'Full Diagnostic: {title_text} vs. Historical',
        xaxis=dict(title='Date'),
        yaxis=dict(title=y_axis_title),
        legend=dict(x=0.01,y=0.99),
        height=500,
        margin=dict(l=40,r=40,t=60,b=40),
        paper_bgcolor='#2a2a2a',
        plot_bgcolor='#2a2a2a',
        font_color='white'
    )
    fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)")
    return fig

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']):
        x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'daily' in day_data and pd.notna(day_data['daily']):
        x_data.append('Time of Day Effect');y_data.append(day_data['daily']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']):
        x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']):
        holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"
        x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')

    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty or actual_col not in df.columns or forecast_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white',
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{"text": "No data available for this period.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
        )
        return fig

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual',
        line=dict(color='#3b82f6', width=2), marker=dict(symbol='circle', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast',
        line=dict(color='#d62728', dash='dash', width=2), marker=dict(symbol='x', size=7)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=20)),
        xaxis=dict(title='Date', font=dict(color='white', size=14)),
        yaxis=dict(title=y_axis_title, font=dict(color='white', size=14)),
        legend=dict(font=dict(color='white', size=12)),
        height=450, margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#444', tickfont=dict(color='white', size=12))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#444', tickfont=dict(color='white', size=12))

    return fig

def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data.get('weekly', 0),'Time of Year':day_data.get('yearly', 0),'Holidays/Events':day_data.get('holidays', 0)}
    significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data.get('trend', 0):.0f} customers**.\n\n"
    if not significant_effects:
        summary += f"The final forecast of **{day_data.get('yhat', 0):.0f} customers** is driven primarily by this trend."
        return summary
    pos_drivers={k:v for k,v in significant_effects.items()if v>0};neg_drivers={k:v for k,v in significant_effects.items()if v<0}
    if pos_drivers:biggest_pos_driver=max(pos_drivers,key=pos_drivers.get);summary+=f"📈 Main positive driver is **{biggest_pos_driver}**,adding an estimated **{pos_drivers[biggest_pos_driver]:.0f} customers**.\n"
    if neg_drivers:biggest_neg_driver=min(neg_drivers,key=neg_drivers.get);summary+=f"📉 Main negative driver is **{biggest_neg_driver}**,reducing by **{abs(neg_drivers[biggest_neg_driver]):.0f} customers**.\n"
    summary+=f"\nAfter all factors,the final forecast is **{day_data.get('yhat', 0):.0f} customers**.";return summary

def render_activity_card(row, db_client, view_type='compact_list'):
    doc_id = row['doc_id']
    
    if view_type == 'compact_list':
        date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y')
        summary_line = f"**{date_str}** | {row['activity_name']}"
        
        with st.expander(summary_line):
            status = row['remarks']
            if status == 'Confirmed': color = '#22C55E'
            elif status == 'Needs Follow-up': color = '#F59E0B'
            elif status == 'Tentative': color = '#38BDF8'
            else: color = '#EF4444'
            st.markdown(f"**Status:** <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            st.markdown(f"**Potential Sales:** ₱{row['potential_sales']:,.2f}")
            
            with st.form(key=f"compact_update_form_{doc_id}", border=False):
                status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]
                current_status_index = status_options.index(status) if status in status_options else 0
                updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"compact_sales_{doc_id}")
                updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"compact_remarks_{doc_id}")
                
                update_col, delete_col = st.columns(2)
                with update_col:
                    if st.form_submit_button("💾 Update", use_container_width=True):
                        update_data = {"potential_sales": updated_sales, "remarks": updated_remarks}
                        update_activity_in_firestore(db, doc_id, update_data)
                        st.success("Activity updated!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                with delete_col:
                    if st.form_submit_button("🗑️ Delete", use_container_width=True):
                        delete_from_firestore(db, 'future_activities', doc_id)
                        st.warning("Activity deleted.")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
    else: # 'grid' view
        with st.container(border=True):
            activity_date_formatted = pd.to_datetime(row['date']).strftime('%A, %B %d, %Y')
            
            st.markdown(f"**{row['activity_name']}**")
            st.markdown(f"<small>📅 {activity_date_formatted}</small>", unsafe_allow_html=True)
            st.markdown(f"💰 ₱{row['potential_sales']:,.2f}")
            status = row['remarks']
            if status == 'Confirmed': color = '#22C55E'
            elif status == 'Needs Follow-up': color = '#F59E0B'
            elif status == 'Tentative': color = '#38BDF8'
            else: color = '#EF4444'
            st.markdown(f"Status: <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            
            with st.expander("Edit / Manage"):
                status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]
                current_status_index = status_options.index(status) if status in status_options else 0
                with st.form(key=f"full_update_form_{doc_id}", border=False):
                    updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"full_sales_{doc_id}")
                    updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"full_remarks_{doc_id}")
                    
                    update_col, delete_col = st.columns(2)
                    with update_col:
                        if st.form_submit_button("💾 Update", use_container_width=True):
                            update_data = {"potential_sales": updated_sales, "remarks": updated_remarks}
                            update_activity_in_firestore(db, doc_id, update_data)
                            st.success("Activity updated!")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                    with delete_col:
                        if st.form_submit_button("🗑️ Delete", use_container_width=True):
                            delete_from_firestore(db, 'future_activities', doc_id)
                            st.warning("Activity deleted.")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()

def render_historical_record(row, db_client):
    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        st.write(f"**Weather:** {row.get('weather', 'N/A')}")
        
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.write("---")
            st.markdown("**Edit Record**")
            
            edit_cols = st.columns(2)
            updated_sales = edit_cols[0].number_input("Sales (₱)", value=float(row.get('sales', 0)), format="%.2f", key=f"sales_{row['doc_id']}")
            updated_customers = edit_cols[1].number_input("Customers", value=int(row.get('customers', 0)), key=f"cust_{row['doc_id']}")
            
            edit_cols2 = st.columns(2)
            updated_addons = edit_cols2[0].number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales', 0)), format="%.2f", key=f"addon_{row['doc_id']}")
            
            weather_options = ["Sunny", "Cloudy", "Rainy", "Storm"]
            current_weather_index = weather_options.index(row.get('weather')) if row.get('weather') in weather_options else 0
            updated_weather = edit_cols2[1].selectbox("Weather", options=weather_options, index=current_weather_index, key=f"weather_{row['doc_id']}")
            
            btn_cols = st.columns(2)
            if btn_cols[0].form_submit_button("💾 Update Record", use_container_width=True):
                update_data = {
                    'sales': updated_sales,
                    'customers': updated_customers,
                    'add_on_sales': updated_addons,
                    'weather': updated_weather,
                }
                update_historical_record_in_firestore(db_client, row['doc_id'], update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

            if btn_cols[1].form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                st.warning(f"Record for {date_str} deleted.")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
        st.info("Forecasting with Day-Specific AI Models")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning to get latest data.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                recalibrated = check_performance_and_recalibrate(db, st.session_state.historical_df)
                if recalibrated:
                    st.info("Models have been recalibrated. Please click 'Generate Forecast' again to use the new models.")
                else:
                    with st.spinner("🧠 Initializing Day-Specific AI Ensemble..."):
                        base_df = st.session_state.historical_df.copy()
                        FORECAST_HORIZON = 15
                        
                        with st.spinner("🛰️ Fetching live weather and engineering advanced features..."):
                            weather_df = get_weather_forecast(days=FORECAST_HORIZON + 30)
                            if weather_df is None:
                                st.warning("Could not fetch live weather data. Proceeding without it.")
                                weather_df = pd.DataFrame()
                            
                            base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                            capped_df, _, _ = cap_outliers_iqr(base_df, column='base_sales')
                            hist_df_with_atv = calculate_atv(capped_df)
                            hist_df_featured = create_advanced_features(hist_df_with_atv, weather_df)
                            ev_df = st.session_state.events_df.copy()

                        with st.spinner("Stage 1: Building Day-Specific Customer Models..."):
                            cust_f = run_day_specific_pipeline(hist_df_featured, ev_df, weather_df, 'customers', FORECAST_HORIZON)
                        
                        with st.spinner("Stage 2: Building Day-Specific ATV Models..."):
                            atv_f = run_day_specific_pipeline(hist_df_featured, ev_df, weather_df, 'atv', FORECAST_HORIZON, customer_forecasts=cust_f)
                        
                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            if not weather_df.empty:
                                combo_f = pd.merge(combo_f, weather_df[['date', 'weather_condition']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                                combo_f.rename(columns={'weather_condition': 'weather'}, inplace=True)
                            else:
                                combo_f['weather'] = 'Unavailable'

                            st.session_state.forecast_df = combo_f
                            
                            # Note: Prophet components are no longer generated from a single model
                            # The insight tab might need to be re-evaluated or simplified
                            st.session_state.forecast_components = pd.DataFrame()
                            
                            st.success("Day-Specific AI forecast generated successfully!")
                        else:
                            st.error("Forecast generation failed. Check data availability for each day of the week.")

        st.markdown("---")
        st.download_button(
            "📥 Download Forecast",
            convert_df_to_csv(st.session_state.forecast_df),
            "forecast_data.csv",
            "text/csv",
            use_container_width=True,
            disabled=st.session_state.forecast_df.empty
        )
        st.download_button(
            "📥 Download Historical",
            convert_df_to_csv(st.session_state.historical_df),
            "historical_data.csv",
            "text/csv",
            use_container_width=True
        )

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
                st.info("This chart shows how the final stacked model forecast compares against historical data. This is the ultimate measure of the model's performance on past data.");
                d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);
                hist_atv=calculate_atv(st.session_state.historical_df.copy())
                
                customer_plot_df = st.session_state.forecast_df.rename(columns={'forecast_customers': 'yhat'})
                atv_plot_df = st.session_state.forecast_df.rename(columns={'forecast_atv': 'yhat'})

                with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv, customer_plot_df, st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv, atv_plot_df, st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
        else:st.info("Click the 'Generate Forecast' button to begin.")
    
    with tabs[1]:
        st.header("💡 Forecast Insights")
        st.warning("Forecast breakdown is not available with the new Day-Specific Model architecture.")
        st.info("The new approach builds seven independent models (one for each day of the week) instead of a single model. While this significantly improves accuracy by comparing 'apples to apples' (e.g., Mondays to Mondays), it makes a single component breakdown chart impractical. The forecast is now a composite of these seven specialized models.")

    
    with tabs[2]:
        st.header("📈 Forecast Evaluator")
        st.info(
            "This report compares actual results against the forecast that was generated **the day before**. "
            "This provides a true measure of the model's day-ahead prediction accuracy."
        )

        def render_true_accuracy_content(days):
            try:
                log_docs = db.collection('forecast_log').stream()
                log_records = [doc.to_dict() for doc in log_docs]
                if not log_records:
                    st.warning("No forecast logs found. Please generate a forecast to begin logging.")
                    raise StopIteration

                forecast_log_df = pd.DataFrame(log_records)
                forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.tz_localize(None)
                forecast_log_df['generated_on'] = pd.to_datetime(forecast_log_df['generated_on']).dt.tz_localize(None)

                forecast_log_df = forecast_log_df[
                    forecast_log_df['forecast_for_date'] - forecast_log_df['generated_on'] == timedelta(days=1)
                ].copy()

                if forecast_log_df.empty:
                    st.warning("Not enough consecutive forecast logs to calculate true day-ahead accuracy.")
                    raise StopIteration

                historical_actuals_df = st.session_state.historical_df[['date', 'sales', 'customers', 'add_on_sales']].copy()
                true_accuracy_df = pd.merge(
                    historical_actuals_df, forecast_log_df,
                    left_on='date', right_on='forecast_for_date', how='inner'
                )

                if true_accuracy_df.empty:
                    st.warning("No matching historical data for the logged forecasts.")
                    raise StopIteration
                
                period_start_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=days)
                final_df = true_accuracy_df[true_accuracy_df['date'] >= period_start_date].copy()

                if final_df.empty:
                    st.warning(f"No forecast data in the last {days} days to evaluate.")
                    raise StopIteration

                st.subheader(f"Accuracy Metrics for the Last {days} Days")
                
                sales_mae = mean_absolute_error(final_df['sales'], final_df['predicted_sales'])
                cust_mae = mean_absolute_error(final_df['customers'], final_df['predicted_customers'])
                
                actual_sales_safe = final_df['sales'].replace(0, np.nan)
                sales_mape = np.nanmean(np.abs((final_df['sales'] - final_df['predicted_sales']) / actual_sales_safe)) * 100
                
                actual_cust_safe = final_df['customers'].replace(0, np.nan)
                cust_mape = np.nanmean(np.abs((final_df['customers'] - final_df['predicted_customers']) / actual_cust_safe)) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Sales MAPE (Accuracy)", value=f"{100 - sales_mape:.2f}%")
                    st.metric(label="Sales MAE (Avg Error)", value=f"₱{sales_mae:,.2f}")
                with col2:
                    st.metric(label="Customer MAPE (Accuracy)", value=f"{100 - cust_mape:.2f}%")
                    st.metric(label="Customer MAE (Avg Error)", value=f"{cust_mae:,.0f} customers")

                st.markdown("---")
                st.subheader(f"Comparison Charts for the Last {days} Days")
                
                sales_fig = plot_evaluation_graph(
                    final_df, date_col='date', actual_col='sales', forecast_col='predicted_sales',
                    title='Actual Sales vs. Day-Ahead Forecasted Sales', y_axis_title='Sales (₱)'
                )
                st.plotly_chart(sales_fig, use_container_width=True)
                
                cust_fig = plot_evaluation_graph(
                    final_df, date_col='date', actual_col='customers', forecast_col='predicted_customers',
                    title='Actual Customers vs. Day-Ahead Forecasted Customers', y_axis_title='Customers'
                )
                st.plotly_chart(cust_fig, use_container_width=True)

            except StopIteration:
                pass 
            except Exception as e:
                st.error(f"An error occurred while building the report: {e}")

        eval_tab_7, eval_tab_30 = st.tabs(["Last 7 Days", "Last 30 Days"])
        with eval_tab_7:
            render_true_accuracy_content(7)
        with eval_tab_30:
            render_true_accuracy_content(30)

    with tabs[3]:
        form_col, display_col = st.columns([2, 3], gap="large")

        with form_col:
            st.subheader("✍️ Add New Daily Record")
            with st.form("new_record_form",clear_on_submit=True, border=False):
                new_date=st.date_input("Date", date.today())
                new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                new_customers=st.number_input("Customer Count",min_value=0)
                new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"],help="Describe general weather.")
                
                if st.form_submit_button("✅ Save Record"):
                    new_rec={
                        "date":new_date,
                        "sales":new_sales,
                        "customers":new_customers,
                        "weather":new_weather,
                        "add_on_sales":new_addons,
                    }
                    add_to_firestore(db,'historical_data',new_rec, st.session_state.historical_df)
                    st.cache_data.clear()
                    st.success("Record added to Firestore!");
                    time.sleep(1)
                    st.rerun()
        
        with display_col:
            if st.button("🗓️ Show/Hide Recent Entries"):
                st.session_state.show_recent_entries = not st.session_state.show_recent_entries
            
            if st.session_state.show_recent_entries:
                st.subheader("🗓️ Recent Entries")
                with st.container(border=True):
                    recent_df = st.session_state.historical_df.copy().sort_values(by="date", ascending=False).head(10)
                    if not recent_df.empty:
                        display_cols = ['date', 'sales', 'customers', 'add_on_sales', 'weather']
                        cols_to_show = [col for col in display_cols if col in recent_df.columns]
                        
                        st.dataframe(
                            recent_df[cols_to_show],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                "sales": st.column_config.NumberColumn("Sales (₱)", format="₱%.2f"),
                                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                                "add_on_sales": st.column_config.NumberColumn("Add-on Sales (₱)", format="₱%.2f"),
                                "weather": "Weather",
                            }
                        )
                    else:
                        st.info("No recent data to display.")
    
    with tabs[4]:
        def set_view_all(): st.session_state.show_all_activities = True
        def set_overview(): st.session_state.show_all_activities = False

        if st.session_state.get('show_all_activities'):
            st.markdown("#### All Upcoming Activities")
            st.button("⬅️ Back to Overview", on_click=set_overview)
            
            activities_df = st.session_state.events_df.copy()
            all_upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy()
            
            if all_upcoming_df.empty:
                st.info("No upcoming activities scheduled.")
            else:
                all_upcoming_df['month_year'] = all_upcoming_df['date'].dt.strftime('%B %Y')
                sorted_months_df = all_upcoming_df.sort_values('date')
                month_tabs_list = sorted_months_df['month_year'].unique().tolist()
                
                if month_tabs_list:
                    month_tabs = st.tabs(month_tabs_list)
                    for i, tab in enumerate(month_tabs):
                        with tab:
                            month_name = month_tabs_list[i]
                            month_df = sorted_months_df[sorted_months_df['month_year'] == month_name]
                            
                            header_cols = st.columns([2, 1, 1])
                            with header_cols[1]:
                                total_sales = month_df['potential_sales'].sum()
                                st.metric(label="Total Expected Sales", value=f"₱{total_sales:,.2f}")
                            with header_cols[2]:
                                unconfirmed_count = len(month_df[month_df['remarks'] != 'Confirmed'])
                                st.metric(label="Unconfirmed Activities", value=unconfirmed_count)
                            st.markdown("---")

                            activities = month_df.to_dict('records')
                            for i in range(0, len(activities), 4):
                                cols = st.columns(4)
                                row_activities = activities[i:i+4]
                                for j, activity in enumerate(row_activities):
                                    with cols[j]:
                                        render_activity_card(activity, db, view_type='grid')

        else:
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                st.markdown("##### Add New Activity")
                with st.form("new_activity_form", clear_on_submit=True, border=True):
                    activity_name = st.text_input("Activity/Event Name", placeholder="e.g., Catering for Birthday")
                    activity_date = st.date_input("Date of Activity", min_value=date.today())
                    potential_sales = st.number_input("Potential Sales (₱)", min_value=0.0, format="%.2f")
                    remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
                    
                    submitted = st.form_submit_button("✅ Save Activity", use_container_width=True)
                    if submitted:
                        if activity_name and activity_date:
                            new_activity = {
                                "activity_name": activity_name,
                                "date": pd.to_datetime(activity_date).to_pydatetime(),
                                "potential_sales": float(potential_sales),
                                "remarks": remarks
                            }
                            db.collection('future_activities').add(new_activity)
                            st.success(f"Activity '{activity_name}' saved!")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Activity name and date are required.")

            with col2:
                st.markdown("##### Next 10 Upcoming Activities")
                
                btn_cols = st.columns(2)
                btn_cols[0].button("🔄 Refresh List", key='refresh_activities', use_container_width=True, on_click=st.rerun)
                btn_cols[1].button("📂 View All Upcoming Activities", use_container_width=True, on_click=set_view_all)
                
                st.markdown("---",)
                activities_df = st.session_state.events_df.copy()
                upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy().head(10)
                
                if upcoming_df.empty:
                    st.info("No upcoming activities scheduled.")
                else:
                    for _, row in upcoming_df.iterrows():
                        render_activity_card(row, db, view_type='compact_list')


    with tabs[5]:
        st.subheader("View & Edit Historical Data")
        df = st.session_state.historical_df.copy()
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)

            all_years = sorted(df['date'].dt.year.unique(), reverse=True)

            if all_years:
                filter_cols = st.columns(2)
                with filter_cols[0]:
                    selected_year = st.selectbox("Select Year to View:", options=all_years)

                df_year_filtered = df[df['date'].dt.year == selected_year]
                
                all_months = sorted(df_year_filtered['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)

                if all_months:
                    with filter_cols[1]:
                        selected_month_str = st.selectbox("Select Month to View:", options=all_months)

                    selected_month_num = pd.to_datetime(selected_month_str, format='%B').month

                    filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].copy()

                    if filtered_df.empty:
                        st.info("No data for the selected month and year.")
                    else:
                        df_first_half = filtered_df[filtered_df['date'].dt.day <= 15]
                        df_second_half = filtered_df[filtered_df['date'].dt.day > 15]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("##### Days 1-15")
                            st.markdown("---")
                            for index, row in df_first_half.iterrows():
                                render_historical_record(row, db)
                        
                        with col2:
                            st.markdown("##### Days 16-31")
                            st.markdown("---")
                            for index, row in df_second_half.iterrows():
                                render_historical_record(row, db)
                else:
                    st.write(f"No data available for the year {selected_year}.")
            else:
                st.write("No historical data to display.")
        else:
            st.write("No historical data to display.")
