import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from xgboost.callback import EarlyStopping
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
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
import shap
import matplotlib.pyplot as plt
import inspect
from fredapi import Fred # --- NEW: Import for economic data

# --- Suppress informational messages ---
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
        .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); /* Red gradient */
            color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button {
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
        'shap_explainer_cust': None,
        'shap_values_cust': None,
        'X_cust': None,
        'X_cust_dates': None,
        'future_X_cust': None,
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
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
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
    
# --- NEW: Function to get economic data from FRED ---
@st.cache_data(ttl="24h")
def get_economic_data(start_date, end_date):
    """
    Fetches Philippine CPI data from FRED and prepares it for modeling.
    Requires a FRED_API_KEY in st.secrets.
    """
    try:
        api_key = st.secrets.fred_api.key
        fred = Fred(api_key=api_key)
        # FPCPITOTLZGPHL is the series ID for Consumer Price Index for the Philippines
        cpi_data = fred.get_series('FPCPITOTLZGPHL', observation_start=start_date)
        if cpi_data.empty:
            st.warning("No economic data returned from FRED API.")
            return pd.DataFrame()

        df = pd.DataFrame({'cpi': cpi_data})
        df.index.name = 'date'
        df = df.resample('D').ffill().reset_index() # Resample monthly data to daily
        df['date'] = pd.to_datetime(df['date'])
        return df

    except Exception as e:
        st.warning(f"Could not fetch economic data. Forecasting will proceed without it. Error: {e}")
        return pd.DataFrame()

# --- Core Forecasting Models ---

def create_advanced_features(df, economic_df):
    """IMPROVED: Now merges economic data if available."""
    df['date'] = pd.to_datetime(df['date'])
    
    # Merge economic data
    if not economic_df.empty:
        df = pd.merge(df, economic_df, on='date', how='left')
        df['cpi'].ffill(inplace=True) # Forward fill any missing daily values
        df['cpi'].bfill(inplace=True) # Backfill for any initial missing values
    else:
        df['cpi'] = 0 # If no data, use a neutral value

    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    df = df.sort_values('date')
    
    if 'sales' in df.columns:
        df['sales_lag_7'] = df['sales'].shift(7)
    if 'customers' in df.columns:
        df['customers_lag_7'] = df['customers'].shift(7)
    
    if 'sales' in df.columns:
        df['sales_rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
        df['sales_rolling_std_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).std()
    if 'customers' in df.columns:
        df['customers_rolling_mean_7'] = df['customers'].shift(1).rolling(window=7, min_periods=1).mean()

    if 'atv' in df.columns:
        df['atv_lag_7'] = df['atv'].shift(7)
        df['atv_rolling_mean_7'] = df['atv'].shift(1).rolling(window=7, min_periods=1).mean()
        df['atv_rolling_std_7'] = df['atv'].shift(1).rolling(window=7, min_periods=1).std()
        
    if 'sales' in df.columns:
        df['sales_ewm_7'] = df['sales'].shift(1).ewm(span=7, adjust=False).mean()
    if 'customers' in df.columns:
        df['customers_ewm_7'] = df['customers'].shift(1).ewm(span=7, adjust=False).mean()
    if 'atv' in df.columns:
        df['atv_ewm_7'] = df['atv'].shift(1).ewm(span=7, adjust=False).mean()
        
    return df

@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    
    if df_train.empty or len(df_train) < 15:
        return pd.DataFrame(), None

    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)

    manual_events_renamed = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_manual_events = pd.concat([manual_events_renamed, recurring_events])
    all_manual_events.dropna(subset=['ds', 'holiday'], inplace=True)

    if 'day_type' in historical_df.columns:
        not_normal_days_df = historical_df[historical_df['day_type'] == 'Not Normal Day'].copy()
        if not not_normal_days_df.empty:
            not_normal_events = pd.DataFrame({
                'holiday': 'Unusual_Day',
                'ds': pd.to_datetime(not_normal_days_df['date']),
                'lower_window': 0,
                'upper_window': 0,
            })
            all_manual_events = pd.concat([all_manual_events, not_normal_events])
    
    use_yearly_seasonality = len(df_train) >= 365

    prophet_model = Prophet(
        growth='linear',
        holidays=all_manual_events,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=use_yearly_seasonality, 
        changepoint_prior_scale=0.5, 
        changepoint_range=0.95,
    )
    
    # IMPROVED: Add CPI as an external regressor if available
    if 'cpi' in df_prophet.columns:
        prophet_model.add_regressor('cpi', standardize=True, mode='multiplicative')

    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    
    # IMPROVED: Add future CPI values to the future dataframe
    if 'cpi' in df_prophet.columns:
        future = pd.merge(future, df_prophet[['ds', 'cpi']], on='ds', how='left')
        future['cpi'].ffill(inplace=True)

    forecast = prophet_model.predict(future)
    
    return forecast[['ds', 'yhat']], prophet_model

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, events_df, periods, target_col, atv_forecast_df=None):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)

    if df_train.empty:
        return pd.DataFrame(), None, pd.DataFrame(), None, None
        
    df_featured = create_advanced_features(df_train.copy(), pd.DataFrame()) # Pass empty df for now

    if 'day_type' in df_featured.columns:
        df_featured['is_not_normal_day'] = df_featured['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0)
    else:
        df_featured['is_not_normal_day'] = 0

    base_features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'is_not_normal_day', 'cpi']

    if target_col == 'customers':
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7', 'customers_ewm_7', 'sales_ewm_7']
    elif target_col == 'atv':
        features = base_features + ['atv_lag_7', 'atv_rolling_mean_7', 'atv_rolling_std_7', 'atv_ewm_7']
    else: 
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7', 'sales_ewm_7', 'customers_ewm_7']

    features = [f for f in features if f in df_featured.columns]
    target = target_col
    
    X = df_featured[features].copy()
    y = df_featured[target].copy()
    X.dropna(inplace=True)
    y = y[X.index]
    
    X_dates = df_featured.loc[X.index, 'date']

    if X.empty or len(X) < 50:
        st.warning(f"Not enough data to train XGBoost for {target_col} after feature engineering.")
        return pd.DataFrame(), None, pd.DataFrame(), None, None
    
    fit_params = inspect.signature(xgb.XGBRegressor.fit).parameters
    use_callbacks = 'callbacks' in fit_params
    use_early_stopping_rounds = 'early_stopping_rounds' in fit_params
    
    def objective(trial):
        cv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in cv.split(X):
            if len(train_idx) < 10 or len(val_idx) < 10:
                continue 

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
            param = {
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'random_state': 42
            }
            model = xgb.XGBRegressor(**param)
            sample_weights = np.exp(np.linspace(-1, 0, len(y_train)))
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
                "sample_weight": sample_weights
            }
            if use_callbacks:
                fit_kwargs["callbacks"] = [EarlyStopping(rounds=50, save_best=True)]
            elif use_early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = 50
            
            model.fit(X_train, y_train, **fit_kwargs)
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds, squared=False))

        return np.mean(scores) if scores else float('inf')

    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25, timeout=300)
        best_params = study.best_params
    except Exception as e:
        st.warning(f"Optuna tuning failed for {target_col}. Using default parameters. Error: {e}")
        best_params = {
            'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
            'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
        }

    final_model = xgb.XGBRegressor(**best_params)
    sample_weights = np.exp(np.linspace(-1, 0, len(y)))
    final_model.fit(X, y, sample_weight=sample_weights)

    future_predictions = []
    future_feature_sets = {}
    history_with_features = df_featured.copy()
    atv_lookup = atv_forecast_df.set_index('ds')['yhat'] if atv_forecast_df is not None and not atv_forecast_df.empty else None

    for i in range(periods):
        last_date = history_with_features['date'].max()
        next_date = last_date + timedelta(days=1)
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history_with_features, future_step_df], ignore_index=True)
        extended_featured_df = create_advanced_features(extended_history, pd.DataFrame())
        extended_featured_df['is_not_normal_day'] = extended_featured_df['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0)
        X_future = extended_featured_df[features].tail(1)
        
        prediction = final_model.predict(X_future)[0]
        future_predictions.append({'ds': next_date, 'yhat': prediction})
        
        future_feature_sets[next_date.strftime('%Y-%m-%d')] = X_future
        
        history_with_features = extended_featured_df.copy()
        history_with_features.loc[history_with_features.index.max(), target] = prediction
        if target_col == 'customers' and atv_lookup is not None:
            forecasted_atv = atv_lookup.get(pd.to_datetime(next_date), history_with_features['atv'].dropna().tail(7).mean())
            history_with_features.loc[history_with_features.index.max(), 'sales'] = prediction * forecasted_atv

    final_df = pd.DataFrame(future_predictions)
    historical_predictions = df_featured[['date']].copy()
    historical_predictions = historical_predictions.loc[X.index] 
    historical_predictions.rename(columns={'date': 'ds'}, inplace=True)
    historical_predictions['yhat'] = final_model.predict(X)

    full_forecast_df = pd.concat([historical_predictions, final_df], ignore_index=True)
    return full_forecast_df, final_model, X, X_dates, future_feature_sets

# --- NEW: Functions for LightGBM and CatBoost Models ---
@st.cache_resource
def train_and_forecast_lightgbm_tuned(historical_df, events_df, periods, target_col):
    # This function mirrors the XGBoost one, adapted for LightGBM
    # (Implementation is omitted for brevity but would be structured identically)
    return pd.DataFrame(), None # Placeholder

@st.cache_resource
def train_and_forecast_catboost_tuned(historical_df, events_df, periods, target_col):
    # This function mirrors the XGBoost one, adapted for CatBoost
    # (Implementation is omitted for brevity but would be structured identically)
    return pd.DataFrame(), None # Placeholder

# --- IMPROVED: Advanced Stacking Ensemble ---
@st.cache_resource
def train_and_forecast_stacked_ensemble(base_forecasts_dict, historical_target, target_col_name):
    # Merge all base model forecasts
    final_df = None
    for name, fcst_df in base_forecasts_dict.items():
        if fcst_df.empty:
            continue
        renamed_df = fcst_df.rename(columns={'yhat': f'yhat_{name}'})
        if final_df is None:
            final_df = renamed_df
        else:
            final_df = pd.merge(final_df, renamed_df, on='ds', how='outer')
    
    if final_df is None or final_df.empty:
        st.error("All base models failed to produce forecasts.")
        return pd.DataFrame()

    final_df.interpolate(inplace=True) # Handle potential misaligned dates

    training_data = pd.merge(final_df, historical_target[['date', target_col_name]], left_on='ds', right_on='date')
    
    meta_features = [col for col in training_data.columns if 'yhat_' in col]
    X_meta = training_data[meta_features]
    y_meta = training_data[target_col_name]
    
    if len(X_meta) < 20:
        st.warning(f"Not enough historical data for advanced stacking. Falling back to simple averaging.")
        final_df['yhat'] = X_meta.mean(axis=1)
        return final_df[['ds', 'yhat']]

    # IMPROVED: Use LightGBM as the meta-learner instead of simple Ridge
    meta_model = lgb.LGBMRegressor(random_state=42, n_estimators=200)
    meta_model.fit(X_meta, y_meta)
    
    X_future_meta = final_df[meta_features]
    stacked_prediction = meta_model.predict(X_future_meta)
    
    forecast_df = final_df[['ds']].copy()
    forecast_df['yhat'] = stacked_prediction
    
    return forecast_df


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
        
    # Add regressor effects if they exist
    regressor_effects = day_data.get('extra_regressors_multiplicative', 0)
    if 'cpi' in components.columns and day_data['cpi'] != 0:
        x_data.append('Economic Factors (CPI)');y_data.append(day_data['cpi']);measure_data.append('relative')

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

# ... (The rest of the helper and UI functions: generate_insight_summary, render_activity_card, etc. remain unchanged) ...
# ... They are omitted here for brevity but should be included in your final file. ...

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
        st.info("Forecasting with an Advanced AI Ensemble")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning to get latest data.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                with st.spinner("🧠 Initializing Advanced Ensemble Forecast..."):
                    base_df = st.session_state.historical_df.copy()
                    
                    # --- Data Preparation ---
                    with st.spinner("Fetching dynamic economic data..."):
                        start_date = base_df['date'].min() - pd.DateOffset(months=1)
                        end_date = base_df['date'].max() + pd.DateOffset(days=15)
                        economic_df = get_economic_data(start_date, end_date)

                    with st.spinner("Engineering features and capping outliers..."):
                        base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                        capped_df, capped_count, upper_bound = cap_outliers_iqr(base_df, column='base_sales')
                        if capped_count > 0:
                            st.warning(f"Capped {capped_count} outlier day(s) with base sales over ₱{upper_bound:,.2f}.")

                        hist_df_with_atv = calculate_atv(capped_df)
                        hist_df_featured = create_advanced_features(hist_df_with_atv, economic_df)
                        ev_df = st.session_state.events_df.copy()
                    
                    FORECAST_HORIZON = 15
                    
                    # --- Base Model Training ---
                    with st.spinner("Training Prophet Model..."):
                        prophet_atv_f, _ = train_and_forecast_prophet(hist_df_featured, ev_df, FORECAST_HORIZON, 'atv')
                        prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_featured, ev_df, FORECAST_HORIZON, 'customers')
                    
                    with st.spinner("Tuning & Training XGBoost Model..."):
                        xgb_atv_f, _, _, _, _ = train_and_forecast_xgboost_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'atv')
                        xgb_cust_f, xgb_cust_model, X_cust, X_cust_dates, future_X_cust = train_and_forecast_xgboost_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'customers', atv_forecast_df=prophet_atv_f)

                    # --- Placeholder calls for new models ---
                    lgbm_atv_f, _ = train_and_forecast_lightgbm_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'atv')
                    lgbm_cust_f, _ = train_and_forecast_lightgbm_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'customers')
                    cat_atv_f, _ = train_and_forecast_catboost_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'atv')
                    cat_cust_f, _ = train_and_forecast_catboost_tuned(hist_df_featured, ev_df, FORECAST_HORIZON, 'customers')
                    
                    # --- Stacking and Final Prediction ---
                    atv_f = pd.DataFrame()
                    with st.spinner("Stacking ATV models for final prediction..."):
                        base_atv_forecasts = {"prophet": prophet_atv_f, "xgb": xgb_atv_f, "lgbm": lgbm_atv_f, "cat": cat_atv_f}
                        atv_f = train_and_forecast_stacked_ensemble(base_atv_forecasts, hist_df_featured, 'atv')
                    
                    cust_f = pd.DataFrame()
                    with st.spinner("Stacking Customer models and preparing explanations..."):
                        base_cust_forecasts = {"prophet": prophet_cust_f, "xgb": xgb_cust_f, "lgbm": lgbm_cust_f, "cat": cat_cust_f}
                        cust_f = train_and_forecast_stacked_ensemble(base_cust_forecasts, hist_df_featured, 'customers')
                        
                        if xgb_cust_model and not X_cust.empty:
                            with st.spinner("Calculating SHAP values for XGBoost component..."):
                                explainer = shap.TreeExplainer(xgb_cust_model, feature_perturbation="tree_path_dependent")
                                try:
                                    shap_values = explainer(X_cust)
                                except shap.utils.ExplainerError:
                                    st.warning("SHAP additivity check failed. Calculating values without it.")
                                    shap_values = explainer(X_cust, check_additivity=False)
                                st.session_state.shap_explainer_cust = explainer
                                st.session_state.shap_values_cust = shap_values
                                st.session_state.X_cust = X_cust
                                st.session_state.X_cust_dates = X_cust_dates
                                st.session_state.future_X_cust = future_X_cust

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
                        
                        if prophet_model_cust:
                            full_future = prophet_model_cust.make_future_dataframe(periods=FORECAST_HORIZON)
                            if 'cpi' in hist_df_featured.columns:
                                full_future = pd.merge(full_future, hist_df_featured[['date', 'cpi']].rename(columns={'date':'ds'}), on='ds', how='left')
                                full_future['cpi'].ffill(inplace=True)
                            prophet_forecast_components = prophet_model_cust.predict(full_future)
                            st.session_state.forecast_components = prophet_forecast_components
                            st.session_state.all_holidays = prophet_model_cust.holidays
                        
                        st.success("Advanced AI forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed. One or more components could not be built.")


        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    # The rest of your UI tabs (Dashboard, Insights, Evaluator, etc.) remain here...
    # They are omitted for brevity.
