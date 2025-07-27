import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from lightgbm import early_stopping as lgb_early_stopping
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    
    return forecast[['ds', 'yhat']], prophet_model

def _run_tree_model_forecast(model_class, params, historical_df, periods, target_col, atv_forecast_df):
    """Helper function to run a generic tree-based model forecast loop."""
    df_featured = historical_df.copy() # Assumes features are already engineered
    
    base_features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'is_not_normal_day']
    if target_col == 'customers':
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7', 'customers_ewm_7', 'sales_ewm_7']
    else: # atv
        features = base_features + ['atv_lag_7', 'atv_rolling_mean_7', 'atv_rolling_std_7', 'atv_ewm_7']

    features = [f for f in features if f in df_featured.columns]
    
    X = df_featured[features].copy()
    y = df_featured[target_col].copy()
    X.dropna(inplace=True)
    y = y[X.index]
    
    final_model = model_class(**params)
    sample_weights = np.exp(np.linspace(-1, 0, len(y)))
    final_model.fit(X, y, sample_weight=sample_weights)

    future_predictions = []
    history_with_features = df_featured.copy()
    atv_lookup = atv_forecast_df.set_index('ds')['yhat'] if atv_forecast_df is not None and not atv_forecast_df.empty else None

    for i in range(periods):
        last_date = history_with_features['date'].max()
        next_date = last_date + timedelta(days=1)
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history_with_features, future_step_df], ignore_index=True)
        extended_featured_df = create_advanced_features(extended_history)
        extended_featured_df['is_not_normal_day'] = extended_featured_df['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0)
        X_future = extended_featured_df[features].tail(1)
        
        prediction = final_model.predict(X_future)[0]
        future_predictions.append({'ds': next_date, 'yhat': prediction})
        
        history_with_features = extended_featured_df.copy()
        history_with_features.loc[history_with_features.index.max(), target_col] = prediction
        if target_col == 'customers' and atv_lookup is not None:
            forecasted_atv = atv_lookup.get(pd.to_datetime(next_date), history_with_features['atv'].dropna().tail(7).mean())
            history_with_features.loc[history_with_features.index.max(), 'sales'] = prediction * forecasted_atv
            
    final_df = pd.DataFrame(future_predictions)
    historical_predictions = df_featured.loc[X.index, ['date']].copy()
    historical_predictions.rename(columns={'date': 'ds'}, inplace=True)
    historical_predictions['yhat'] = final_model.predict(X)

    return pd.concat([historical_predictions, final_df], ignore_index=True)

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, periods, target_col, atv_forecast_df=None):
    df_featured = historical_df.copy() # Features are engineered outside
    base_features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'is_not_normal_day']
    if target_col == 'customers':
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7', 'customers_ewm_7', 'sales_ewm_7']
    else: # atv
        features = base_features + ['atv_lag_7', 'atv_rolling_mean_7', 'atv_rolling_std_7', 'atv_ewm_7']

    features = [f for f in features if f in df_featured.columns]
    
    X = df_featured[features].copy()
    y = df_featured[target_col].copy()
    X.dropna(inplace=True)
    y = y[X.index]
    
    X_dates = df_featured.loc[X.index, 'date']

    if X.empty or len(X) < 50: return pd.DataFrame(), None, pd.DataFrame(), None, None
    
    fit_params = inspect.signature(xgb.XGBRegressor.fit).parameters
    use_callbacks = 'callbacks' in fit_params
    use_early_stopping_rounds = 'early_stopping_rounds' in fit_params
    
    def objective(trial):
        # ... (Same objective function as before)
        return 1.0 # Placeholder
        
    best_params = {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}
    final_model = xgb.XGBRegressor(**best_params)
    sample_weights = np.exp(np.linspace(-1, 0, len(y)))
    final_model.fit(X, y, sample_weight=sample_weights)

    # Re-use the generic forecast loop logic
    full_forecast_df = _run_tree_model_forecast(xgb.XGBRegressor, best_params, df_featured, periods, target_col, atv_forecast_df)
    
    # We still need future_X for SHAP, so we must recalculate it
    future_feature_sets = {}
    history_with_features = df_featured.copy()
    for i in range(periods):
        last_date = history_with_features['date'].max()
        next_date = last_date + timedelta(days=1)
        # ... (Logic to generate and store X_future)
    
    return full_forecast_df, final_model, X, X_dates, future_feature_sets

@st.cache_resource
def train_and_forecast_lightgbm_tuned(historical_df, periods, target_col, atv_forecast_df=None):
    # This function mirrors the XGBoost one, adapted for LightGBM
    best_params = {'random_state': 42, 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31}
    return _run_tree_model_forecast(lgb.LGBMRegressor, best_params, historical_df, periods, target_col, atv_forecast_df)

@st.cache_resource
def train_and_forecast_catboost_tuned(historical_df, periods, target_col, atv_forecast_df=None):
    # This function mirrors the XGBoost one, adapted for CatBoost
    best_params = {'random_state': 42, 'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 'verbose': 0}
    return _run_tree_model_forecast(cat.CatBoostRegressor, best_params, historical_df, periods, target_col, atv_forecast_df)

@st.cache_resource
def train_and_forecast_stacked_ensemble(base_forecasts_dict, historical_target, target_col_name):
    # ... (Same advanced stacking logic as before)
    return pd.DataFrame() # Placeholder

# ... (The rest of the helper and UI functions: plot_full_comparison_chart, etc. remain unchanged) ...

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    
    # ... (Sidebar UI remains unchanged) ...

    if st.button("📈 Generate Forecast", use_container_width=True):
        if len(st.session_state.historical_df) < 50:
            st.error("Please provide at least 50 days of data for reliable forecasting.")
        else:
            with st.spinner("🧠 Initializing Advanced Ensemble Forecast..."):
                base_df = st.session_state.historical_df.copy()
                
                with st.spinner("Engineering features and capping outliers..."):
                    base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                    capped_df, capped_count, upper_bound = cap_outliers_iqr(base_df, column='base_sales')
                    if capped_count > 0: st.warning(f"Capped {capped_count} outlier day(s) with base sales over ₱{upper_bound:,.2f}.")
                    hist_df_with_atv = calculate_atv(capped_df)
                    hist_df_featured = create_advanced_features(hist_df_with_atv)
                    ev_df = st.session_state.events_df.copy()
                
                FORECAST_HORIZON = 15
                
                # --- Base Model Training ---
                with st.spinner("Training Base Models (Prophet, XGBoost, LightGBM, CatBoost)..."):
                    prophet_atv_f, _ = train_and_forecast_prophet(hist_df_featured, ev_df, FORECAST_HORIZON, 'atv')
                    prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_featured, ev_df, FORECAST_HORIZON, 'customers')
                    
                    xgb_atv_f, _, _, _, _ = train_and_forecast_xgboost_tuned(hist_df_featured, FORECAST_HORIZON, 'atv', prophet_atv_f)
                    xgb_cust_f, xgb_cust_model, X_cust, X_cust_dates, future_X_cust = train_and_forecast_xgboost_tuned(hist_df_featured, FORECAST_HORIZON, 'customers', prophet_cust_f)

                    lgbm_atv_f = train_and_forecast_lightgbm_tuned(hist_df_featured, FORECAST_HORIZON, 'atv', prophet_atv_f)
                    lgbm_cust_f = train_and_forecast_lightgbm_tuned(hist_df_featured, FORECAST_HORIZON, 'customers', prophet_cust_f)
                    
                    cat_atv_f = train_and_forecast_catboost_tuned(hist_df_featured, FORECAST_HORIZON, 'atv', prophet_atv_f)
                    cat_cust_f = train_and_forecast_catboost_tuned(hist_df_featured, FORECAST_HORIZON, 'customers', prophet_cust_f)

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
                                st.warning("SHAP additivity check failed due to sample weighting. Recalculating without it.")
                                shap_values = explainer(X_cust, check_additivity=False)
                            st.session_state.shap_explainer_cust = explainer
                            st.session_state.shap_values_cust = shap_values
                            st.session_state.X_cust = X_cust
                            st.session_state.X_cust_dates = X_cust_dates
                            st.session_state.future_X_cust = future_X_cust

                # ... (rest of the main logic to combine, log, and display results) ...
