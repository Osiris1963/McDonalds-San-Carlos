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
        'shap_values_store': {}, # NEW: Store for SHAP values {day_of_week: values}
        'model_store': {}, # NEW: Store for trained models {day_of_week: model}
        'feature_store': {}, # NEW: Store for features {day_of_week: features}
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

# --- RE-ENGINEERED: Day-Specific Feature Creation ---
def create_dow_features(df, target_col):
    """
    Creates features for a dataframe containing data for only ONE day of the week.
    This is the core of the "apples-to-apples" analysis.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['time_index'] = range(len(df)) # Trend for this specific day
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Lag features (e.g., value from the previous Monday, 2 Mondays ago, etc.)
    for lag in [1, 2, 3, 4]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
    # Rolling window features (e.g., average of the last 4 Mondays)
    for window in [2, 3, 4, 6]:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()

    # Exponentially Weighted Mean
    df[f'{target_col}_ewm_4'] = df[target_col].shift(1).ewm(span=4, adjust=False).mean()

    # Add holiday/event features
    start_date = df['date'].min()
    end_date = df['date'].max() + timedelta(days=30) # Look ahead for events
    
    recurring_events = generate_recurring_local_events(start_date, end_date)
    recurring_events.rename(columns={'ds': 'date'}, inplace=True)
    
    # Mark paydays and near paydays
    df['is_payday'] = df['date'].isin(recurring_events[recurring_events['holiday'] == 'Payday']['date']).astype(int)
    df['is_near_payday'] = df['date'].isin(recurring_events[recurring_events['holiday'] == 'Near_Payday']['date']).astype(int)
    
    # Mark 'Not Normal Day'
    if 'day_type' in df.columns:
        df['is_not_normal_day'] = (df['day_type'] == 'Not Normal Day').astype(int)
    else:
        df['is_not_normal_day'] = 0
        
    return df

# --- RE-ENGINEERED: Day-Specific Forecasting Models ---
def train_and_forecast_tree_model_dow(model_class, model_params, df_dow, periods_to_forecast, target_col, is_catboost=False):
    """
    Trains a tree-based model on day-specific data and forecasts future occurrences of that day.
    """
    df_featured = create_dow_features(df_dow.copy(), target_col)
    
    features = [col for col in df_featured.columns if col not in ['date', 'doc_id', 'sales', 'customers', 'atv', 'base_sales', 'add_on_sales', 'day_type', 'day_type_notes', 'weather', target_col]]
    features = [f for f in features if f in df_featured.columns and not df_featured[f].isnull().all()]
    
    X = df_featured[features].copy()
    y = df_featured[target_col].copy()
    
    # Drop rows with NaNs created by lags/rolling features
    non_nan_idx = X.dropna().index
    X = X.loc[non_nan_idx]
    y = y.loc[non_nan_idx]

    if X.empty:
        return pd.DataFrame(), None, pd.DataFrame() # Not enough data

    # Train the final model on all available day-specific data
    final_model = model_class(**model_params)
    final_model.fit(X, y)

    # Iteratively forecast future dates
    future_predictions = []
    history_copy = df_featured.copy()

    for i in range(periods_to_forecast):
        last_date = history_copy['date'].max()
        next_date = last_date + timedelta(days=7)
        
        # Create a placeholder for the next date to generate features
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history_copy, future_step_df], ignore_index=True)
        
        # Re-create features for the extended history
        extended_featured_df = create_dow_features(extended_history, target_col)
        
        # Get the last row, which is our future feature set
        X_future = extended_featured_df[features].tail(1)
        
        prediction = final_model.predict(X_future)[0]
        future_predictions.append({'ds': next_date, 'yhat': prediction})
        
        # Add the prediction back to the history to be used for the next iteration's feature calculation
        history_copy.loc[history_copy.index.max(), target_col] = prediction

    forecast_df = pd.DataFrame(future_predictions)
    return forecast_df, final_model, X

# --- RE-ENGINEERED: Main Forecasting Pipeline Orchestrator ---
def generate_day_by_day_forecast(historical_df, events_df, forecast_horizon=15):
    """
    Orchestrates the new forecasting strategy.
    Loops through each day of the week, trains models, forecasts, and combines results.
    """
    st.session_state.model_store = {}
    st.session_state.shap_values_store = {}
    st.session_state.feature_store = {}

    # --- Step 1: Pre-process data ---
    base_df = historical_df.copy()
    base_df['base_sales'] = base_df['sales'] - base_df.get('add_on_sales', 0)
    capped_df, _, _ = cap_outliers_iqr(base_df, column='base_sales')
    hist_df_with_atv = calculate_atv(capped_df)

    all_forecasts_cust = []
    all_forecasts_atv = []

    # --- Step 2: Loop through each day of the week (0=Mon, 1=Tue, ..., 6=Sun) ---
    for dow in range(7):
        day_name = pd.to_datetime(dow, format='%w').strftime('%A')
        
        with st.spinner(f"Analyzing and forecasting for all future {day_name}s..."):
            # Filter data for the specific day of the week
            df_dow = hist_df_with_atv[hist_df_with_atv['date'].dt.dayofweek == dow].copy()
            
            if len(df_dow) < 15:
                st.warning(f"Skipping {day_name}s due to insufficient data (found {len(df_dow)}, need at least 15).")
                continue

            # Determine how many future occurrences of this day we need to forecast
            today = pd.to_datetime('today').normalize()
            days_until_next_dow = (dow - today.dayofweek + 7) % 7
            first_forecast_date = today + timedelta(days=days_until_next_dow)
            remaining_horizon = forecast_horizon - (first_forecast_date - today).days
            periods_to_forecast = (remaining_horizon // 7) + 1

            # --- Step 3: Forecast Customers for this DOW ---
            xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}
            lgbm_params = {'random_state': 42, 'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 500, 'verbosity': -1}
            cat_params = {'random_seed': 42, 'objective': 'RMSE', 'iterations': 500, 'verbose': 0}

            cust_xgb_f, xgb_model_cust, X_cust = train_and_forecast_tree_model_dow(xgb.XGBRegressor, xgb_params, df_dow, periods_to_forecast, 'customers')
            cust_lgbm_f, _, _ = train_and_forecast_tree_model_dow(lgb.LGBMRegressor, lgbm_params, df_dow, periods_to_forecast, 'customers')
            cust_cat_f, _, _ = train_and_forecast_tree_model_dow(cat.CatBoostRegressor, cat_params, df_dow, periods_to_forecast, 'customers', is_catboost=True)
            
            # Ensemble the customer forecasts for this DOW
            if not cust_xgb_f.empty:
                cust_ensemble_f = cust_xgb_f.copy()
                cust_ensemble_f['yhat'] = (cust_xgb_f['yhat'] + cust_lgbm_f['yhat'] + cust_cat_f['yhat']) / 3
                all_forecasts_cust.append(cust_ensemble_f)
                
                # Store model and features for SHAP analysis
                if xgb_model_cust and not X_cust.empty:
                    st.session_state.model_store[dow] = xgb_model_cust
                    st.session_state.feature_store[dow] = X_cust
                    try:
                        explainer = shap.TreeExplainer(xgb_model_cust)
                        shap_values = explainer(X_cust)
                        st.session_state.shap_values_store[dow] = shap_values
                    except Exception as e:
                        st.warning(f"Could not generate SHAP values for {day_name}s: {e}")


            # --- Step 4: Forecast ATV for this DOW ---
            atv_xgb_f, _, _ = train_and_forecast_tree_model_dow(xgb.XGBRegressor, xgb_params, df_dow, periods_to_forecast, 'atv')
            atv_lgbm_f, _, _ = train_and_forecast_tree_model_dow(lgb.LGBMRegressor, lgbm_params, df_dow, periods_to_forecast, 'atv')
            atv_cat_f, _, _ = train_and_forecast_tree_model_dow(cat.CatBoostRegressor, cat_params, df_dow, periods_to_forecast, 'atv', is_catboost=True)

            # Ensemble the ATV forecasts for this DOW
            if not atv_xgb_f.empty:
                atv_ensemble_f = atv_xgb_f.copy()
                atv_ensemble_f['yhat'] = (atv_xgb_f['yhat'] + atv_lgbm_f['yhat'] + atv_cat_f['yhat']) / 3
                all_forecasts_atv.append(atv_ensemble_f)

    # --- Step 5: Combine all DOW forecasts into a single timeline ---
    if not all_forecasts_cust or not all_forecasts_atv:
        st.error("Model training failed for all days of the week. Cannot produce a forecast.")
        return pd.DataFrame()

    final_cust_f = pd.concat(all_forecasts_cust).rename(columns={'yhat': 'forecast_customers'})
    final_atv_f = pd.concat(all_forecasts_atv).rename(columns={'yhat': 'forecast_atv'})

    combo_f = pd.merge(final_cust_f, final_atv_f, on='ds', how='inner')
    combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
    
    # Sort and truncate to the desired forecast horizon
    today = pd.to_datetime('today').normalize()
    end_date = today + timedelta(days=forecast_horizon - 1)
    final_forecast_df = combo_f[(combo_f['ds'] >= today) & (combo_f['ds'] <= end_date)].sort_values('ds').reset_index(drop=True)

    return final_forecast_df


def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist, fcst, metrics, target):
    fig=go.Figure()
    # This function needs historical forecasts to plot, which the new method doesn't generate.
    # For now, we'll just plot the future forecast against history.
    fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst[f'forecast_{target}'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')))
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
    return fig

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
                        db_client.collection('future_activities').document(doc_id).update(update_data)
                        st.success("Activity updated!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                with delete_col:
                    if st.form_submit_button("🗑️ Delete", use_container_width=True):
                        db_client.collection('future_activities').document(doc_id).delete()
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
                            db_client.collection('future_activities').document(doc_id).update(update_data)
                            st.success("Activity updated!")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                    with delete_col:
                        if st.form_submit_button("🗑️ Delete", use_container_width=True):
                            db_client.collection('future_activities').document(doc_id).delete()
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
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        if day_type == 'Not Normal Day':
            st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")

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

            day_type_options = ["Normal Day", "Not Normal Day"]
            current_day_type_index = day_type_options.index(day_type) if day_type in day_type_options else 0
            updated_day_type = st.selectbox("Day Type", options=day_type_options, index=current_day_type_index, key=f"day_type_{row['doc_id']}")
            updated_day_type_notes = st.text_input("Notes", value=row.get('day_type_notes', ''), key=f"notes_{row['doc_id']}")
            
            btn_cols = st.columns(2)
            if btn_cols[0].form_submit_button("💾 Update Record", use_container_width=True):
                update_data = {
                    'sales': updated_sales,
                    'customers': updated_customers,
                    'add_on_sales': updated_addons,
                    'weather': updated_weather,
                    'day_type': updated_day_type,
                    'day_type_notes': updated_day_type_notes if updated_day_type == 'Not Normal Day' else ''
                }
                db_client.collection('historical_data').document(row['doc_id']).update(update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

            if btn_cols[1].form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                db_client.collection('historical_data').document(row['doc_id']).delete()
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
        st.info("Forecasting with a Day-Specific AI Ensemble")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning to get latest data.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                # --- RE-ENGINEERED FORECASTING CALL ---
                st.session_state.forecast_df = pd.DataFrame() # Clear previous forecast
                forecast_df = generate_day_by_day_forecast(
                    st.session_state.historical_df, 
                    st.session_state.events_df, 
                    forecast_horizon=15
                )

                if not forecast_df.empty:
                    with st.spinner("🛰️ Fetching live weather..."):
                        weather_df = get_weather_forecast()
                    if weather_df is not None:
                        forecast_df = pd.merge(forecast_df, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                    else:
                        forecast_df['weather'] = 'Not Available'
                    
                    st.session_state.forecast_df = forecast_df
                    
                    try:
                        with st.spinner("📝 Saving forecast log for future accuracy tracking..."):
                            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                            today_date_naive = pd.to_datetime('today').tz_localize(None).normalize()
                            
                            for _, row in forecast_df.iterrows():
                                log_entry = {
                                    "generated_on": today_date_naive,
                                    "forecast_for_date": row['ds'],
                                    "predicted_sales": row['forecast_sales'],
                                    "predicted_customers": row['forecast_customers']
                                }
                                doc_id = f"{today_date_naive.strftime('%Y-%m-%d')}_{row['ds'].strftime('%Y-%m-%d')}"
                                db.collection('forecast_log').document(doc_id).set(log_entry)
                            st.info("Forecast log saved successfully.")
                    except Exception as e:
                        st.error(f"Failed to save forecast log: {e}")

                    st.success("Day-Specific AI forecast generated successfully!")
                else:
                    st.error("Forecast generation failed. Check warnings for details on which days had insufficient data.")

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
            future_forecast_df = st.session_state.forecast_df.copy()
            disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
            existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns};display_df=future_forecast_df.rename(columns=existing_disp_cols);final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
            st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
            st.markdown("#### Forecast Visualization");fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
            with st.expander("🔬 View Full Model Diagnostic Chart"):
                st.info("This chart shows how the new forecast compares against historical data.");
                d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);
                hist_atv=calculate_atv(st.session_state.historical_df.copy())
                
                with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv, st.session_state.forecast_df, {},'customers'),use_container_width=True)
                with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv, st.session_state.forecast_df, {},'atv'),use_container_width=True)
        else:st.info("Click the 'Generate Forecast' button to begin.")
    
    with tabs[1]:
        st.header("💡 Forecast Insights (XGBoost Day-Specific Drivers)")
        st.info("This analysis shows what features drove the forecast for a specific day, using the model trained **only** for that day of the week (e.g., the 'Monday Model').")
        if not st.session_state.model_store:
            st.info("Click 'Generate Forecast' to build the day-specific models and see their insights.")
        else:
            future_forecast_df = st.session_state.forecast_df.copy()
            future_forecast_df['date_str'] = future_forecast_df['ds'].dt.strftime('%A, %B %d, %Y')
            selected_date_str = st.selectbox("Select a day to analyze its forecast drivers:", options=future_forecast_df['date_str'])
            
            selected_row = future_forecast_df[future_forecast_df['date_str'] == selected_date_str].iloc[0]
            selected_date = selected_row['ds']
            selected_dow = selected_date.dayofweek
            
            model = st.session_state.model_store.get(selected_dow)
            shap_values = st.session_state.shap_values_store.get(selected_dow)
            X_cust = st.session_state.feature_store.get(selected_dow)

            if model and shap_values is not None and X_cust is not None:
                try:
                    st.subheader(f"Analysis for {selected_date_str}")
                    
                    # Find the corresponding historical data point to explain
                    # For simplicity, we'll explain the last data point used for training that model
                    st.markdown("##### SHAP Explanation for Last Trained Point")
                    st.info("This waterfall plot explains the model's prediction for the most recent historical data point for this day of the week.")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.waterfall_plot(shap_values[-1], show=False)
                    plt.title(f'XGBoost Driver Analysis for most recent {selected_date.strftime("%A")}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()

                    with st.expander(f"View Overall Feature Importance for all {selected_date.strftime('%A')}s"):
                        st.info("This plot shows the average impact of each feature across all historical predictions for this day of the week.")
                        fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
                        shap.summary_plot(shap_values.values, X_cust, plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig_summary)
                        plt.clf()

                except Exception as e:
                    st.error(f"An error occurred while building the SHAP plot: {e}")
            else:
                st.warning(f"Insight data not available for {selected_date.strftime('%A')}s. The model may not have been trained due to insufficient data.")

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
                
                new_day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"], help="Select 'Not Normal Day' if an unexpected event significantly impacted sales.")
                new_day_type_notes = ""
                if new_day_type == "Not Normal Day":
                    new_day_type_notes = st.text_area("Notes for Not Normal Day (Optional)", placeholder="e.g., Power outage, unexpected local event...")

                if st.form_submit_button("✅ Save Record"):
                    new_rec={
                        "date":pd.to_datetime(new_date).to_pydatetime(),
                        "sales":new_sales,
                        "customers":new_customers,
                        "weather":new_weather,
                        "add_on_sales":new_addons,
                        "day_type": new_day_type,
                        "day_type_notes": new_day_type_notes
                    }
                    db.collection('historical_data').add(new_rec)
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
                        display_cols = ['date', 'sales', 'customers', 'add_on_sales', 'weather', 'day_type']
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
                                "day_type": "Day Type"
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
            
            activities_df = load_from_firestore(db, 'future_activities')
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
                activities_df = load_from_firestore(db, 'future_activities')
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

                    filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].copy().sort_values('date')

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
