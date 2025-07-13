import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    if 'recurring_events_df' not in st.session_state: st.session_state.recurring_events_df = load_from_firestore(db_client, 'recurring_events')

    defaults = {
        'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688", 'authentication_status': False,
        'access_level': 0, 'username': None, 'forecast_components': pd.DataFrame(), 'migration_done': False,
        'show_recent_entries': False, 'show_all_activities': False, 'show_recurring_events_dialog': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- User Management Functions ---
def get_user(db_client, username):
    users_ref = db_client.collection('users')
    query = users_ref.where('username', '==', username).limit(1)
    results = list(query.stream())
    return results[0] if results else None

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
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'potential_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

def remove_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25); Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1; upper_bound = Q3 + 1.5 * IQR
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
        st.error(f"Could not fetch weather data. Error: {e}"); return None

def map_weather_code(code):
    if code in [0, 1]: return "Sunny"
    if code == 2: return "Partly Cloudy"
    if code == 3: return "Cloudy"
    if code in [45, 48]: return "Foggy"
    if code in [51, 53, 55, 61, 63, 65]: return "Rainy"
    if code in [80, 81, 82]: return "Rain Showers"
    if code in [95, 96, 99]: return "Thunderstorm"
    return "Cloudy"

# --- Core Forecasting Models ---
def create_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek; df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month; df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear; df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df = df.sort_values('date')
    df['sales_lag_7'] = df['sales'].shift(7); df['customers_lag_7'] = df['customers'].shift(7)
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
    
    # --- REVERTED: Recurring events are no longer part of the main training ---
    manual_events_renamed = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_manual_events = manual_events_renamed.dropna(subset=['ds', 'holiday'])

    if 'day_type' in historical_df.columns:
        not_normal_days_df = historical_df[historical_df['day_type'] == 'Not Normal Day'].copy()
        if not not_normal_days_df.empty:
            not_normal_events = pd.DataFrame({'holiday': 'Unusual_Day', 'ds': pd.to_datetime(not_normal_days_df['date']), 'lower_window': 0, 'upper_window': 0})
            all_manual_events = pd.concat([all_manual_events, not_normal_events])
    
    use_yearly_seasonality = len(df_train) >= 365

    prophet_model = Prophet(
        growth='linear', holidays=all_manual_events, daily_seasonality=False,
        weekly_seasonality=True, yearly_seasonality=use_yearly_seasonality, 
        changepoint_prior_scale=0.15, changepoint_range=0.9,
    )
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
    
    full_df_featured['is_not_normal_day'] = full_df_featured['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0) if 'day_type' in full_df_featured.columns else 0

    df_featured = full_df_featured[full_df_featured['date'] <= last_date].copy()
    future_df_featured = full_df_featured[full_df_featured['date'] > last_date].copy()
    df_featured.dropna(subset=['sales_lag_7'], inplace=True)

    features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7', 'is_not_normal_day']
    features = [f for f in features if f in df_featured.columns]
    target = target_col
    X = df_featured[features]; y = df_featured[target]
    
    if X.empty or len(X) < 10:
        st.warning("Not enough data for XGBoost model."); return pd.DataFrame()
        
    sample_weights = np.linspace(0.5, 1.0, len(y))
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42, shuffle=False)

    def objective(trial):
        params = {'objective': 'reg:squarederror', 'n_estimators': trial.suggest_int('n_estimators', 500, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'random_state': 42, 'early_stopping_rounds': 50}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=weights_train, eval_set=[(X_test, y_test)], sample_weight_eval_set=[weights_test], verbose=False)
        mae = mean_absolute_error(y_test, model.predict(X_test), sample_weight=weights_test)
        return mae

    study = optuna.create_study(direction='minimize'); study.optimize(objective, n_trials=50) 
    best_params = study.best_params; best_params.pop('early_stopping_rounds', None)
    final_model = xgb.XGBRegressor(**best_params); final_model.fit(X, y, sample_weight=sample_weights)

    future_predictions = final_model.predict(future_df_featured[features])
    full_prediction_df = df_featured[['date']].copy(); full_prediction_df['yhat'] = final_model.predict(X)
    future_df_template['yhat'] = future_predictions
    return pd.concat([full_prediction_df, future_df_template], ignore_index=True).rename(columns={'date': 'ds'})

# --- Plotting & Firestore I/O ---
def add_to_firestore(db_client, collection_name, data, historical_df=None):
    if db_client is None: return
    if collection_name == 'historical_data' and historical_df is not None:
        if 'date' in data and pd.notna(data['date']):
            current_date = pd.to_datetime(data['date'])
            last_year_date = current_date - timedelta(days=364)
            hist_copy = historical_df.copy(); hist_copy['date_only'] = pd.to_datetime(hist_copy['date']).dt.date
            last_year_record = hist_copy[hist_copy['date_only'] == last_year_date.date()]
            data['last_year_sales'] = last_year_record['sales'].iloc[0] if not last_year_record.empty else 0.0
            data['last_year_customers'] = last_year_record['customers'].iloc[0] if not last_year_record.empty else 0.0
            data['date'] = current_date.to_pydatetime()
        else: return
        all_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'weather', 'day_type', 'day_type_notes']
        for col in all_cols:
            if col in data and data[col] is not None:
                if col not in ['weather', 'day_type', 'day_type_notes']: data[col] = float(pd.to_numeric(data[col], errors='coerce'))
            else:
                if col not in ['weather', 'day_type', 'day_type_notes']: data[col] = 0.0
                else: data[col] = "N/A" if col == 'weather' else "Normal Day"
    elif collection_name == 'recurring_events':
        data['date'] = pd.to_datetime(data['date']).to_pydatetime()

    db_client.collection(collection_name).add(data)

def update_historical_record_in_firestore(db_client, doc_id, data):
    if db_client: db_client.collection('historical_data').document(doc_id).update(data)

def update_activity_in_firestore(db_client, doc_id, data):
    if db_client and 'potential_sales' in data:
        data['potential_sales'] = float(data['potential_sales'])
        db_client.collection('future_activities').document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client: db_client.collection(collection_name).document(doc_id).delete()

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist,fcst,metrics,target):
    fig=go.Figure();fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')));fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')));title_text=f"{target.replace('_',' ').title()} Forecast";y_axis_title=title_text+' (₱)'if'atv'in target or'sales'in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical',xaxis_title='Date',yaxis_title=y_axis_title,legend=dict(x=0.01,y=0.99),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)");return fig

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']): x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']): x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']):
        holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"
        x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')
    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

def create_daily_evaluation_data(historical_df, forecast_df):
    if forecast_df.empty or historical_df.empty: return pd.DataFrame()
    hist_eval = historical_df.copy(); hist_eval['date'] = pd.to_datetime(hist_eval['date'])
    hist_eval = hist_eval[['date', 'sales', 'customers', 'add_on_sales']].rename(columns={'sales': 'actual_sales', 'customers': 'actual_customers'})
    fcst_eval = forecast_df.copy(); fcst_eval['ds'] = pd.to_datetime(fcst_eval['ds'])
    fcst_eval = fcst_eval[['ds', 'forecast_sales', 'forecast_customers']]
    eval_df = pd.merge(hist_eval, fcst_eval, left_on='date', right_on='ds', how='inner')
    if eval_df.empty: return pd.DataFrame()
    eval_df.drop(columns=['ds'], inplace=True)
    forecast_sales_numeric = pd.to_numeric(eval_df['forecast_sales'], errors='coerce').fillna(0)
    add_on_sales_numeric = pd.to_numeric(eval_df['add_on_sales'], errors='coerce').fillna(0)
    eval_df['adjusted_forecast_sales'] = forecast_sales_numeric + add_on_sales_numeric
    return eval_df

def calculate_accuracy_metrics(historical_df, forecast_df, days=30):
    eval_df = create_daily_evaluation_data(historical_df, forecast_df)
    if eval_df is None or eval_df.empty: return None
    period_start_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=days)
    eval_df_period = eval_df[eval_df['date'] >= period_start_date].copy()
    if eval_df_period.empty: return None
    actual_s = eval_df_period['actual_sales']; forecast_s = eval_df_period['adjusted_forecast_sales']
    non_zero_mask_s = actual_s != 0
    sales_mape = np.mean(np.abs((actual_s[non_zero_mask_s] - forecast_s[non_zero_mask_s]) / actual_s[non_zero_mask_s])) * 100 if non_zero_mask_s.any() else float('inf')
    sales_mae = mean_absolute_error(actual_s, forecast_s)
    actual_c = eval_df_period['actual_customers']; forecast_c = eval_df_period['forecast_customers']
    non_zero_mask_c = actual_c != 0
    customer_mape = np.mean(np.abs((actual_c[non_zero_mask_c] - forecast_c[non_zero_mask_c]) / actual_c[non_zero_mask_c])) * 100 if non_zero_mask_c.any() else float('inf')
    customer_mae = mean_absolute_error(actual_c, forecast_c)
    return {"sales_accuracy": 100 - sales_mape, "sales_mae": sales_mae, "customer_accuracy": 100 - customer_mape, "customer_mae": customer_mae}

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty or actual_col not in df.columns or forecast_col not in df.columns:
        return go.Figure().update_layout(title=title, paper_bgcolor='white', plot_bgcolor='white', font_color='black', xaxis={"visible": False}, yaxis={"visible": False}, annotations=[{"text": "No data available.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual', line=dict(color='#1f77b4', width=2), marker=dict(symbol='circle', size=6)))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash', width=2), marker=dict(symbol='x', size=7)))
    fig.update_layout(title=dict(text=title, font=dict(color='black', size=20)), xaxis_title=dict(text='Date', font=dict(color='black', size=14)), yaxis_title=dict(text=y_axis_title, font=dict(color='black', size=14)), legend=dict(font=dict(color='black', size=12)), height=450, margin=dict(l=50, r=50, t=80, b=50), paper_bgcolor='white', plot_bgcolor='white', font_color='black')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', tickfont=dict(color='black', size=12)); fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', tickfont=dict(color='black', size=12))
    return fig

def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data.get('weekly', 0),'Time of Year':day_data.get('yearly', 0),'Holidays/Events':day_data.get('holidays', 0)}
    significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data.get('trend', 0):.0f} customers**.\n\n"
    if not significant_effects: summary += f"The final forecast of **{day_data.get('yhat', 0):.0f} customers** is driven by this trend."; return summary
    pos_drivers={k:v for k,v in significant_effects.items()if v>0};neg_drivers={k:v for k,v in significant_effects.items()if v<0}
    if pos_drivers:biggest_pos_driver=max(pos_drivers,key=pos_drivers.get);summary+=f"📈 Main positive driver is **{biggest_pos_driver}**,adding an estimated **{pos_drivers[biggest_pos_driver]:.0f} customers**.\n"
    if neg_drivers:biggest_neg_driver=min(neg_drivers,key=neg_drivers.get);summary+=f"📉 Main negative driver is **{biggest_neg_driver}**,reducing by **{abs(neg_drivers[biggest_neg_driver]):.0f} customers**.\n"
    summary+=f"\nAfter all factors,the final forecast is **{day_data.get('yhat', 0):.0f} customers**.";return summary

def render_activity_card(row, db_client, view_type='compact_list', access_level=3):
    if 'doc_id' not in row: return 
    doc_id = row['doc_id']
    if view_type == 'compact_list':
        date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y'); summary_line = f"**{date_str}** | {row['activity_name']}"
        with st.expander(summary_line):
            status = row['remarks']; color = {'Confirmed': '#22C55E', 'Needs Follow-up': '#F59E0B', 'Tentative': '#38BDF8'}.get(status, '#EF4444')
            st.markdown(f"**Status:** <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True); st.markdown(f"**Potential Sales:** ₱{row['potential_sales']:,.2f}")
            if access_level <= 2:
                with st.form(key=f"compact_update_form_{doc_id}", border=False):
                    status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]; current_status_index = status_options.index(status) if status in status_options else 0
                    updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"compact_sales_{doc_id}")
                    updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"compact_remarks_{doc_id}")
                    update_col, delete_col = st.columns(2)
                    if update_col.form_submit_button("💾 Update", use_container_width=True):
                        update_activity_in_firestore(db, doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks}); st.success("Activity updated!"); st.cache_data.clear(); time.sleep(1); st.rerun()
                    if delete_col.form_submit_button("🗑️ Delete", use_container_width=True):
                        delete_from_firestore(db, 'future_activities', doc_id); st.warning("Activity deleted."); st.cache_data.clear(); time.sleep(1); st.rerun()
    else:
        with st.container(border=True):
            st.markdown(f"**{row['activity_name']}**"); st.markdown(f"<small>📅 {pd.to_datetime(row['date']).strftime('%A, %B %d, %Y')}</small>", unsafe_allow_html=True)
            st.markdown(f"💰 ₱{row['potential_sales']:,.2f}"); status = row['remarks']; color = {'Confirmed': '#22C55E', 'Needs Follow-up': '#F59E0B', 'Tentative': '#38BDF8'}.get(status, '#EF4444')
            st.markdown(f"Status: <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            if access_level <= 2:
                with st.expander("Edit / Manage"):
                    status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]; current_status_index = status_options.index(status) if status in status_options else 0
                    with st.form(key=f"full_update_form_{doc_id}", border=False):
                        updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"full_sales_{doc_id}")
                        updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"full_remarks_{doc_id}")
                        update_col, delete_col = st.columns(2)
                        if update_col.form_submit_button("💾 Update", use_container_width=True):
                            update_activity_in_firestore(db, doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks}); st.success("Activity updated!"); st.cache_data.clear(); time.sleep(1); st.rerun()
                        if delete_col.form_submit_button("🗑️ Delete", use_container_width=True):
                            delete_from_firestore(db, 'future_activities', doc_id); st.warning("Activity deleted."); st.cache_data.clear(); time.sleep(1); st.rerun()

def render_historical_record(row, db_client):
    if 'doc_id' not in row: return
    date_str = row['date'].strftime('%B %d, %Y'); expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}"); st.write(f"**Weather:** {row.get('weather', 'N/A')}")
        day_type = row.get('day_type', 'Normal Day'); st.write(f"**Day Type:** {day_type}")
        if day_type == 'Not Normal Day': st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")
        if st.session_state['access_level'] <= 2:
            with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
                st.write("---"); st.markdown("**Edit Record**")
                edit_cols = st.columns(2)
                updated_sales = edit_cols[0].number_input("Sales (₱)", value=float(row.get('sales', 0)), format="%.2f", key=f"sales_{row['doc_id']}")
                updated_customers = edit_cols[1].number_input("Customers", value=int(row.get('customers', 0)), key=f"cust_{row['doc_id']}")
                edit_cols2 = st.columns(2)
                updated_addons = edit_cols2[0].number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales', 0)), format="%.2f", key=f"addon_{row['doc_id']}")
                weather_options = ["Sunny", "Cloudy", "Rainy", "Storm"]; current_weather_index = weather_options.index(row.get('weather')) if row.get('weather') in weather_options else 0
                updated_weather = edit_cols2[1].selectbox("Weather", options=weather_options, index=current_weather_index, key=f"weather_{row['doc_id']}")
                day_type_options = ["Normal Day", "Not Normal Day"]; current_day_type_index = day_type_options.index(day_type) if day_type in day_type_options else 0
                updated_day_type = st.selectbox("Day Type", options=day_type_options, index=current_day_type_index, key=f"day_type_{row['doc_id']}")
                updated_day_type_notes = st.text_input("Notes", value=row.get('day_type_notes', ''), key=f"notes_{row['doc_id']}")
                btn_cols = st.columns(2)
                if btn_cols[0].form_submit_button("💾 Update Record", use_container_width=True):
                    update_data = {'sales': updated_sales, 'customers': updated_customers, 'add_on_sales': updated_addons, 'weather': updated_weather, 'day_type': updated_day_type, 'day_type_notes': updated_day_type_notes if updated_day_type == 'Not Normal Day' else ''}
                    update_historical_record_in_firestore(db_client, row['doc_id'], update_data); st.success(f"Record for {date_str} updated!"); st.cache_data.clear(); time.sleep(1); st.rerun()
                if btn_cols[1].form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                    delete_from_firestore(db_client, 'historical_data', row['doc_id']); st.warning(f"Record for {date_str} deleted."); st.cache_data.clear(); time.sleep(1); st.rerun()

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
                            st.session_state['authentication_status'] = True; st.session_state['username'] = username
                            st.session_state['access_level'] = user_data['access_level']; st.rerun()
                        else: st.error("Incorrect password")
                    else: st.error("User not found")
    
    else:
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
            st.info("Forecasting with a Hybrid Ensemble Model")
            if st.button("🔄 Refresh Data from Firestore"):
                st.cache_data.clear(); st.success("Data cache cleared."); time.sleep(1); st.rerun()
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
                        FORECAST_HORIZON = 15
                        cust_f, atv_f = pd.DataFrame(), pd.DataFrame()
                        with st.spinner("Forecasting Customers..."):
                            prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                            xgb_cust_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                            if not prophet_cust_f.empty and not xgb_cust_f.empty:
                                cust_f = pd.merge(prophet_cust_f, xgb_cust_f, on='ds', suffixes=('_prophet', '_xgb')); cust_f['yhat'] = (cust_f['yhat_prophet'] + cust_f['yhat_xgb']) / 2
                            else: st.error("Failed to generate customer forecast.")
                        with st.spinner("Forecasting Average Sale..."):
                            VALIDATION_PERIOD = 30; train_df = hist_df_with_atv.iloc[:-VALIDATION_PERIOD]; validation_df = hist_df_with_atv.iloc[-VALIDATION_PERIOD:]
                            prophet_atv_val, _ = train_and_forecast_prophet(train_df, ev_df, VALIDATION_PERIOD, 'atv')
                            xgb_atv_val = train_and_forecast_xgboost_tuned(train_df, ev_df, VALIDATION_PERIOD, 'atv')
                            meta_model_atv = None
                            if not prophet_atv_val.empty and not xgb_atv_val.empty:
                                validation_preds_atv = pd.merge(prophet_atv_val[['ds', 'yhat']], xgb_atv_val[['ds', 'yhat']], on='ds', suffixes=('_prophet', '_xgb'))
                                validation_data_atv = pd.merge(validation_preds_atv, validation_df[['date', 'atv']], left_on='ds', right_on='date')
                                meta_model_atv = LinearRegression(); meta_model_atv.fit(validation_data_atv[['yhat_prophet', 'yhat_xgb']], validation_data_atv['atv'])
                            if meta_model_atv:
                                prophet_atv_f, _ = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                                xgb_atv_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                                if not prophet_atv_f.empty and not xgb_atv_f.empty:
                                    atv_f = pd.merge(prophet_atv_f, xgb_atv_f, on='ds', suffixes=('_prophet', '_xgb'))
                                    atv_f['yhat'] = meta_model_atv.predict(atv_f[['yhat_prophet', 'yhat_xgb']])
                                else: st.error("Failed to generate ATV forecast.")
                            else: st.error("Failed to train ATV meta-model.")
                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_customers'}), atv_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            # --- MODIFICATION: Apply recurring event impact post-forecast ---
                            rec_ev_df = st.session_state.recurring_events_df.copy()
                            if not rec_ev_df.empty:
                                rec_ev_df['date'] = pd.to_datetime(rec_ev_df['date'])
                                for _, event in rec_ev_df.iterrows():
                                    if event['date'].date() in pd.to_datetime(combo_f['ds']).dt.date.values:
                                        last_year_event_date = event['date'] - pd.DateOffset(years=1)
                                        hist_event_data = hist_df_with_atv[hist_df_with_atv['date'] == last_year_event_date]
                                        if not hist_event_data.empty:
                                            event_sales = hist_event_data['sales'].iloc[0]
                                            event_customers = hist_event_data['customers'].iloc[0]
                                            
                                            # Compare to average of surrounding days
                                            surrounding_dates = pd.date_range(start=last_year_event_date - timedelta(days=3), end=last_year_event_date + timedelta(days=3))
                                            surrounding_data = hist_df_with_atv[hist_df_with_atv['date'].isin(surrounding_dates) & (hist_df_with_atv['date'] != last_year_event_date)]
                                            
                                            if not surrounding_data.empty:
                                                avg_sales = surrounding_data['sales'].mean()
                                                avg_customers = surrounding_data['customers'].mean()
                                                
                                                sales_impact = event_sales - avg_sales
                                                customer_impact = event_customers - avg_customers
                                                
                                                # Apply impact to the forecast
                                                combo_f.loc[combo_f['ds'].dt.date == event['date'].date(), 'forecast_sales'] += sales_impact
                                                combo_f.loc[combo_f['ds'].dt.date == event['date'].date(), 'forecast_customers'] += customer_impact
                                                st.success(f"Applied historical impact for upcoming event: {event['event_name']}")

                            with st.spinner("🛰️ Fetching live weather..."): weather_df = get_weather_forecast()
                            if weather_df is not None: combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                            else: combo_f['weather'] = 'Not Available'
                            st.session_state.forecast_df = combo_f
                            if prophet_model_cust:
                                st.session_state.forecast_components = prophet_model_cust.predict(prophet_cust_f[['ds']])
                                st.session_state.all_holidays = prophet_model_cust.holidays
                            st.success("Hybrid ensemble forecast generated successfully!")
                        else: st.error("Forecast generation failed.")
            st.markdown("---")
            st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
            st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)
            if st.button("Logout"):
                st.session_state['authentication_status'] = False; st.session_state['username'] = None
                st.session_state['access_level'] = 0; st.rerun()
        
        tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data", "📅 Future Activities", "📜 Historical Data"]
        if st.session_state['access_level'] == 1: tab_list.append("👥 User Interface")
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
                with st.expander("🔬 View Full Forecast vs. Historical Data"):
                    st.info("This view shows how the component models performed against past data.");d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);hist_atv=calculate_atv(st.session_state.historical_df.copy())
                    with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_customers':'yhat'}),st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                    with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_atv':'yhat'}),st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
            else:st.info("Click 'Generate Forecast' to begin.")
        
        with tabs[1]:
            st.info("The breakdown below is generated by the Prophet model component of the forecast.")
            if'forecast_components'not in st.session_state or st.session_state.forecast_components.empty:st.info("Generate a forecast first to see the breakdown of its drivers.")
            else:
                future_components=st.session_state.forecast_components[st.session_state.forecast_components['ds']>=pd.to_datetime('today').normalize()].copy()
                if not future_components.empty:
                    future_components['date_str']=future_components['ds'].dt.strftime('%A,%B %d,%Y')
                    selected_date_str=st.selectbox("Select a day to analyze its forecast drivers:",options=future_components['date_str'])
                    selected_date=future_components[future_components['date_str']==selected_date_str]['ds'].iloc[0]
                    breakdown_fig,day_data=plot_forecast_breakdown(st.session_state.forecast_components,selected_date,st.session_state.all_holidays)
                    st.plotly_chart(breakdown_fig,use_container_width=True);st.markdown("---");st.subheader("Insight Summary");st.markdown(generate_insight_summary(day_data,selected_date))
                else:st.warning("No future dates available in the forecast components to analyze.")
        
        with tabs[2]:
            st.header("📈 Forecast Performance Evaluator")
            st.info("ℹ️ **Comparison Logic**: The 'Forecast' sales value is adjusted by adding the *actual add-on sales* for each day. The metrics and charts below use this adjusted forecast.")
            eval_tab_7, eval_tab_30 = st.tabs(["Last 7 Days", "Last 30 Days"])
            def render_evaluation_content(days):
                st.subheader(f"Accuracy Metrics for the Last {days} Days"); metrics = calculate_accuracy_metrics(st.session_state.historical_df, st.session_state.forecast_df, days=days)
                if metrics:
                    col1, col2 = st.columns(2)
                    with col1: st.metric(label="Sales MAPE (Accuracy)", value=f"{metrics['sales_accuracy']:.2f}%"); st.metric(label="Sales MAE (Avg Error)", value=f"₱{metrics['sales_mae']:,.2f}")
                    with col2: st.metric(label="Customer MAPE (Accuracy)", value=f"{metrics['customer_accuracy']:.2f}%"); st.metric(label="Customer MAE (Avg Error)", value=f"{int(round(metrics['customer_mae']))} customers")
                else: st.warning(f"Not enough overlapping data in the last {days} days to calculate metrics.")
                st.markdown("---"); st.subheader(f"Comparison Charts for the Last {days} Days")
                evaluation_df_daily = create_daily_evaluation_data(st.session_state.historical_df, st.session_state.forecast_df)
                period_start_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=days)
                chart_df = evaluation_df_daily[evaluation_df_daily['date'] >= period_start_date]
                if chart_df.empty: st.info(f"No overlapping data found for the last {days} days to display charts.")
                else:
                    st.plotly_chart(plot_evaluation_graph(chart_df, date_col='date', actual_col='actual_sales', forecast_col='adjusted_forecast_sales', title='Actual Sales vs. Forecast Sales', y_axis_title='Sales (₱)'), use_container_width=True)
                    st.plotly_chart(plot_evaluation_graph(chart_df, date_col='date', actual_col='actual_customers', forecast_col='forecast_customers', title='Actual Customers vs. Forecast Customers', y_axis_title='Number of Customers'), use_container_width=True)
            with eval_tab_7: render_evaluation_content(7)
            with eval_tab_30: render_evaluation_content(30)
        
        with tabs[3]:
            if st.session_state['access_level'] <= 2:
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    with st.container(border=True):
                        st.subheader("✍️ Add New Daily Record")
                        with st.form("new_record_form",clear_on_submit=True):
                            new_date=st.date_input("Date", date.today()); new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                            new_customers=st.number_input("Customer Count",min_value=0); new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                            new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"]); new_day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"], help="Select 'Not Normal Day' for unexpected events.")
                            new_day_type_notes = st.text_area("Notes for Not Normal Day (Optional)", placeholder="e.g., Power outage...") if new_day_type == "Not Normal Day" else ""
                            if st.form_submit_button("✅ Save Record", use_container_width=True):
                                new_rec={"date":new_date, "sales":new_sales, "customers":new_customers, "weather":new_weather, "add_on_sales":new_addons, "day_type": new_day_type, "day_type_notes": new_day_type_notes}
                                add_to_firestore(db,'historical_data',new_rec, st.session_state.historical_df); st.cache_data.clear(); st.success("Record added!"); time.sleep(1); st.rerun()
                
                with col2:
                    with st.container(border=True):
                        st.subheader("🎉 Manage Recurring Events")
                        with st.form("add_recurring_event_form"):
                            event_name = st.text_input("Event Name", placeholder="e.g., Pintaflores Festival")
                            event_date = st.date_input("Date of Event")
                            if st.form_submit_button("Add Event", use_container_width=True):
                                if event_name and event_date:
                                    add_to_firestore(db, 'recurring_events', {'event_name': event_name, 'date': event_date})
                                    st.success(f"Added '{event_name}' to recurring events."); st.cache_data.clear(); time.sleep(1); st.rerun()
                                else: st.warning("Event name and date are required.")

                st.markdown("---")
                st.subheader("🗓️ Recent Entries & Events")
                entry_tab, event_tab = st.tabs(["Recent Daily Entries", "Registered Recurring Events"])
                with entry_tab:
                    recent_df = st.session_state.historical_df.copy().sort_values(by="date", ascending=False).head(10)
                    if not recent_df.empty:
                        display_cols = ['date', 'sales', 'customers', 'add_on_sales', 'weather', 'day_type']
                        st.dataframe(recent_df[[col for col in display_cols if col in recent_df.columns]], use_container_width=True, hide_index=True,
                            column_config={"date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"), "sales": st.column_config.NumberColumn("Sales (₱)", format="₱%.2f"),
                                           "customers": st.column_config.NumberColumn("Customers", format="%d"), "add_on_sales": st.column_config.NumberColumn("Add-on Sales (₱)", format="₱%.2f"),
                                           "weather": "Weather", "day_type": "Day Type"})
                    else: st.info("No recent data to display.")
                
                with event_tab:
                    recurring_df = st.session_state.recurring_events_df
                    if not recurring_df.empty:
                        for index, row in recurring_df.iterrows():
                            re_col1, re_col2 = st.columns([3,1])
                            event_display_date = pd.to_datetime(row['date']).strftime('%B %d, %Y')
                            re_col1.write(f"**{row['event_name']}** - {event_display_date}")
                            if re_col2.button("Delete", key=f"delete_rec_{row['doc_id']}", use_container_width=True):
                                delete_from_firestore(db, 'recurring_events', row['doc_id']); st.warning(f"Deleted '{row['event_name']}'."); st.cache_data.clear(); time.sleep(1); st.rerun()
                    else: st.info("No recurring events added yet.")

            else: st.warning("You do not have permission to add or edit data.")

        with tabs[4]:
            def set_view_all(): st.session_state.show_all_activities = True
            def set_overview(): st.session_state.show_all_activities = False
            if st.session_state.get('show_all_activities'):
                st.markdown("#### All Upcoming Activities"); st.button("⬅️ Back to Overview", on_click=set_overview)
                activities_df = load_from_firestore(db, 'future_activities')
                all_upcoming_df = pd.DataFrame()
                if not activities_df.empty:
                    all_upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy()
                
                if all_upcoming_df.empty: st.info("No upcoming activities scheduled.")
                else:
                    all_upcoming_df['month_year'] = pd.to_datetime(all_upcoming_df['date']).dt.strftime('%B %Y')
                    sorted_months_df = all_upcoming_df.sort_values('date'); month_tabs_list = sorted_months_df['month_year'].unique().tolist()
                    if month_tabs_list:
                        month_tabs = st.tabs(month_tabs_list)
                        for i, tab in enumerate(month_tabs):
                            with tab:
                                month_name = month_tabs_list[i]; month_df = sorted_months_df[sorted_months_df['month_year'] == month_name]
                                header_cols = st.columns([2, 1, 1])
                                with header_cols[1]: st.metric(label="Total Expected Sales", value=f"₱{month_df['potential_sales'].sum():,.2f}")
                                with header_cols[2]: st.metric(label="Unconfirmed Activities", value=len(month_df[month_df['remarks'] != 'Confirmed']))
                                st.markdown("---")
                                activities = month_df.to_dict('records')
                                for i in range(0, len(activities), 4):
                                    cols = st.columns(4); row_activities = activities[i:i+4]
                                    for j, activity in enumerate(row_activities):
                                        with cols[j]: render_activity_card(activity, db, view_type='grid', access_level=st.session_state['access_level'])
            else:
                col1, col2 = st.columns([1, 2], gap="large")
                with col1:
                    if st.session_state['access_level'] <= 3:
                        st.markdown("##### Add New Activity")
                        with st.form("new_activity_form", clear_on_submit=True, border=True):
                            activity_name = st.text_input("Activity/Event Name", placeholder="e.g., Catering for Birthday")
                            activity_date = st.date_input("Date of Activity", min_value=date.today()); potential_sales = st.number_input("Potential Sales (₱)", min_value=0.0, format="%.2f")
                            remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
                            if st.form_submit_button("✅ Save Activity", use_container_width=True):
                                if activity_name and activity_date:
                                    new_activity = {"activity_name": activity_name, "date": pd.to_datetime(activity_date).to_pydatetime(), "potential_sales": float(potential_sales), "remarks": remarks}
                                    db.collection('future_activities').add(new_activity); st.success(f"Activity '{activity_name}' saved!"); st.cache_data.clear(); time.sleep(1); st.rerun()
                                else: st.warning("Activity name and date are required.")
                    else: st.warning("You do not have permission to add new activities.")
                with col2:
                    st.markdown("##### Next 10 Upcoming Activities")
                    btn_cols = st.columns(2); btn_cols[0].button("🔄 Refresh List", key='refresh_activities', use_container_width=True, on_click=st.rerun)
                    btn_cols[1].button("📂 View All Upcoming Activities", use_container_width=True, on_click=set_view_all)
                    st.markdown("---",)
                    activities_df = load_from_firestore(db, 'future_activities')
                    upcoming_df = pd.DataFrame()
                    if not activities_df.empty:
                        upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy().head(10)
                    
                    if upcoming_df.empty: st.info("No upcoming activities scheduled.")
                    else:
                        for _, row in upcoming_df.iterrows(): render_activity_card(row, db, view_type='compact_list', access_level=st.session_state['access_level'])

        with tabs[5]:
            st.subheader("View & Edit Historical Data")
            df = st.session_state.historical_df.copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce'); df.dropna(subset=['date'], inplace=True)
                all_years = sorted(df['date'].dt.year.unique(), reverse=True)
                if all_years:
                    filter_cols = st.columns(2)
                    with filter_cols[0]: selected_year = st.selectbox("Select Year to View:", options=all_years)
                    df_year_filtered = df[df['date'].dt.year == selected_year]
                    all_months = sorted(df_year_filtered['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)
                    if all_months:
                        with filter_cols[1]: selected_month_str = st.selectbox("Select Month to View:", options=all_months)
                        selected_month_num = pd.to_datetime(selected_month_str, format='%B').month
                        filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].copy()
                        if filtered_df.empty: st.info("No data for the selected month and year.")
                        else:
                            df_first_half = filtered_df[filtered_df['date'].dt.day <= 15]; df_second_half = filtered_df[filtered_df['date'].dt.day > 15]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("##### Days 1-15"); st.markdown("---")
                                for index, row in df_first_half.iterrows(): render_historical_record(row, db)
                            with col2:
                                st.markdown("##### Days 16-31"); st.markdown("---")
                                for index, row in df_second_half.iterrows(): render_historical_record(row, db)
                    else: st.write(f"No data available for the year {selected_year}.")
                else: st.write("No historical data to display.")
            else: st.write("No historical data to display.")

        if st.session_state['access_level'] == 1:
            with tabs[6]:
                st.subheader("User Management")
                with st.expander("Add New User"):
                    with st.form("new_user_form", clear_on_submit=True):
                        new_username = st.text_input("Username"); new_password = st.text_input("Password", type="password")
                        new_access_level = st.selectbox("Access Level", [1, 2, 3])
                        if st.form_submit_button("Add User"):
                            if get_user(db, new_username): st.error("User already exists")
                            else:
                                hashed_password = hash_password(new_password)
                                db.collection('users').add({'username': new_username, 'password': hashed_password.decode('utf-8'), 'access_level': new_access_level})
                                st.success("User added successfully"); st.rerun()
                users_df = load_from_firestore(db, 'users')
                if not users_df.empty:
                    st.write("Existing Users")
                    for index, row in users_df.iterrows():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.write(row['username'])
                        with col2: st.write(f"Access Level: {row['access_level']}")
                        with col3:
                            if st.button("Edit", key=f"edit_{row['doc_id']}"):
                                with st.form(f"edit_user_{row['doc_id']}"):
                                    new_level = st.selectbox("New Access Level", [1, 2, 3], index=row['access_level'] - 1)
                                    if st.form_submit_button("Update"):
                                        db.collection('users').document(row['doc_id']).update({'access_level': new_level}); st.success("User updated"); st.rerun()
                        with col4:
                            if st.button("Delete", key=f"delete_{row['doc_id']}"):
                                db.collection('users').document(row['doc_id']).delete(); st.success("User deleted"); st.rerun()
