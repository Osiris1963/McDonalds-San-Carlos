import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
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
from datetime import timedelta, date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
import json
import logging

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

# --- Firestore Initialization & Data Operations ---
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

def add_to_firestore(db_client, collection_name, record, existing_df):
    record['date'] = pd.to_datetime(record['date']).to_pydatetime()
    if not existing_df.empty and record['date'].date() in pd.to_datetime(existing_df['date']).dt.date.values:
        st.error(f"A record for {record['date'].strftime('%Y-%m-%d')} already exists. Please edit the existing record.")
        return
    try:
        db_client.collection(collection_name).add(record)
        st.success("Record added!")
    except Exception as e:
        st.error(f"Error adding record to Firestore: {e}")

def update_activity_in_firestore(db_client, doc_id, update_data):
    try:
        db_client.collection('future_activities').document(doc_id).update(update_data)
        st.success("Activity updated!")
    except Exception as e:
        st.error(f"Error updating activity: {e}")

def update_historical_record_in_firestore(db_client, doc_id, update_data):
    try:
        db_client.collection('historical_data').document(doc_id).update(update_data)
        st.success("Record updated!")
    except Exception as e:
        st.error(f"Error updating historical record: {e}")

def delete_from_firestore(db_client, collection_name, doc_id):
    try:
        db_client.collection(collection_name).document(doc_id).delete()
        st.warning("Record deleted.")
    except Exception as e:
        st.error(f"Error deleting record: {e}")

# --- App State Management ---
def initialize_state_firestore(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {
        'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688",
        'authentication_status': True, 'access_level': 1, 'username': "Admin",
        'forecast_components': pd.DataFrame(), 'show_recent_entries': False,
        'show_all_activities': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="1h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    
    docs = _db_client.collection(collection_name).stream()
    records = [doc.to_dict() | {'doc_id': doc.id} for doc in docs]
        
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    numeric_cols = ['sales', 'customers', 'add_on_sales', 'potential_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

def cap_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25); Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1; upper_bound = Q3 + 1.5 * IQR
    capped_count = (df[column] > upper_bound).sum()
    df_capped = df.copy()
    df_capped.loc[df_capped[column] > upper_bound, column] = upper_bound
    return df_capped, capped_count, upper_bound

@st.cache_data
def calculate_atv(df):
    df_copy = df.copy()
    df_copy['base_sales'] = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(df_copy['base_sales'], df_copy['customers'])
    df_copy['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df_copy

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        params={
            "latitude":10.48, "longitude":123.42, "daily":"weather_code",
            "timezone":"Asia/Manila", "forecast_days":days }
        response=requests.get("https://api.open-meteo.com/v1/forecast",params=params)
        response.raise_for_status(); data=response.json(); df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather']=df['weather_code'].apply(map_weather_code)
        return df[['date', 'weather']]
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch weather data. Error: {e}")
        return None

def map_weather_code(code):
    if code in [0, 1]: return "Sunny"
    if code == 2: return "Partly Cloudy"
    if code == 3: return "Cloudy"
    if code in [45, 48]: return "Foggy"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "Rainy"
    if code in [95, 96, 99]: return "Thunderstorm"
    return "Cloudy"

def generate_recurring_local_events(start_date, end_date):
    events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [15, 30]:
            events.append({'holiday': 'Payday', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
            for i in range(1, 3):
                events.append({'holiday': 'Near_Payday', 'ds': current_date - timedelta(days=i), 'lower_window': 0, 'upper_window': 0})
        if current_date.month == 7 and current_date.day == 1:
            events.append({'holiday': 'San Carlos Charter Day', 'ds': current_date, 'lower_window': 0, 'upper_window': 0})
        current_date += timedelta(days=1)
    return pd.DataFrame(events)

# --- Self-Recalibration Logic ---
def check_performance_and_recalibrate(db, historical_df, degradation_threshold=0.98, short_term_days=7, long_term_days=30):
    try:
        log_docs = db.collection('forecast_log').stream()
        log_records = [doc.to_dict() for doc in log_docs]
        if not log_records: return False

        log_df = pd.DataFrame(log_records)
        log_df['forecast_for_date'] = pd.to_datetime(log_df['forecast_for_date']).dt.tz_localize(None)
        log_df['generated_on'] = pd.to_datetime(log_df['generated_on']).dt.tz_localize(None)
        day_ahead_logs = log_df[log_df['forecast_for_date'] - log_df['generated_on'] == timedelta(days=1)].copy()
        
        if day_ahead_logs.empty: return False

        merged_df = pd.merge(historical_df, day_ahead_logs, left_on='date', right_on='forecast_for_date', how='inner')
        if len(merged_df) < long_term_days: return False

        today = pd.to_datetime('today').normalize()
        long_term_df = merged_df[merged_df['date'] >= today - pd.Timedelta(days=long_term_days)]
        short_term_df = merged_df[merged_df['date'] >= today - pd.Timedelta(days=short_term_days)]

        if long_term_df.empty or short_term_df.empty: return False
        
        long_term_accuracy = 100 - (np.nanmean(np.abs((long_term_df['sales'] - long_term_df['predicted_sales']) / long_term_df['sales'].replace(0, np.nan))) * 100)
        short_term_accuracy = 100 - (np.nanmean(np.abs((short_term_df['sales'] - short_term_df['predicted_sales']) / short_term_df['sales'].replace(0, np.nan))) * 100)

        if short_term_accuracy < (long_term_accuracy * degradation_threshold):
            st.warning(f"🚨 Recent 7-day accuracy ({short_term_accuracy:.2f}%) dropped below 30-day baseline ({long_term_accuracy:.2f}%). Triggering model recalibration.")
            st.cache_resource.clear()
            st.cache_data.clear()
            time.sleep(2) 
            return True
    except Exception as e:
        st.warning(f"Could not perform automatic accuracy check. Error: {e}")
    return False

# --- Core Forecasting Models ---
@st.cache_data
def create_day_specific_features(df, target_col):
    df = df.sort_values('date')
    lags = [1, 2, 3, 4] 
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    windows = [2, 4]
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
    
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    
    return df

@st.cache_resource
def train_and_forecast_prophet(_historical_df, _events_df, periods, target_col):
    df_train = _historical_df.rename(columns={'date': 'ds', target_col: 'y'})
    if df_train.empty or len(df_train) < 15: return pd.DataFrame(), None

    start_date = df_train['ds'].min(); end_date = df_train['ds'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    manual_events = _events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_events = pd.concat([manual_events, recurring_events]).dropna(subset=['ds', 'holiday'])

    prophet_model = Prophet(
        growth='linear', holidays=all_events, daily_seasonality=False,
        weekly_seasonality=True, yearly_seasonality=(len(df_train) >= 365), 
        changepoint_prior_scale=0.5, changepoint_range=0.95,
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_train)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    return forecast[['ds', 'yhat']], prophet_model

@st.cache_resource(show_spinner=False)
def train_day_specific_tree_model(model_name, _df_day, target_col, periods):
    df_featured = create_day_specific_features(_df_day.copy(), target_col)
    
    features = [col for col in df_featured.columns if col not in ['date', 'doc_id', 'sales', 'customers', 'atv', 'base_sales', 'add_on_sales', 'weather']]
    
    X = df_featured[features].copy()
    y = df_featured[target_col].copy()
    
    X.dropna(inplace=True); y = y[X.index]
    if X.empty or len(X) < 10: return pd.DataFrame()

    params = {
        'xgb': {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42},
        'lgbm': {'random_state': 42, 'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000, 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 31},
        'cat': {'random_seed': 42, 'objective': 'RMSE', 'iterations': 1000, 'verbose': 0, 'learning_rate': 0.05, 'depth': 6}
    }
    
    model_map = {'xgb': xgb.XGBRegressor, 'lgbm': lgb.LGBMRegressor, 'cat': cat.CatBoostRegressor}
    model = model_map[model_name](**params[model_name])

    sample_weights = np.exp(np.linspace(-1, 0, len(y)))
    model.fit(X, y, sample_weight=sample_weights)
    
    future_predictions = []
    history_copy = df_featured.copy()
    for i in range(periods):
        last_date = history_copy['date'].max()
        next_date = last_date + timedelta(days=7)
        
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history_copy, future_step_df], ignore_index=True)
        extended_featured_df = create_day_specific_features(extended_history, target_col)
        
        X_future = extended_featured_df[features].tail(1)
        prediction = model.predict(X_future)[0]
        
        future_predictions.append({'ds': next_date, 'yhat': prediction})
        history_copy.loc[history_copy.index.max() + 1, 'date'] = next_date
        history_copy.loc[history_copy.index.max(), target_col] = prediction

    historical_preds = pd.DataFrame({'ds': df_featured.loc[X.index, 'date'], 'yhat': model.predict(X)})
    future_preds = pd.DataFrame(future_predictions)

    return pd.concat([historical_preds, future_preds], ignore_index=True)

@st.cache_resource
def train_and_forecast_stacked_ensemble(base_forecasts_dict, historical_target, target_col_name):
    final_df = None
    for name, fcst_df in base_forecasts_dict.items():
        if fcst_df is None or fcst_df.empty: continue
        renamed_df = fcst_df[['ds', 'yhat']].rename(columns={'yhat': f'yhat_{name}'})
        if final_df is None: final_df = renamed_df
        else: final_df = pd.merge(final_df, renamed_df, on='ds', how='outer')
    
    if final_df is None or final_df.empty: return pd.DataFrame()

    final_df = final_df.sort_values('ds').interpolate(method='linear', limit_direction='both', axis=0)
    final_df.bfill(inplace=True); final_df.ffill(inplace=True)

    training_data = pd.merge(final_df, historical_target[['date', target_col_name]], left_on='ds', right_on='date')
    
    meta_features = [col for col in training_data.columns if 'yhat_' in col]
    X_meta = training_data[meta_features]
    y_meta = training_data[target_col_name]
    
    if len(X_meta) < 20:
        st.warning(f"Not enough historical data for advanced stacking. Falling back to simple averaging.")
        final_df['yhat'] = final_df[meta_features].mean(axis=1)
        return final_df[['ds', 'yhat']]

    meta_model = lgb.LGBMRegressor(random_state=42, n_estimators=200, objective='regression_l1', verbosity=-1)
    meta_model.fit(X_meta, y_meta)
    
    X_future_meta = final_df[meta_features].dropna()
    final_df['yhat'] = meta_model.predict(X_future_meta)
    
    return final_df[['ds', 'yhat']]

# --- Plotting and UI ---
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']):
        x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']):
        x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']):
        holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"
        x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')

    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty: return go.Figure().update_layout(title=title, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white', annotations=[{"text": "No data available.", "xref": "paper", "yref": "paper", "showarrow": False}])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual',line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast',line=dict(color='#d62728', dash='dash')))
    fig.update_layout(
        title=dict(text=title), xaxis=dict(title='Date'), yaxis=dict(title=y_axis_title),
        legend=dict(font=dict(color='white')), height=450, margin=dict(l=50,r=50,t=80,b=50),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white',
        xaxis_gridcolor='#444', yaxis_gridcolor='#444'
    )
    return fig

def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data.get('weekly',0),'Time of Year':day_data.get('yearly',0),'Holidays/Events':day_data.get('holidays',0)}
    significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data.get('trend', 0):.0f} customers**.\n\n"
    if not significant_effects:
        summary += f"The final forecast of **{day_data.get('yhat', 0):.0f} customers** is driven primarily by this trend."
    else:
        pos_drivers={k:v for k,v in significant_effects.items()if v>0}
        neg_drivers={k:v for k,v in significant_effects.items()if v<0}
        if pos_drivers: summary+=f"📈 Main positive driver is **{max(pos_drivers,key=pos_drivers.get)}**, adding an estimated **{max(pos_drivers.values()):.0f} customers**.\n"
        if neg_drivers: summary+=f"📉 Main negative driver is **{min(neg_drivers,key=neg_drivers.get)}**, reducing by **{abs(min(neg_drivers.values())):.0f} customers**.\n"
    summary+=f"\nAfter all factors, the final forecast is **{day_data.get('yhat', 0):.0f} customers**."
    return summary

def render_activity_card(row, db_client, view_type='compact_list'):
    doc_id = row['doc_id']
    status_colors = {'Confirmed': '#22C55E', 'Needs Follow-up': '#F59E0B', 'Tentative': '#38BDF8', 'Cancelled': '#EF4444'}
    status = row.get('remarks', 'Tentative')
    color = status_colors.get(status, '#A9A9A9')
    
    if view_type == 'compact_list':
        with st.expander(f"**{pd.to_datetime(row['date']).strftime('%b %d')}** | {row['activity_name']}"):
            st.markdown(f"**Status:** <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            st.markdown(f"**Potential Sales:** ₱{row['potential_sales']:,.2f}")
            with st.form(key=f"compact_update_form_{doc_id}", border=False):
                status_options = list(status_colors.keys())
                updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"compact_sales_{doc_id}")
                updated_remarks = st.selectbox("Status", options=status_options, index=status_options.index(status), key=f"compact_remarks_{doc_id}")
                uc, dc = st.columns(2)
                if uc.form_submit_button("💾 Update", use_container_width=True):
                    update_activity_in_firestore(db_client, doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks})
                    st.cache_data.clear(); time.sleep(1); st.rerun()
                if dc.form_submit_button("🗑️ Delete", use_container_width=True):
                    delete_from_firestore(db_client, 'future_activities', doc_id)
                    st.cache_data.clear(); time.sleep(1); st.rerun()
    else: # 'grid' view
        with st.container(border=True):
            st.markdown(f"**{row['activity_name']}**")
            st.markdown(f"<small>📅 {pd.to_datetime(row['date']).strftime('%A, %B %d, %Y')}</small>", unsafe_allow_html=True)
            st.markdown(f"💰 ₱{row['potential_sales']:,.2f}")
            st.markdown(f"Status: <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            with st.expander("Edit / Manage"):
                with st.form(key=f"full_update_form_{doc_id}", border=False):
                    status_options = list(status_colors.keys())
                    updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"full_sales_{doc_id}")
                    updated_remarks = st.selectbox("Status", options=status_options, index=status_options.index(status), key=f"full_remarks_{doc_id}")
                    uc, dc = st.columns(2)
                    if uc.form_submit_button("💾 Update", use_container_width=True):
                        update_activity_in_firestore(db_client, doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks})
                        st.cache_data.clear(); time.sleep(1); st.rerun()
                    if dc.form_submit_button("🗑️ Delete", use_container_width=True):
                        delete_from_firestore(db_client, 'future_activities', doc_id)
                        st.cache_data.clear(); time.sleep(1); st.rerun()

def render_historical_record(row, db_client):
    date_str = pd.to_datetime(row['date']).strftime('%B %d, %Y')
    with st.expander(f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f} | **Weather:** {row.get('weather', 'N/A')}")

        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            c1, c2, c3 = st.columns(3)
            updated_sales = c1.number_input("Sales (₱)", value=float(row.get('sales', 0)), key=f"s_{row['doc_id']}")
            updated_customers = c2.number_input("Customers", value=int(row.get('customers', 0)), key=f"c_{row['doc_id']}")
            updated_addons = c3.number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales',0)), key=f"a_{row['doc_id']}")
            
            b1, b2 = st.columns(2)
            if b1.form_submit_button("💾 Update Record", use_container_width=True):
                update_data = { 'sales': updated_sales, 'customers': updated_customers, 'add_on_sales': updated_addons }
                update_historical_record_in_firestore(db_client, row['doc_id'], update_data)
                st.cache_data.clear(); time.sleep(1); st.rerun()

            if b2.form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                st.cache_data.clear(); time.sleep(1); st.rerun()

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['username']}*");st.markdown("---")
        st.info("Forecasting with an Advanced AI Ensemble")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.cache_resource.clear()
            st.success("Cache cleared. Rerunning to get latest data."); time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                if check_performance_and_recalibrate(db, st.session_state.historical_df):
                    st.info("Models recalibrated. Please click 'Generate Forecast' again.")
                else:
                    with st.spinner("🚀 Launching AI Ensemble Forecast..."):
                        hist_df = st.session_state.historical_df.copy()
                        ev_df = st.session_state.events_df.copy()
                        FORECAST_HORIZON = 15
                        
                        with st.spinner("Engineering features and handling outliers..."):
                            hist_df, _, _ = cap_outliers_iqr(hist_df, column='sales')
                            hist_df_atv = calculate_atv(hist_df)
                        
                        with st.spinner("Training Foundational Prophet Model..."):
                            prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_atv, ev_df, FORECAST_HORIZON, 'customers')
                            prophet_atv_f, _ = train_and_forecast_prophet(hist_df_atv, ev_df, FORECAST_HORIZON, 'atv')

                        all_atv_forecasts = {'prophet': prophet_atv_f}
                        all_cust_forecasts = {'prophet': prophet_cust_f}
                        
                        df_by_day = [hist_df_atv[hist_df_atv['date'].dt.dayofweek == i] for i in range(7)]
                        
                        for model_name in ['xgb', 'lgbm', 'cat']:
                            with st.spinner(f"Training {model_name.upper()} day-specific models..."):
                                atv_day_forecasts = []
                                cust_day_forecasts = []
                                for i in range(7):
                                    if len(df_by_day[i]) > 10:
                                        fcst_atv = train_day_specific_tree_model(model_name, df_by_day[i], 'atv', FORECAST_HORIZON)
                                        fcst_cust = train_day_specific_tree_model(model_name, df_by_day[i], 'customers', FORECAST_HORIZON)
                                        atv_day_forecasts.append(fcst_atv)
                                        cust_day_forecasts.append(fcst_cust)

                                all_atv_forecasts[model_name] = pd.concat(atv_day_forecasts, ignore_index=True) if atv_day_forecasts else pd.DataFrame()
                                all_cust_forecasts[model_name] = pd.concat(cust_day_forecasts, ignore_index=True) if cust_day_forecasts else pd.DataFrame()

                        with st.spinner("Stacking all models for final predictions..."):
                            atv_f = train_and_forecast_stacked_ensemble(all_atv_forecasts, hist_df_atv, 'atv')
                            cust_f = train_and_forecast_stacked_ensemble(all_cust_forecasts, hist_df_atv, 'customers')

                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            weather_df = get_weather_forecast()
                            if weather_df is not None:
                                combo_f = pd.merge(combo_f, weather_df, left_on='ds', right_on='date', how='left').drop(columns=['date'])
                            
                            st.session_state.forecast_df = combo_f
                            
                            try:
                                with st.spinner("📝 Saving forecast log..."):
                                    today_naive = pd.to_datetime('today').normalize()
                                    to_log = combo_f[combo_f['ds'] > today_naive]
                                    for _, row in to_log.iterrows():
                                        log_id = f"{today_naive.strftime('%Y-%m-%d')}_{row['ds'].strftime('%Y-%m-%d')}"
                                        log_entry = {"generated_on": today_naive, "forecast_for_date": row['ds'], "predicted_sales": row['forecast_sales'], "predicted_customers": row['forecast_customers']}
                                        db.collection('forecast_log').document(log_id).set(log_entry, merge=True)
                            except Exception as e:
                                st.error(f"Failed to save forecast log: {e}")
                            
                            if prophet_model_cust:
                                p_future = prophet_model_cust.make_future_dataframe(periods=FORECAST_HORIZON)
                                st.session_state.forecast_components = prophet_model_cust.predict(p_future)
                                st.session_state.all_holidays = prophet_model_cust.holidays
                            
                            st.success("✅ Advanced AI forecast generated successfully!")
                        else:
                            st.error("Forecast generation failed. Check model outputs.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data", "📅 Future Activities", "📜 Historical Data"]
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        if not st.session_state.forecast_df.empty:
            future_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'] >= pd.to_datetime('today').normalize()].copy()
            if not future_df.empty:
                disp_cols = {'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                display_df = future_df.rename(columns=disp_cols)[[col for col in disp_cols.values() if col in future_df.rename(columns=disp_cols).columns]]
                st.markdown("#### Forecasted Values"); st.dataframe(display_df.set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                
                fig=go.Figure(); fig.add_trace(go.Scatter(x=future_df['ds'],y=future_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c'))); fig.add_trace(go.Scatter(x=future_df['ds'],y=future_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e'))); fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(orientation='h'),height=500,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white'); st.plotly_chart(fig,use_container_width=True)
        else: st.info("Click the 'Generate Forecast' button to begin.")
    
    with tabs[1]:
        st.header("💡 Forecast Insights")
        st.info("This waterfall chart shows the foundational drivers from the Prophet model, such as overall trend and seasonal effects, before the final stacking process.")
        
        if st.session_state.forecast_components.empty:
            st.info("Click 'Generate Forecast' to see prediction drivers.")
        else:
            future_components = st.session_state.forecast_components[st.session_state.forecast_components['ds'] >= pd.to_datetime('today').normalize()].copy()
            if not future_components.empty:
                cust_final = st.session_state.forecast_df[['ds', 'forecast_customers']].rename(columns={'forecast_customers': 'final_yhat'})
                future_components = pd.merge(future_components, cust_final, on='ds', how='left')
                future_components['yhat'] = future_components['final_yhat'].fillna(future_components['yhat'])

                selected_date = st.selectbox("Select a day to analyze:", options=future_components['ds'], format_func=lambda d: d.strftime('%A, %B %d, %Y'))
                
                fig, day_data = plot_forecast_breakdown(future_components, selected_date, st.session_state.all_holidays)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.subheader("Prophet Insight Summary")
                st.markdown(generate_insight_summary(day_data, selected_date))

    with tabs[2]:
        st.header("📈 Forecast Evaluator")
        st.info("Compares actual results against the forecast generated the day before.")
        def render_true_accuracy_content(days):
            try:
                log_docs = db.collection('forecast_log').stream()
                log_records = [doc.to_dict() for doc in log_docs]
                if not log_records: raise ValueError("No forecast logs found.")

                log_df = pd.DataFrame(log_records)
                log_df['forecast_for_date'] = pd.to_datetime(log_df['forecast_for_date']).dt.tz_localize(None)
                log_df['generated_on'] = pd.to_datetime(log_df['generated_on']).dt.tz_localize(None)
                day_ahead_logs = log_df[log_df['forecast_for_date'] - log_df['generated_on'] == timedelta(days=1)].copy()
                if day_ahead_logs.empty: raise ValueError("Not enough consecutive logs.")

                merged_df = pd.merge(st.session_state.historical_df, day_ahead_logs, left_on='date', right_on='forecast_for_date', how='inner')
                if merged_df.empty: raise ValueError("No matching historical data for logs.")
                
                final_df = merged_df[merged_df['date'] >= pd.to_datetime('today').normalize() - pd.Timedelta(days=days)].copy()
                if final_df.empty: raise ValueError(f"No forecast data in the last {days} days.")

                st.subheader(f"Accuracy Metrics for the Last {days} Days")
                sales_mape = np.nanmean(np.abs((final_df['sales'] - final_df['predicted_sales']) / final_df['sales'].replace(0, np.nan))) * 100
                cust_mape = np.nanmean(np.abs((final_df['customers'] - final_df['predicted_customers']) / final_df['customers'].replace(0, np.nan))) * 100
                col1, col2 = st.columns(2)
                col1.metric("Sales Accuracy (MAPE)", f"{100 - sales_mape:.2f}%")
                col2.metric("Customer Accuracy (MAPE)", f"{100 - cust_mape:.2f}%")

                st.plotly_chart(plot_evaluation_graph(final_df, 'date', 'sales', 'predicted_sales', 'Actual vs. Forecasted Sales', 'Sales (₱)'), use_container_width=True)
                st.plotly_chart(plot_evaluation_graph(final_df, 'date', 'customers', 'predicted_customers', 'Actual vs. Forecasted Customers', 'Customers'), use_container_width=True)
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"An error occurred building the report: {e}")

        eval_tab_7, eval_tab_30 = st.tabs(["Last 7 Days", "Last 30 Days"])
        with eval_tab_7: render_true_accuracy_content(7)
        with eval_tab_30: render_true_accuracy_content(30)

    with tabs[3]:
        with st.form("new_record_form", clear_on_submit=True):
            st.subheader("✍️ Add New Daily Record")
            new_date=st.date_input("Date", date.today())
            c1, c2, c3 = st.columns(3)
            new_sales=c1.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
            new_customers=c2.number_input("Customer Count",min_value=0)
            new_addons = c3.number_input("Add-on Sales (₱)", min_value=0.0, format="%.2f", help="Sales from promotions or up-selling.")
            
            if st.form_submit_button("✅ Save Record", use_container_width=True):
                add_to_firestore(db, 'historical_data', { "date":new_date, "sales":new_sales, "customers":new_customers, "add_on_sales": new_addons, "weather": "Cloudy" }, st.session_state.historical_df)
                st.cache_data.clear(); time.sleep(1); st.rerun()

    with tabs[4]:
        c1, c2 = st.columns([1,2], gap="large")
        with c1, st.form("new_activity_form", clear_on_submit=True, border=True):
            st.markdown("##### Add New Activity")
            name = st.text_input("Activity/Event Name")
            act_date = st.date_input("Date of Activity", min_value=date.today())
            sales = st.number_input("Potential Sales (₱)", min_value=0.0, format="%.2f")
            remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
            if st.form_submit_button("✅ Save Activity", use_container_width=True):
                if name and act_date:
                    db.collection('future_activities').add({"activity_name": name, "date": pd.to_datetime(act_date), "potential_sales": float(sales), "remarks": remarks})
                    st.cache_data.clear(); time.sleep(1); st.rerun()
                else: st.warning("Name and date are required.")

        with c2:
            st.markdown("##### Upcoming Activities")
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
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            all_years = sorted(df['date'].dt.year.unique(), reverse=True)
            if all_years:
                c1, c2 = st.columns(2)
                sel_year = c1.selectbox("Select Year:", options=all_years)
                
                df_year = df[df['date'].dt.year == sel_year]
                all_months = sorted(df_year['date'].dt.strftime('%B').unique(), key=lambda m: datetime.strptime(m, '%B').month, reverse=True)
                if all_months:
                    sel_month_str = c2.selectbox("Select Month:", options=all_months)
                    sel_month_num = datetime.strptime(sel_month_str, '%B').month
                    
                    filtered_df = df[(df['date'].dt.year == sel_year) & (df['date'].dt.month == sel_month_num)].sort_values('date')
                    for _, row in filtered_df.iterrows():
                        render_historical_record(row, db)
