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
import plotly.express as px
import io
import time
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import bcrypt
import requests

# --- Suppress informational messages from libraries for a cleaner console output ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Application Configuration ---
# Moved hardcoded values to a central config dictionary for easier management.
APP_CONFIG = {
    "FORECAST_HORIZON": 15,          # Days to forecast into the future
    "VALIDATION_PERIOD": 30,         # Days to use for training the stacking meta-model
    "MIN_DATA_FOR_FORECAST": 50,     # Minimum days of historical data required
    "MAX_SALES_INPUT": 1000000,      # Input validation: reasonable upper limit for daily sales
    "MAX_CUSTOMERS_INPUT": 10000,    # Input validation: reasonable upper limit for daily customers
    "LOCATION_LATITUDE": 10.48,      # Latitude for weather API
    "LOCATION_LONGITUDE": 123.42,    # Longitude for weather API
    "OPTUNA_TRIALS": 30,             # Reduced for faster tuning in a web app context
}


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
            # Securely load credentials from Streamlit Secrets
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
def initialize_state(db_client):
    """Initializes the session state with default values."""
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    
    defaults = {
        'historical_df': pd.DataFrame(), 
        'events_df': pd.DataFrame(),
        'forecast_df': pd.DataFrame(), 
        'metrics': {}, 
        'name': "Store 688", 
        'authentication_status': False,
        'access_level': 0,
        'username': None,
        'forecast_components': pd.DataFrame(), 
        'show_recent_entries': False,
        'show_all_activities': False,
        'trained_prophet_model': None, # NEW: Cache trained models
        'trained_xgb_model': None,     # NEW: Cache trained models
        'feature_importance_fig': None # NEW: Cache feature importance plot
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    
    # NEW: More efficient data loading. Only load if the dataframe is empty.
    if st.session_state.historical_df.empty:
        st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if st.session_state.events_df.empty:
        st.session_state.events_df = load_from_firestore(db_client, 'future_activities')

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
def load_from_firestore(_db_client, collection_name, max_records=None):
    """Loads data from Firestore, with an option to limit records for performance."""
    if _db_client is None: return pd.DataFrame()
    
    query = _db_client.collection(collection_name)
    if max_records:
        query = query.limit(max_records)
        
    docs = query.stream()
    records = [doc.to_dict() | {'doc_id': doc.id} for doc in docs]
        
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
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df = df[df[column] <= upper_bound].copy()
    return cleaned_df, len(df) - len(cleaned_df), upper_bound

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(df['base_sales'], df['customers'])
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    """Fetches weather forecast and handles potential API errors gracefully."""
    try:
        params = {
            "latitude": APP_CONFIG["LOCATION_LATITUDE"],
            "longitude": APP_CONFIG["LOCATION_LONGITUDE"],
            "daily": "weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max",
            "timezone": "Asia/Manila",
            "forecast_days": days,
        }
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])
        df.rename(columns={'time': 'date', 'temperature_2m_max': 'temp_max', 'precipitation_sum': 'precipitation', 'wind_speed_10m_max': 'wind_speed'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['weather_code'] = df['weather_code'].astype(float) # Ensure numeric for models
        return df
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch weather data. Forecasting will proceed without it. Error: {e}")
        return None

def generate_recurring_local_events(start_date, end_date):
    local_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [15, 30]:
            local_events.append({'holiday': 'Payday', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
            for i in range(1, 3):
                local_events.append({'holiday': 'Near_Payday', 'ds': current_date - timedelta(days=i), 'lower_window': 0, 'upper_window': 0})
        if current_date.month == 7 and current_date.day == 1:
            local_events.append({'holiday': 'San Carlos Charter Day', 'ds': current_date, 'lower_window': 0, 'upper_window': 0})
        current_date += timedelta(days=1)
    return pd.DataFrame(local_events)

# --- Core Forecasting Models ---

def create_advanced_features(df):
    """NEW: Enhanced feature engineering for XGBoost."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # NEW: Payday proximity features
    df['days_to_15th'] = abs(df['date'].dt.day - 15)
    df['days_to_30th'] = abs(df['date'].dt.day - 30)
    df['days_since_payday'] = df[['days_to_15th', 'days_to_30th']].min(axis=1)

    # Lag and rolling features
    for lag in [7, 14]:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        df[f'customers_lag_{lag}'] = df['customers'].shift(lag)
    
    for window in [7, 14]:
        df[f'sales_rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'customers_rolling_mean_{window}'] = df['customers'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'sales_rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).std()

    return df

@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, weather_df, periods, target_col):
    """NEW: Prophet model now incorporates weather data as an external regressor."""
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    
    if df_train.empty or len(df_train) < 15: return pd.DataFrame(), None

    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    # NEW: Merge weather data for training
    if weather_df is not None:
        df_prophet = pd.merge(df_prophet, weather_df[['date', 'temp_max', 'precipitation']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
        df_prophet.fillna(method='ffill', inplace=True) # Fill any missing weather data

    start_date, end_date = df_train['date'].min(), df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    manual_events = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_events = pd.concat([manual_events, recurring_events]).dropna(subset=['ds', 'holiday'])
    
    use_yearly = len(df_train) >= 365
    prophet_model = Prophet(
        holidays=all_events,
        yearly_seasonality=use_yearly, 
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # NEW: Add weather regressors if available
    if weather_df is not None and 'temp_max' in df_prophet.columns:
        prophet_model.add_regressor('temp_max')
        prophet_model.add_regressor('precipitation')

    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    
    # NEW: Add future weather data to the future dataframe
    if weather_df is not None:
        future = pd.merge(future, weather_df[['date', 'temp_max', 'precipitation']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
        future.fillna(method='ffill', inplace=True)

    forecast = prophet_model.predict(future)
    return forecast[['ds', 'yhat']], prophet_model

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, events_df, weather_df, periods, target_col):
    """NEW: XGBoost model now uses enhanced features and weather data."""
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    if df_train.empty: return pd.DataFrame(), None

    last_date = df_train['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    future_df_template = pd.DataFrame({'date': future_dates})

    full_df = pd.concat([df_train, future_df_template], ignore_index=True)
    
    # NEW: Merge weather data
    if weather_df is not None:
        full_df = pd.merge(full_df, weather_df, on='date', how='left')
        full_df[['temp_max', 'precipitation', 'wind_speed', 'weather_code']].fillna(method='ffill', inplace=True)

    full_df_featured = create_advanced_features(full_df)
    
    df_featured = full_df_featured[full_df_featured['date'] <= last_date].copy().dropna()
    future_df_featured = full_df_featured[full_df_featured['date'] > last_date].copy()
    
    features = [
        'dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'days_since_payday',
        'sales_lag_7', 'customers_lag_7', 'sales_lag_14', 'customers_lag_14',
        'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7',
        'sales_rolling_mean_14', 'customers_rolling_mean_14', 'sales_rolling_std_14',
        'temp_max', 'precipitation', 'wind_speed', 'weather_code' # NEW weather features
    ]
    features = [f for f in features if f in df_featured.columns]
    
    X, y = df_featured[features], df_featured[target_col]
    if X.empty or len(X) < 10: return pd.DataFrame(), None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=APP_CONFIG["OPTUNA_TRIALS"]) 

    final_model = xgb.XGBRegressor(**study.best_params)
    final_model.fit(X, y)

    future_predictions = final_model.predict(future_df_featured[features])
    
    final_df = pd.DataFrame({'ds': future_df_featured['date'], 'yhat': future_predictions})
    
    return final_df, final_model

def plot_feature_importance(model, features):
    """NEW: Creates a feature importance plot for the XGBoost model."""
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title='Top 15 Most Important Features (XGBoost)',
                 labels={'importance': 'Importance Score', 'feature': 'Feature'},
                 template='plotly_dark')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white'
    )
    return fig


# --- Firestore Data I/O ---
def add_to_firestore(db_client, collection_name, data, historical_df):
    if db_client is None: return
    
    if 'date' in data and pd.notna(data['date']):
        current_date = pd.to_datetime(data['date'])
        last_year_date = current_date - timedelta(days=364)
        
        hist_copy = historical_df.copy()
        hist_copy['date_only'] = pd.to_datetime(hist_copy['date']).dt.date
        
        last_year_record = hist_copy[hist_copy['date_only'] == last_year_date.date()]
        
        data['last_year_sales'] = float(last_year_record['sales'].iloc[0]) if not last_year_record.empty else 0.0
        data['last_year_customers'] = float(last_year_record['customers'].iloc[0]) if not last_year_record.empty else 0.0
        data['date'] = current_date.to_pydatetime()
    else: 
        return

    for col in ['sales', 'customers', 'add_on_sales']:
        data[col] = float(pd.to_numeric(data.get(col), errors='coerce'))
    
    # NEW: Placeholder for audit trail
    # db_client.collection('audit_log').add({'user': st.session_state.username, 'action': 'add', 'collection': collection_name, 'timestamp': firestore.SERVER_TIMESTAMP, 'data': data})
    db_client.collection(collection_name).add(data)

def update_historical_record_in_firestore(db_client, doc_id, data):
    if db_client: db_client.collection('historical_data').document(doc_id).update(data)

def update_activity_in_firestore(db_client, doc_id, data):
    if db_client: 
        data['potential_sales'] = float(data['potential_sales'])
        db_client.collection('future_activities').document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client: db_client.collection(collection_name).document(doc_id).delete()

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

# --- Plotting and UI Functions ---
# (plot_full_comparison_chart, plot_forecast_breakdown, etc. remain largely the same, minor style tweaks)
def plot_full_comparison_chart(hist,fcst,metrics,target):
    fig=go.Figure();fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')));fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')));title_text=f"{target.replace('_',' ').title()} Forecast";y_axis_title=title_text+' (₱)'if'atv'in target or'sales'in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical',xaxis_title='Date',yaxis_title=y_axis_title,legend=dict(x=0.01,y=0.99),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)");return fig

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']):
        x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']):
        x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    # NEW: Show regressor effects if they exist
    if 'temp_max' in day_data and pd.notna(day_data['temp_max']):
        x_data.append('Temperature Effect');y_data.append(day_data['temp_max']);measure_data.append('relative')
    if 'precipitation' in day_data and pd.notna(day_data['precipitation']):
        x_data.append('Precipitation Effect');y_data.append(day_data['precipitation']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']):
        holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"
        x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')
    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

# (Other plotting and UI functions like create_daily_evaluation_data, calculate_accuracy_metrics, etc. remain the same)
def create_daily_evaluation_data(historical_df, forecast_df):
    if forecast_df.empty or historical_df.empty: return pd.DataFrame()
    hist_eval = historical_df[['date', 'sales', 'customers', 'add_on_sales']].copy()
    hist_eval['date'] = pd.to_datetime(hist_eval['date'])
    hist_eval.rename(columns={'sales': 'actual_sales', 'customers': 'actual_customers'}, inplace=True)
    fcst_eval = forecast_df[['ds', 'forecast_sales', 'forecast_customers']].copy()
    fcst_eval['ds'] = pd.to_datetime(fcst_eval['ds'])
    eval_df = pd.merge(hist_eval, fcst_eval, left_on='date', right_on='ds', how='inner')
    if eval_df.empty: return pd.DataFrame()
    eval_df.drop(columns=['ds'], inplace=True)
    eval_df['adjusted_forecast_sales'] = eval_df['forecast_sales'] + eval_df['add_on_sales']
    return eval_df

def calculate_accuracy_metrics(historical_df, forecast_df):
    eval_df = create_daily_evaluation_data(historical_df, forecast_df)
    if eval_df.empty: return None
    eval_df_30_days = eval_df[eval_df['date'] >= pd.to_datetime('today').normalize() - pd.Timedelta(days=30)].copy()
    if eval_df_30_days.empty: return None
    actual_s, forecast_s = eval_df_30_days['actual_sales'], eval_df_30_days['adjusted_forecast_sales']
    non_zero_mask_s = actual_s != 0
    sales_mape = np.mean(np.abs((actual_s[non_zero_mask_s] - forecast_s[non_zero_mask_s]) / actual_s[non_zero_mask_s])) * 100 if non_zero_mask_s.any() else float('inf')
    actual_c, forecast_c = eval_df_30_days['actual_customers'], eval_df_30_days['forecast_customers']
    non_zero_mask_c = actual_c != 0
    customer_mape = np.mean(np.abs((actual_c[non_zero_mask_c] - forecast_c[non_zero_mask_c]) / actual_c[non_zero_mask_c])) * 100 if non_zero_mask_c.any() else float('inf')
    return {
        "sales_accuracy": 100 - sales_mape, "sales_mae": mean_absolute_error(actual_s, forecast_s),
        "customer_accuracy": 100 - customer_mape, "customer_mae": mean_absolute_error(actual_c, forecast_c),
    }

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty: return go.Figure().update_layout(title=title, paper_bgcolor='white', plot_bgcolor='white', annotations=[{"text": "No data available.", "showarrow": False}])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash')))
    fig.update_layout(title=dict(text=title, font=dict(color='black', size=20)), xaxis_title='Date', yaxis_title=y_axis_title, paper_bgcolor='white', plot_bgcolor='white', font_color='black')
    return fig
    
# (render_activity_card and render_historical_record remain largely the same)
def render_activity_card(row, db_client, view_type='compact_list', access_level=3):
    doc_id = row['doc_id']
    date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y')
    status = row['remarks']
    if status == 'Confirmed': color = '#22C55E'
    elif status == 'Needs Follow-up': color = '#F59E0B'
    else: color = '#EF4444'

    with st.expander(f"**{date_str}** | {row['activity_name']}"):
        st.markdown(f"**Status:** <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
        st.markdown(f"**Potential Sales:** ₱{row['potential_sales']:,.2f}")
        if access_level <= 2:
            with st.form(key=f"update_form_{doc_id}", border=False):
                updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"sales_{doc_id}")
                updated_remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"], index=["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"].index(status), key=f"remarks_{doc_id}")
                c1, c2 = st.columns(2)
                if c1.form_submit_button("💾 Update", use_container_width=True):
                    update_activity_in_firestore(db_client, doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks})
                    st.success("Activity updated!"); time.sleep(1); st.rerun()
                if c2.form_submit_button("🗑️ Delete", use_container_width=True):
                    delete_from_firestore(db_client, 'future_activities', doc_id)
                    st.warning("Activity deleted."); time.sleep(1); st.rerun()

def render_historical_record(row, db_client):
    date_str = row['date'].strftime('%B %d, %Y')
    with st.expander(f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f} | **Weather:** {row.get('weather', 'N/A')}")
        if st.session_state['access_level'] <= 2:
            with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
                c1, c2 = st.columns(2)
                updated_sales = c1.number_input("Sales (₱)", value=float(row.get('sales', 0)), max_value=APP_CONFIG["MAX_SALES_INPUT"], key=f"sales_{row['doc_id']}")
                updated_customers = c2.number_input("Customers", value=int(row.get('customers', 0)), max_value=APP_CONFIG["MAX_CUSTOMERS_INPUT"], key=f"cust_{row['doc_id']}")
                c3, c4 = st.columns(2)
                updated_addons = c3.number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales', 0)), key=f"addon_{row['doc_id']}")
                updated_weather = c4.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Storm"], index=["Sunny", "Cloudy", "Rainy", "Storm"].index(row['weather']) if row.get('weather') in ["Sunny", "Cloudy", "Rainy", "Storm"] else 0, key=f"weather_{row['doc_id']}")
                b1, b2 = st.columns(2)
                if b1.form_submit_button("💾 Update Record", use_container_width=True):
                    update_historical_record_in_firestore(db_client, row['doc_id'], {'sales': updated_sales, 'customers': updated_customers, 'add_on_sales': updated_addons, 'weather': updated_weather})
                    st.success(f"Record for {date_str} updated!"); time.sleep(1); st.rerun()
                if b2.form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                    delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                    st.warning(f"Record for {date_str} deleted."); time.sleep(1); st.rerun()
                    
# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state(db)
    
    if not st.session_state["authentication_status"]:
        st.markdown("<style>div[data-testid='stHorizontalBlock'] { margin-top: 5%; }</style>", unsafe_allow_html=True)
        _, col2, _ = st.columns([1.5, 1, 1.5])
        with col2:
            with st.container(border=True):
                st.title("Login")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", use_container_width=True):
                    user = get_user(db, username)
                    if user and verify_password(password, user.to_dict()['password']):
                        st.session_state.update({'authentication_status': True, 'username': username, 'access_level': user.to_dict()['access_level']})
                        st.rerun()
                    else:
                        st.error("Incorrect username or password")
    
    else: # --- MAIN APP AFTER LOGIN ---
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
            st.title(f"Welcome, *{st.session_state['username']}*")
            st.markdown("---")
            st.info("Hybrid Ensemble Forecasting")

            if st.button("🔄 Refresh Data from Firestore"):
                st.cache_data.clear(); st.cache_resource.clear()
                st.session_state.historical_df = pd.DataFrame() # Force reload
                st.success("Cache cleared. Rerunning..."); time.sleep(1); st.rerun()

            if st.button("📈 Generate Forecast", use_container_width=True):
                if len(st.session_state.historical_df) < APP_CONFIG["MIN_DATA_FOR_FORECAST"]:
                    st.error(f"Please provide at least {APP_CONFIG['MIN_DATA_FOR_FORECAST']} days of data.")
                else:
                    try:
                        with st.spinner("🧠 Initializing Hybrid Forecast..."):
                            base_df = st.session_state.historical_df.copy()
                            base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                            cleaned_df, removed_count, _ = remove_outliers_iqr(base_df, column='base_sales')
                            if removed_count > 0: st.warning(f"Removed {removed_count} outlier day(s).")

                            hist_df_with_atv = calculate_atv(cleaned_df)
                            ev_df = st.session_state.events_df.copy()
                            
                            with st.spinner("🛰️ Fetching live weather..."):
                                weather_df = get_weather_forecast()

                            # --- CUSTOMER FORECAST ---
                            with st.spinner("Forecasting Customers..."):
                                prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_with_atv, ev_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'customers')
                                xgb_cust_f, xgb_model_cust = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'customers')
                                if prophet_cust_f.empty or xgb_cust_f.empty: raise Exception("Customer forecast generation failed.")
                                cust_f = pd.merge(prophet_cust_f, xgb_cust_f, on='ds', suffixes=('_prophet', '_xgb'))
                                cust_f['yhat'] = (cust_f['yhat_prophet'] + cust_f['yhat_xgb']) / 2

                            # --- ATV FORECAST (Stacking) ---
                            with st.spinner("Forecasting Average Sale (Stacking)..."):
                                train_df = hist_df_with_atv.iloc[:-APP_CONFIG["VALIDATION_PERIOD"]]
                                validation_df = hist_df_with_atv.iloc[-APP_CONFIG["VALIDATION_PERIOD"]:]
                                prophet_atv_val, _ = train_and_forecast_prophet(train_df, ev_df, weather_df, APP_CONFIG["VALIDATION_PERIOD"], 'atv')
                                xgb_atv_val, _ = train_and_forecast_xgboost_tuned(train_df, ev_df, weather_df, APP_CONFIG["VALIDATION_PERIOD"], 'atv')
                                if prophet_atv_val.empty or xgb_atv_val.empty: raise Exception("ATV validation forecast failed.")
                                
                                validation_preds = pd.merge(prophet_atv_val[['ds', 'yhat']], xgb_atv_val[['ds', 'yhat']], on='ds', suffixes=('_prophet', '_xgb'))
                                validation_data = pd.merge(validation_preds, validation_df[['date', 'atv']], left_on='ds', right_on='date')
                                meta_model_atv = LinearRegression().fit(validation_data[['yhat_prophet', 'yhat_xgb']], validation_data['atv'])
                                
                                prophet_atv_f, _ = train_and_forecast_prophet(hist_df_with_atv, ev_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'atv')
                                xgb_atv_f, _ = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'atv')
                                if prophet_atv_f.empty or xgb_atv_f.empty: raise Exception("Final ATV forecast failed.")
                                atv_f = pd.merge(prophet_atv_f, xgb_atv_f, on='ds', suffixes=('_prophet', '_xgb'))
                                atv_f['yhat'] = meta_model_atv.predict(atv_f[['yhat_prophet', 'yhat_xgb']])

                            # --- FINAL COMBINATION ---
                            combo_f = pd.merge(cust_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_customers'}), atv_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            if weather_df is not None:
                                combo_f = pd.merge(combo_f, weather_df[['date', 'weather_code']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                                # Simple mapping for display
                                combo_f['weather'] = combo_f['weather_code'].apply(lambda x: "Rainy" if x > 50 else "Cloudy" if x > 1 else "Sunny")
                            
                            st.session_state.forecast_df = combo_f
                            if prophet_model_cust:
                                st.session_state.forecast_components = prophet_model_cust.predict(prophet_cust_f[['ds']])
                                st.session_state.all_holidays = prophet_model_cust.holidays
                            if xgb_model_cust:
                                st.session_state.feature_importance_fig = plot_feature_importance(xgb_model_cust, [f for f in create_advanced_features(pd.DataFrame()).columns if f in xgb_model_cust.get_booster().feature_names])
                            
                            st.success("Hybrid forecast generated successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")

            st.markdown("---")
            st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
            st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)
            if st.button("Logout"):
                st.session_state.clear(); st.rerun()
        
        # --- REFACTORED TABS ---
        tab_list = ["🔮 Forecast", "💡 Insights", "📊 Evaluator", "✍️ Add Data", "📅 Activities", "📜 History"]
        if st.session_state['access_level'] == 1: tab_list.append("👥 Users")
        
        tabs = st.tabs(tab_list)
        
        with tabs[0]: # Forecast Dashboard
            if not st.session_state.forecast_df.empty:
                future_forecast_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'] >= pd.to_datetime('today').normalize()].copy()
                if future_forecast_df.empty: st.warning("No future dates in forecast.")
                else:
                    display_df = future_forecast_df.rename(columns={'ds':'Date', 'forecast_customers':'Pred. Customers', 'forecast_atv':'Pred. Avg Sale (₱)', 'forecast_sales':'Pred. Sales (₱)', 'weather':'Pred. Weather'})
                    st.dataframe(display_df[['Date', 'Pred. Customers', 'Pred. Avg Sale (₱)', 'Pred. Sales (₱)', 'Pred. Weather']].set_index('Date').style.format({'Pred. Customers':'{:,.0f}', 'Pred. Avg Sale (₱)':'₱{:,.2f}', 'Pred. Sales (₱)':'₱{:,.2f}'}), height=560)
            else: st.info("Click 'Generate Forecast' to begin.")

        with tabs[1]: # Forecast Insights
            st.info("Breakdown from Prophet model, Feature Importance from XGBoost model.")
            if st.session_state.forecast_components.empty: st.info("Generate a forecast to see insights.")
            else:
                t1, t2 = st.tabs(["Daily Breakdown", "Feature Importance"])
                with t1:
                    future_components = st.session_state.forecast_components[st.session_state.forecast_components['ds'] >= pd.to_datetime('today').normalize()].copy()
                    if not future_components.empty:
                        selected_date_str = st.selectbox("Select a day to analyze:", options=future_components['ds'].dt.strftime('%A, %B %d, %Y'))
                        selected_date = pd.to_datetime(selected_date_str)
                        breakdown_fig, _ = plot_forecast_breakdown(st.session_state.forecast_components, selected_date, st.session_state.all_holidays)
                        st.plotly_chart(breakdown_fig, use_container_width=True)
                with t2:
                    if st.session_state.feature_importance_fig:
                        st.plotly_chart(st.session_state.feature_importance_fig, use_container_width=True)
                    else: st.warning("Feature importance plot not available.")

        with tabs[2]: # Forecast Evaluator
            st.header("📈 Forecast Performance Evaluator (Last 30 Days)")
            metrics = calculate_accuracy_metrics(st.session_state.historical_df, st.session_state.forecast_df)
            if metrics:
                c1, c2 = st.columns(2)
                c1.metric("Sales Accuracy (MAPE)", f"{metrics['sales_accuracy']:.2f}%")
                c1.metric("Sales Avg Error (MAE)", f"₱{metrics['sales_mae']:,.2f}")
                c2.metric("Customer Accuracy (MAPE)", f"{metrics['customer_accuracy']:.2f}%")
                c2.metric("Customer Avg Error (MAE)", f"{int(round(metrics['customer_mae']))} customers")
            else: st.warning("Not enough data to calculate metrics.")
            
        with tabs[3]: # Add/Edit Data
            if st.session_state['access_level'] <= 2:
                t1, t2 = st.tabs(["Add Single Record", "Bulk Upload from CSV"])
                with t1:
                    with st.form("new_record_form", clear_on_submit=True):
                        new_date=st.date_input("Date", date.today())
                        c1, c2 = st.columns(2)
                        new_sales=c1.number_input("Total Sales (₱)", min_value=0.0, max_value=APP_CONFIG["MAX_SALES_INPUT"], format="%.2f")
                        new_customers=c2.number_input("Customer Count", min_value=0, max_value=APP_CONFIG["MAX_CUSTOMERS_INPUT"])
                        new_addons=c1.number_input("Add-on Sales (₱)", min_value=0.0, format="%.2f")
                        new_weather=c2.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Storm"])
                        if st.form_submit_button("✅ Save Record"):
                            add_to_firestore(db, 'historical_data', {"date":new_date, "sales":new_sales, "customers":new_customers, "weather":new_weather, "add_on_sales":new_addons}, st.session_state.historical_df)
                            st.success("Record added!"); time.sleep(1); st.rerun()
                with t2:
                    st.info("Upload a CSV file with columns: date, sales, customers, add_on_sales, weather")
                    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                    if uploaded_file:
                        try:
                            upload_df = pd.read_csv(uploaded_file)
                            # Basic validation
                            required_cols = ['date', 'sales', 'customers']
                            if not all(col in upload_df.columns for col in required_cols):
                                st.error(f"CSV must contain at least these columns: {', '.join(required_cols)}")
                            else:
                                st.write("Data Preview:")
                                st.dataframe(upload_df.head())
                                if st.button("Upload to Firestore"):
                                    with st.spinner("Uploading records..."):
                                        for _, row in upload_df.iterrows():
                                            add_to_firestore(db, 'historical_data', row.to_dict(), st.session_state.historical_df)
                                    st.success(f"Successfully uploaded {len(upload_df)} records.")
                                    time.sleep(1); st.rerun()
                        except Exception as e:
                            st.error(f"Error processing file: {e}")
            else: st.warning("You do not have permission to add data.")
            
        with tabs[4]: # Future Activities
            if st.session_state['access_level'] <= 3:
                with st.expander("Add New Activity", expanded=False):
                     with st.form("new_activity_form", clear_on_submit=True):
                        name = st.text_input("Activity Name")
                        act_date = st.date_input("Date", min_value=date.today())
                        sales = st.number_input("Potential Sales (₱)", min_value=0.0)
                        remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
                        if st.form_submit_button("Save Activity"):
                            db.collection('future_activities').add({"activity_name": name, "date": pd.to_datetime(act_date), "potential_sales": sales, "remarks": remarks})
                            st.success("Activity saved!"); time.sleep(1); st.rerun()
                st.markdown("---")
                st.subheader("Upcoming Activities")
                upcoming_df = st.session_state.events_df[pd.to_datetime(st.session_state.events_df['date']).dt.date >= date.today()].copy().head(10)
                if upcoming_df.empty: st.info("No upcoming activities.")
                else:
                    for _, row in upcoming_df.iterrows():
                        render_activity_card(row, db, access_level=st.session_state['access_level'])
            else: st.warning("You do not have permission to view or add activities.")

        with tabs[5]: # Historical Data
            st.subheader("View & Edit Historical Data")
            df = st.session_state.historical_df.copy()
            if not df.empty:
                c1, c2 = st.columns(2)
                years = sorted(df['date'].dt.year.unique(), reverse=True)
                selected_year = c1.selectbox("Year:", options=years)
                months = sorted(df[df['date'].dt.year == selected_year]['date'].dt.strftime('%B').unique(), key=lambda m: list(pd.to_datetime(m, format='%B').month), reverse=True)
                selected_month = c2.selectbox("Month:", options=months)
                month_num = pd.to_datetime(selected_month, format='%B').month
                filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == month_num)].sort_values('date')
                for _, row in filtered_df.iterrows():
                    render_historical_record(row, db)

        if st.session_state['access_level'] == 1:
            with tabs[6]: # User Management
                st.subheader("User Management")
                with st.expander("Add New User"):
                    with st.form("new_user_form", clear_on_submit=True):
                        username = st.text_input("Username")
                        password = st.text_input("Password", type="password")
                        level = st.selectbox("Access Level", [1, 2, 3], format_func=lambda x: f"Level {x} ({'Admin' if x==1 else 'Manager' if x==2 else 'Viewer'})")
                        if st.form_submit_button("Add User"):
                            if get_user(db, username): st.error("User already exists.")
                            else:
                                db.collection('users').add({'username': username, 'password': hash_password(password).decode('utf-8'), 'access_level': level})
                                st.success("User added."); time.sleep(1); st.rerun()
                st.markdown("---")
                users_df = load_from_firestore(db, 'users')
                if not users_df.empty:
                    st.write("Existing Users")
                    for _, row in users_df.iterrows():
                        c1, c2, c3 = st.columns([2,2,1])
                        c1.write(row['username'])
                        c2.write(f"Access Level: {row['access_level']}")
                        if c3.button("Delete", key=f"delete_{row['doc_id']}", type="primary"):
                            if row['username'] != st.session_state.username: # Prevent self-deletion
                                delete_from_firestore(db, 'users', row['doc_id'])
                                st.success(f"User {row['username']} deleted."); time.sleep(1); st.rerun()
                            else:
                                st.warning("Cannot delete the currently logged-in user.")

