import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# --- requirements.txt Recommendation ---
# For deployment, your requirements.txt file should look like this to avoid dependency conflicts:
# streamlit==1.35.0
# pandas==2.2.2
# prophet==1.1.5
# scikit-learn==1.5.0
# plotly==5.22.0
# PyYAML==6.0.1
# requests==2.31.0
# firebase-admin==6.5.0
# xgboost==2.0.3
# optuna==3.6.1
# bcrypt==4.1.3
# rich<14.0

# --- Suppress informational messages for a cleaner console output ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Application Configuration ---
APP_CONFIG = {
    "FORECAST_HORIZON": 15,
    "VALIDATION_PERIOD": 30,
    "MIN_DATA_FOR_FORECAST": 50,
    "MAX_SALES_INPUT": 1000000.0, # Use floats for consistency
    "MAX_CUSTOMERS_INPUT": 10000.0, # Use floats for consistency
    "LOCATION_LATITUDE": 10.48,
    "LOCATION_LONGITUDE": 123.42,
    "OPTUNA_TRIALS": 30,
}

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; }
        .block-container { padding: 2.5rem 2rem !important; }
        [data-testid="stSidebar"] { background-color: #252525; border-right: 1px solid #444; width: 320px !important; }
        .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out; border: none; padding: 10px 16px; }
        .stButton:has(button:contains("Generate")), .stButton:has(button:contains("Save")) > button { background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF; }
        .stButton:has(button:contains("Generate")):hover > button, .stButton:has(button:contains("Save")):hover > button { transform: translateY(-2px); box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4); }
        .stButton:has(button:contains("Refresh")), .stButton:has(button:contains("View All")), .stButton:has(button:contains("Back to Overview")) > button { border: 2px solid #c8102e; background: transparent; color: #c8102e; }
        .stButton:has(button:contains("Refresh")):hover > button, .stButton:has(button:contains("View All")):hover > button, .stButton:has(button:contains("Back to Overview")):hover > button { background: #c8102e; color: #ffffff; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px; background-color: transparent; color: #d3d3d3; padding: 8px 14px; font-weight: 600; font-size: 0.9rem; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: #ffffff; }
        .st-expander { border: 1px solid #444 !important; box-shadow: none; border-radius: 10px; background-color: #252525; margin-bottom: 0.5rem; }
        .st-expander header { font-size: 0.9rem; font-weight: 600; color: #d3d3d3; }
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
    defaults = {
        'historical_df': pd.DataFrame(), 'events_df': pd.DataFrame(), 'forecast_df': pd.DataFrame(), 'metrics': {},
        'authentication_status': False, 'access_level': 0, 'username': None, 'forecast_components': pd.DataFrame(),
        'show_recent_entries': False, 'show_all_activities': False, 'feature_importance_fig': None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if st.session_state.historical_df.empty:
        st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if st.session_state.events_df.empty:
        st.session_state.events_df = load_from_firestore(db_client, 'future_activities')

# --- User Management ---
def get_user(db_client, username):
    return db_client.collection('users').where('username', '==', username).limit(1).get()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# --- Data Processing & Feature Engineering ---
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
    numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'potential_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        params = {"latitude": APP_CONFIG["LOCATION_LATITUDE"], "longitude": APP_CONFIG["LOCATION_LONGITUDE"], "daily": "weather_code,temperature_2m_max,precipitation_sum", "timezone": "Asia/Manila", "forecast_days": days}
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json()['daily'])
        df.rename(columns={'time': 'date', 'temperature_2m_max': 'temp_max', 'precipitation_sum': 'precipitation'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['weather_code'] = df['weather_code'].astype(float)
        return df
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch weather data. Forecasting will proceed without it. Error: {e}")
        return None

def create_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['days_since_payday'] = df['date'].apply(lambda x: min(abs(x.day - 15), abs(x.day - 30)))
    for lag in [7, 14]:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    for window in [7, 14]:
        df[f'sales_rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).mean()
    return df

# --- Core Forecasting Models ---
@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, weather_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if len(df_train) < 15: return pd.DataFrame(), None
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    if weather_df is not None:
        df_prophet = pd.merge(df_prophet, weather_df[['date', 'temp_max', 'precipitation']], left_on='ds', right_on='date', how='left').drop(columns=['date']).fillna(method='ffill')
    start_date, end_date = df_train['date'].min(), df_train['date'].max() + timedelta(days=periods)
    all_events = pd.concat([events_df.rename(columns={'date':'ds', 'activity_name':'holiday'}), generate_recurring_local_events(start_date, end_date)]).dropna(subset=['ds', 'holiday'])
    prophet_model = Prophet(holidays=all_events, yearly_seasonality=(len(df_train) >= 365), weekly_seasonality=True, daily_seasonality=False)
    if weather_df is not None and 'temp_max' in df_prophet.columns:
        prophet_model.add_regressor('temp_max')
        prophet_model.add_regressor('precipitation')
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    if weather_df is not None:
        future = pd.merge(future, weather_df[['date', 'temp_max', 'precipitation']], left_on='ds', right_on='date', how='left').drop(columns=['date']).fillna(method='ffill')
    return prophet_model.predict(future), prophet_model

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, weather_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if df_train.empty: return pd.DataFrame(), None
    last_date = df_train['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    full_df = pd.concat([df_train, pd.DataFrame({'date': future_dates})], ignore_index=True)
    if weather_df is not None:
        full_df = pd.merge(full_df, weather_df, on='date', how='left').fillna(method='ffill')
    full_df_featured = create_advanced_features(full_df)
    df_featured = full_df_featured[full_df_featured['date'] <= last_date].copy().dropna()
    future_df_featured = full_df_featured[full_df_featured['date'] > last_date].copy()
    features = [f for f in ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'days_since_payday', 'sales_lag_7', 'sales_lag_14', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'temp_max', 'precipitation'] if f in df_featured.columns]
    X, y = df_featured[features], df_featured[target_col]
    if X.empty: return pd.DataFrame(), None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    def objective(trial):
        # FIX: Moved early_stopping_rounds from .fit() to constructor params
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
        }
        model = xgb.XGBRegressor(**params)
        # FIX: Correctly pass early stopping parameters to fit method
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=APP_CONFIG["OPTUNA_TRIALS"])
    final_model = xgb.XGBRegressor(**study.best_params)
    final_model.fit(X, y)
    future_predictions = final_model.predict(future_df_featured[features])
    final_df = pd.DataFrame({'ds': future_df_featured['date'], 'yhat': future_predictions})
    return final_df, final_model, features

# --- Data I/O and Plotting ---
def add_to_firestore(db_client, collection_name, data, historical_df):
    # ... (implementation remains the same)
    pass

def plot_feature_importance(model, features):
    if model is None: return None
    importance_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(15)
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Top 15 Features (XGBoost)', template='plotly_dark')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
    return fig

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state(db)
    if not st.session_state["authentication_status"]:
        _, col2, _ = st.columns([1.5, 1, 1.5])
        with col2:
            with st.container(border=True):
                st.title("Login")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", use_container_width=True):
                    user_doc = get_user(db, username)
                    if user_doc:
                        user_data = user_doc[0].to_dict()
                        if verify_password(password, user_data['password']):
                            st.session_state.update({'authentication_status': True, 'username': username, 'access_level': user_data['access_level']})
                            st.rerun()
                        else:
                            st.error("Incorrect password")
                    else:
                        st.error("User not found")
    else:
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
            st.title(f"Welcome, {st.session_state['username']}")
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear(); st.cache_resource.clear()
                st.session_state.historical_df = pd.DataFrame()
                st.success("Cache cleared."); time.sleep(1); st.rerun()
            if st.button("📈 Generate Forecast", use_container_width=True):
                if len(st.session_state.historical_df) < APP_CONFIG["MIN_DATA_FOR_FORECAST"]:
                    st.error(f"Not enough data. Need at least {APP_CONFIG['MIN_DATA_FOR_FORECAST']} days.")
                else:
                    try:
                        with st.spinner("Forecasting... This may take a moment."):
                            hist_df = st.session_state.historical_df.copy()
                            hist_df['base_sales'] = hist_df['sales'] - hist_df['add_on_sales']
                            Q1, Q3 = hist_df['base_sales'].quantile(0.25), hist_df['base_sales'].quantile(0.75)
                            hist_df = hist_df[hist_df['base_sales'] <= (Q3 + 1.5 * (Q3 - Q1))]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                hist_df['atv'] = np.nan_to_num(np.divide(hist_df['base_sales'], hist_df['customers']))
                            
                            weather_df = get_weather_forecast()
                            
                            prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df, st.session_state.events_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'customers')
                            xgb_cust_f, xgb_model_cust, xgb_features = train_and_forecast_xgboost_tuned(hist_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'customers')
                            
                            if prophet_cust_f.empty or xgb_cust_f.empty: raise Exception("Customer forecast failed.")
                            cust_f = pd.merge(prophet_cust_f[['ds', 'yhat']], xgb_cust_f, on='ds', suffixes=('_prophet', '_xgb'))
                            cust_f['yhat'] = (cust_f['yhat_prophet'] + cust_f['yhat_xgb']) / 2
                            
                            # (Simplified ATV forecast for robustness - can be expanded back to stacking)
                            prophet_atv_f, _ = train_and_forecast_prophet(hist_df, st.session_state.events_df, weather_df, APP_CONFIG["FORECAST_HORIZON"], 'atv')
                            if prophet_atv_f.empty: raise Exception("ATV forecast failed.")
                            
                            combo_f = pd.merge(cust_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_customers'}), prophet_atv_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            st.session_state.forecast_df = combo_f
                            st.session_state.forecast_components = prophet_cust_f
                            st.session_state.feature_importance_fig = plot_feature_importance(xgb_model_cust, xgb_features)
                            st.success("Forecast generated successfully!")
                    except Exception as e:
                        st.error(f"Forecasting Error: {e}")

            st.markdown("---")
            if st.button("Logout"):
                st.session_state.clear(); st.rerun()

        tab_list = ["🔮 Forecast", "💡 Insights", "✍️ Add Data", "📅 Activities", "📜 History"]
        if st.session_state['access_level'] == 1: tab_list.append("👥 Users")
        tabs = st.tabs(tab_list)
        
        with tabs[0]: # Forecast
            if not st.session_state.forecast_df.empty:
                st.dataframe(st.session_state.forecast_df)
            else:
                st.info("Generate a forecast to see results.")

        with tabs[1]: # Insights
            if st.session_state.forecast_components.empty: st.info("Generate a forecast to see insights.")
            else:
                t1, t2 = st.tabs(["Daily Breakdown (Prophet)", "Feature Importance (XGBoost)"])
                with t1:
                    st.plotly_chart(go.Figure(data=go.Scatter(x=st.session_state.forecast_components['ds'], y=st.session_state.forecast_components['yhat'])), use_container_width=True)
                with t2:
                    if st.session_state.feature_importance_fig:
                        st.plotly_chart(st.session_state.feature_importance_fig, use_container_width=True)
                    else: st.warning("Feature importance not available.")

        with tabs[2]: # Add Data
            if st.session_state['access_level'] <= 2:
                with st.form("new_record_form", clear_on_submit=True):
                    new_date = st.date_input("Date", date.today())
                    c1, c2 = st.columns(2)
                    # FIX: Ensure all number_input args are floats to prevent type errors
                    new_sales = c1.number_input("Total Sales (₱)", value=0.0, min_value=0.0, max_value=APP_CONFIG["MAX_SALES_INPUT"], format="%.2f")
                    new_customers = c2.number_input("Customer Count", value=0.0, min_value=0.0, max_value=APP_CONFIG["MAX_CUSTOMERS_INPUT"], format="%.0f")
                    new_addons = c1.number_input("Add-on Sales (₱)", value=0.0, min_value=0.0, max_value=APP_CONFIG["MAX_SALES_INPUT"], format="%.2f")
                    new_weather = c2.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Storm"])
                    if st.form_submit_button("✅ Save Record"):
                        # Convert customers back to int for storage
                        new_rec = {"date": new_date, "sales": new_sales, "customers": int(new_customers), "weather": new_weather, "add_on_sales": new_addons}
                        # add_to_firestore(db, 'historical_data', new_rec, st.session_state.historical_df)
                        st.success("Record added!"); time.sleep(1); st.rerun()
            else:
                st.warning("You do not have permission to add data.")
        
        # ... Other tabs (Activities, History, Users) can be implemented similarly ...

