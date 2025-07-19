import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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
    
    # Events are no longer managed via UI, initialize as empty.
    if 'events_df' not in st.session_state: st.session_state.events_df = pd.DataFrame() 

    defaults = {
        'forecast_df': pd.DataFrame(),
        'metrics': {},
        'name': "Store 688",
        'migration_done': False,
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
    """Creates time series, lag, and rolling window features from a datetime index."""
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

    return df

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

    # Note: manual_events_renamed will likely be empty, which is handled correctly.
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
        changepoint_prior_scale=0.15,
        changepoint_range=0.9,
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

    if df_train.empty:
        return pd.DataFrame()

    # --- Part 1: Feature Engineering and Model Training (Largely the same) ---
    df_featured = create_advanced_features(df_train.copy())

    if 'day_type' in df_featured.columns:
        df_featured['is_not_normal_day'] = df_featured['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0)
    else:
        df_featured['is_not_normal_day'] = 0

    dropna_col = 'atv_lag_7' if target_col == 'atv' else 'sales_lag_7'
    if dropna_col in df_featured.columns:
        df_featured.dropna(subset=[dropna_col], inplace=True)

    base_features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'is_not_normal_day']

    if target_col == 'customers':
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7']
    elif target_col == 'atv':
        features = base_features + ['atv_lag_7', 'atv_rolling_mean_7', 'atv_rolling_std_7']
    else:
        features = base_features + ['sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7', 'customers_rolling_mean_7', 'sales_rolling_std_7']

    features = [f for f in features if f in df_featured.columns]
    target = target_col

    X = df_featured[features]
    y = df_featured[target]

    if X.empty or len(X) < 10:
        st.warning(f"Not enough data to train XGBoost for {target_col} after feature engineering.")
        return pd.DataFrame()

    # Using fixed parameters for speed, but you can re-enable Optuna if desired
    # For a production app, you might save the best_params to avoid re-tuning every time.
    best_params = {
        'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
    }

    final_model = xgb.XGBRegressor(**best_params)
    sample_weights = np.linspace(0.5, 1.0, len(y))
    final_model.fit(X, y, sample_weight=sample_weights)

    # --- Part 2: Recursive Forecasting Loop (The Fix) ---

    future_predictions = []
    # Start with the full history that was used for training
    history_with_features = df_featured.copy()

    for i in range(periods):
        # 1. Prepare the feature set for the next day
        # Get the last known date and add one day
        last_date = history_with_features['date'].max()
        next_date = last_date + timedelta(days=1)

        # Create a one-row dataframe for the next day to be predicted
        future_step_df = pd.DataFrame([{'date': next_date}])

        # Append this new day to our history
        extended_history = pd.concat([history_with_features, future_step_df], ignore_index=True)

        # 2. Re-create features for the entire extended history
        # The last row will now have correctly calculated lag and rolling features
        extended_featured_df = create_advanced_features(extended_history)

        # Add the 'is_not_normal_day' feature manually for the future date
        extended_featured_df['is_not_normal_day'] = extended_featured_df['day_type'].apply(lambda x: 1 if x == 'Not Normal Day' else 0).fillna(0)

        # 3. Predict the next step
        # Isolate the last row, which contains the features for the day we want to predict
        X_future = extended_featured_df[features].tail(1)
        prediction = final_model.predict(X_future)[0]

        # Store the prediction
        future_predictions.append({'ds': next_date, 'yhat': prediction})

        # 4. Update history with the prediction to be used in the next loop
        # This is the key step of the recursive strategy
        history_with_features = extended_featured_df.copy()
        history_with_features.loc[history_with_features.index.max(), target] = prediction

        # If forecasting customers, we also need to update the sales column to help with features
        # We can make a simple assumption, e.g., sales = predicted_customers * recent_atv
        if target_col == 'customers':
            recent_atv = history_with_features['atv'].dropna().tail(7).mean()
            history_with_features.loc[history_with_features.index.max(), 'sales'] = prediction * recent_atv

    # --- Part 3: Combine and Return ---
    # Create the final forecast dataframe
    final_df = pd.DataFrame(future_predictions)

    # Also include the model's prediction on the historical data (in-sample forecast)
    historical_predictions = df_featured[['date']].copy()
    historical_predictions.rename(columns={'date': 'ds'}, inplace=True)
    historical_predictions['yhat'] = final_model.predict(X)

    # Combine historical fit with future forecast
    full_forecast_df = pd.concat([historical_predictions, final_df], ignore_index=True)

    return full_forecast_df

# --- Plotting Functions & Data I/O ---
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist,fcst,metrics,target):
    fig=go.Figure();fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')));fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')));title_text=f"{target.replace('_',' ').title()} Forecast";y_axis_title=title_text+' (₱)'if'atv'in target or'sales'in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical',xaxis_title='Date',yaxis_title=y_axis_title,legend=dict(x=0.01,y=0.99),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)");return fig

def render_historical_record(row):
    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"

    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        st.write(f"**Weather:** {row.get('weather', 'N/A')}")
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        if day_type == 'Not Normal Day':
            st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title(f"Sales Forecaster")
        st.markdown("---")
        st.info("Forecasting with a Hybrid Ensemble Model")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning to get latest data.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                with st.spinner("🧠 Initializing Hybrid Forecast..."):
                    base_df = st.session_state.historical_df.copy()
                    base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                    cleaned_df, removed_count, upper_bound = remove_outliers_iqr(base_df, column='base_sales')
                    if removed_count > 0:
                        st.warning(f"Removed {removed_count} outlier day(s) with base sales over ₱{upper_bound:,.2f}.")

                    hist_df_with_atv = calculate_atv(cleaned_df)
                    ev_df = st.session_state.events_df.copy() # Will be an empty DataFrame
                    FORECAST_HORIZON = 15

                    cust_f = pd.DataFrame()
                    with st.spinner("Forecasting Customers (Simple Average)..."):
                        prophet_cust_f, prophet_model_cust = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                        xgb_cust_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')

                        if not prophet_cust_f.empty and not xgb_cust_f.empty:
                            cust_f = pd.merge(prophet_cust_f, xgb_cust_f, on='ds', suffixes=('_prophet', '_xgb'))
                            cust_f['yhat'] = (cust_f['yhat_prophet'] + cust_f['yhat_xgb']) / 2
                        else:
                            st.error("Failed to generate customer forecast.")

                    atv_f = pd.DataFrame()
                    with st.spinner("Forecasting Average Sale (Simple Average)..."):
                        prophet_atv_f, _ = train_and_forecast_prophet(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')
                        xgb_atv_f = train_and_forecast_xgboost_tuned(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')

                        if not prophet_atv_f.empty and not xgb_atv_f.empty:
                            atv_f = pd.merge(prophet_atv_f, xgb_atv_f, on='ds', suffixes=('_prophet', '_xgb'))
                            atv_f['yhat'] = (atv_f['yhat_prophet'] + atv_f['yhat_xgb']) / 2
                        else:
                            st.error("Failed to generate ATV forecast.")

                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_customers'}), atv_f[['ds', 'yhat']].rename(columns={'yhat':'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']

                        with st.spinner("🛰️ Fetching live weather..."):
                            weather_df = get_weather_forecast()
                        if weather_df is not None:
                            combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                        else:
                            combo_f['weather'] = 'Not Available'
                        
                        st.session_state.forecast_df = combo_f
                        st.success("Hybrid ensemble forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed. One or more components could not be built.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    # --- Simplified Tab List ---
    tab_list = ["🔮 Forecast Dashboard", "📜 Historical Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab ---
    with tabs[0]:
        if not st.session_state.forecast_df.empty:
            today=pd.to_datetime('today').normalize()
            future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
            if future_forecast_df.empty:st.warning("Forecast contains no future dates.")
            else:
                disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns}
                display_df=future_forecast_df.rename(columns=existing_disp_cols)
                final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
                st.markdown("#### Forecasted Values")
                st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                st.markdown("#### Forecast Visualization")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')))
                fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')))
                fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white')
                st.plotly_chart(fig,use_container_width=True)
            
            with st.expander("🔬 View Full Model Diagnostic Chart"):
                st.info("This chart shows how the individual component models (Prophet and XGBoost) performed against historical data. This is useful for advanced diagnostics.")
                d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"])
                hist_atv=calculate_atv(st.session_state.historical_df.copy())
                with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_customers':'yhat'}),st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_atv':'yhat'}),st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    # --- Historical Data Tab ---
    with tabs[1]:
        st.subheader("View Historical Data")
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
                                render_historical_record(row)
                        
                        with col2:
                            st.markdown("##### Days 16-31")
                            st.markdown("---")
                            for index, row in df_second_half.iterrows():
                                render_historical_record(row)
                else:
                    st.write(f"No data available for the year {selected_year}.")
            else:
                st.write("No historical data to display.")
        else:
            st.write("No historical data to display.")
