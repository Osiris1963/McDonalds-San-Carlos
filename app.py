import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objs as go
import time
import requests
from datetime import timedelta, date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
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
    if 'db_client' not in st.session_state:
        st.session_state.db_client = db_client

    if 'historical_df' not in st.session_state:
        raw_records, timestamp = load_from_firestore(db_client, 'historical_data')
        st.session_state.raw_records = raw_records # Store raw data for inspection
        st.session_state.historical_df = process_historical_data(raw_records)
        st.session_state.data_last_loaded = timestamp

    defaults = {
        'events_df': pd.DataFrame(),
        'unified_forecast_cust': pd.DataFrame(),
        'unified_forecast_atv': pd.DataFrame(),
        'forecast_df': pd.DataFrame(),
        'name': "Store 688"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
def load_from_firestore(_db_client, collection_name):
    """Fetches raw data from Firestore and returns it with a timestamp."""
    if _db_client is None: return [], pd.Timestamp.now(tz='UTC')
    
    docs = _db_client.collection(collection_name).stream()
    records = [doc.to_dict() for doc in docs]
    fetch_timestamp = pd.Timestamp.now(tz='UTC')
    return records, fetch_timestamp

@st.cache_data(ttl="1h")
def process_historical_data(records):
    """Takes raw records, cleans them, and returns a validated, sorted DataFrame."""
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' not in df.columns:
        return pd.DataFrame()
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['date'] = df['date'].dt.tz_localize(None).dt.normalize()

    df['completeness'] = df.count(axis=1)
    df.sort_values(['date', 'completeness'], ascending=[True, True], inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    df.drop(columns=['completeness'], inplace=True)

    numeric_cols = [
        'sales', 'customers', 'add_on_sales', 
        'last_year_sales', 'last_year_customers'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    
    if 'day_type' not in df.columns:
        df['day_type'] = 'Normal Day'
    else:
        df['day_type'].fillna('Normal Day', inplace=True)

    df = df.sort_values(by='date', ascending=True)
    df = df.reset_index(drop=True)
    
    return df

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(df['base_sales'], df['customers'])
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        params={"latitude":10.48,"longitude":123.42,"daily":"weather_code","timezone":"Asia/Manila","forecast_days":days}
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['weather'] = df['weather_code'].apply(map_weather_code)
        return df[['date', 'weather']]
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch weather data: {e}")
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
        if current_date.month == 7 and current_date.day == 1:
            events.append({'holiday': 'San Carlos Charter Day', 'ds': current_date, 'lower_window': 0, 'upper_window': 0})
        current_date += timedelta(days=1)
    return pd.DataFrame(events)

def create_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df = df.sort_values('date')
    for col in ['sales', 'customers', 'atv']:
        if col in df.columns:
            df[f'{col}_lag_7'] = df[col].shift(7)
            df[f'{col}_rolling_mean_7'] = df[col].shift(1).rolling(window=7, min_periods=1).mean()
    return df

@st.cache_resource
def train_and_forecast_prophet(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if df_train.empty or len(df_train) < 15: return pd.DataFrame()
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods) if periods > 0 else df_train['date'].max()
    recurring_events = generate_recurring_local_events(start_date, end_date)
    
    all_manual_events = pd.concat([events_df.rename(columns={'date':'ds', 'activity_name':'holiday'}), recurring_events])
    
    model = Prophet(growth='linear', holidays=all_manual_events, weekly_seasonality=True, yearly_seasonality=len(df_train) >= 365, changepoint_prior_scale=0.15)
    model.add_country_holidays(country_name='PH')
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=periods, include_history=True)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if df_train.empty: return pd.DataFrame()
    
    df_featured = create_advanced_features(df_train)
    features = ['dayofweek', 'month', 'year', 'weekofyear'] + [f for f in [f'{target_col}_lag_7', f'{target_col}_rolling_mean_7'] if f in df_featured.columns]
    df_featured.dropna(subset=features, inplace=True)
    X_hist, y_hist = df_featured[features], df_featured[target_col]
    if X_hist.empty: return pd.DataFrame()
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    model.fit(X_hist, y_hist)

    hist_preds = model.predict(X_hist)
    full_preds = pd.DataFrame({'ds': df_featured['date'], 'yhat': hist_preds})
    
    future_preds_list = []
    history = df_featured.copy()
    for _ in range(periods):
        last_date = history['date'].max()
        next_date = last_date + timedelta(days=1)
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history, future_step_df], ignore_index=True)
        extended_featured = create_advanced_features(extended_history)
        X_future = extended_featured[features].tail(1)
        pred = model.predict(X_future)[0]
        future_preds_list.append({'ds': next_date, 'yhat': pred})
        history = pd.concat([history, pd.DataFrame([{'date': next_date, target_col: pred}])], ignore_index=True)

    if future_preds_list:
        future_preds = pd.DataFrame(future_preds_list)
        full_preds = pd.concat([full_preds, future_preds], ignore_index=True)
        
    return full_preds

def train_and_forecast_with_meta_model(historical_df, events_df, target_col, forecast_horizon):
    """Orchestrates the training of base models and a meta-model for forecasting."""
    split_index = int(len(historical_df) * 0.8)
    train_df = historical_df.iloc[:split_index]
    validation_df = historical_df.iloc[split_index:]
    
    prophet_val_preds = train_and_forecast_prophet(train_df, events_df, len(validation_df), target_col).tail(len(validation_df))
    xgb_val_preds = train_and_forecast_xgboost_tuned(train_df, events_df, len(validation_df), target_col).tail(len(validation_df))
    if prophet_val_preds.empty or xgb_val_preds.empty: return pd.DataFrame(), None

    meta_train_df = pd.merge(prophet_val_preds, xgb_val_preds, on='ds', suffixes=('_prophet', '_xgb'))
    meta_train_df = pd.merge(meta_train_df, validation_df[['date', target_col]].rename(columns={'date': 'ds'}), on='ds')
    X_meta, y_meta = meta_train_df[['yhat_prophet', 'yhat_xgb']], meta_train_df[target_col]

    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)

    prophet_full_preds = train_and_forecast_prophet(historical_df, events_df, forecast_horizon, target_col)
    xgb_full_preds = train_and_forecast_xgboost_tuned(historical_df, events_df, forecast_horizon, target_col)
    if prophet_full_preds.empty or xgb_full_preds.empty: return pd.DataFrame(), None

    unified_df = pd.merge(prophet_full_preds, xgb_full_preds, on='ds', suffixes=('_prophet', '_xgb'))
    X_unified_meta = unified_df[['yhat_prophet', 'yhat_xgb']]
    unified_df['yhat'] = meta_model.predict(X_unified_meta)

    return unified_df[['ds', 'yhat']], meta_model

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_unified_forecast(historical_df, unified_forecast_df, target_col):
    """Plots historical actuals, in-sample fit, and future forecast in one chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=historical_df['date'], y=historical_df[target_col], mode='lines', name='Historical Actuals', line=dict(color='#3b82f6', width=2.5)))
    
    last_hist_date = historical_df['date'].max()
    in_sample_fit = unified_forecast_df[unified_forecast_df['ds'] <= last_hist_date]
    future_forecast = unified_forecast_df[unified_forecast_df['ds'] > last_hist_date]

    fig.add_trace(go.Scatter(x=in_sample_fit['ds'], y=in_sample_fit['yhat'], mode='lines', name='Model Fit (In-Sample)', line=dict(color='#ffc72c', dash='dash')))
    
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Future Forecast', line=dict(color='#d62728', dash='dot')))

    title_text = f"Full Forecast View: {target_col.replace('_',' ').title()}"
    fig.update_layout(title=title_text, xaxis_title='Date', yaxis_title=target_col.title(),
                      legend=dict(x=0.01, y=0.99), height=500, paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
    return fig

def render_historical_record(row):
    date_str = pd.to_datetime(row['date']).strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        st.write(f"**Weather:** {row.get('weather', 'N/A')}")
        st.write(f"**Day Type:** {row.get('day_type', 'Normal Day')}")

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("Sales Forecaster")
        st.markdown("---")
        st.info("Forecasting with a Meta-Model Ensemble (Prophet + XGBoost)")

        if st.button("🔄 Force Refresh All Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("All caches and session state cleared. Rerunning to fetch fresh data.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                base_df = st.session_state.historical_df.copy()
                hist_df_with_atv = calculate_atv(base_df)
                
                FORECAST_HORIZON = 15

                with st.spinner("Generating customer forecast..."):
                    unified_cust_f, meta_model_cust = train_and_forecast_with_meta_model(hist_df_with_atv, st.session_state.events_df, 'customers', FORECAST_HORIZON)
                    st.session_state.unified_forecast_cust = unified_cust_f
                
                with st.spinner("Generating average transaction forecast..."):
                    unified_atv_f, meta_model_atv = train_and_forecast_with_meta_model(hist_df_with_atv, st.session_state.events_df, 'atv', FORECAST_HORIZON)
                    st.session_state.unified_forecast_atv = unified_atv_f

                if meta_model_cust:
                    weights = meta_model_cust.coef_
                    st.success(f"Customer Meta-Model Weights: Prophet={weights[0]:.2f}, XGBoost={weights[1]:.2f}")
                if meta_model_atv:
                    weights = meta_model_atv.coef_
                    st.success(f"ATV Meta-Model Weights: Prophet={weights[0]:.2f}, XGBoost={weights[1]:.2f}")

                if not unified_cust_f.empty and not unified_atv_f.empty:
                    combo_f = pd.merge(unified_cust_f.rename(columns={'yhat':'forecast_customers'}), unified_atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                    combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                    weather_df = get_weather_forecast()
                    if weather_df is not None:
                        combo_f = pd.merge(combo_f, weather_df, left_on='ds', right_on='date', how='left').drop(columns=['date'])
                    st.session_state.forecast_df = combo_f
                    st.balloons()
                else:
                    st.error("Forecast generation failed.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    # Add the new diagnostic tab
    tabs = st.tabs(["🔮 Forecast Dashboard", "📅 Full Forecast", "📜 Historical Data", "🔬 Raw Data Inspector"])

    with tabs[0]: # Forecast Dashboard
        if not st.session_state.forecast_df.empty:
            future_forecast_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'].dt.date >= date.today()].copy()
            if future_forecast_df.empty:
                st.warning("Forecast contains no future dates.")
            else:
                disp_cols = {'ds': 'Date', 'forecast_customers': 'Predicted Customers', 'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)', 'weather': 'Predicted Weather'}
                display_df = future_forecast_df.rename(columns=disp_cols)
                final_cols_order = [v for k,v in disp_cols.items() if v in display_df.columns]
                st.markdown("#### Future Forecasted Values")
                st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}), use_container_width=True, height=560)
                
                st.markdown("#### Future Forecast Visualization")
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=future_forecast_df['ds'], y=future_forecast_df['forecast_sales'], mode='lines+markers', name='Sales Forecast', line=dict(color='#ffc72c')))
                fig_main.add_trace(go.Scatter(x=future_forecast_df['ds'], y=future_forecast_df['forecast_customers'], mode='lines+markers', name='Customer Forecast', yaxis='y2', line=dict(color='#c8102e')))
                fig_main.update_layout(title='15-Day Sales & Customer Forecast', yaxis=dict(title='Predicted Sales (₱)'), yaxis2=dict(title='Predicted Customers', overlaying='y', side='right'), legend=dict(x=0.01,y=0.99), paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
                st.plotly_chart(fig_main, use_container_width=True)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    with tabs[1]: # Full Forecast
        st.header("Unified Historical and Forecast View")
        st.info("This view shows historical actuals alongside the model's performance on that past data and its forecast for the future.")
        if not st.session_state.unified_forecast_cust.empty and not st.session_state.unified_forecast_atv.empty:
            hist_df_atv = calculate_atv(st.session_state.historical_df.copy())
            
            st.markdown("---")
            st.subheader("Customer Count: Full View")
            fig_cust = plot_unified_forecast(hist_df_atv, st.session_state.unified_forecast_cust, 'customers')
            st.plotly_chart(fig_cust, use_container_width=True)

            st.markdown("---")
            st.subheader("Average Transaction Value (ATV): Full View")
            fig_atv = plot_unified_forecast(hist_df_atv, st.session_state.unified_forecast_atv, 'atv')
            st.plotly_chart(fig_atv, use_container_width=True)
        else:
            st.warning("Please generate a forecast first to see the full view.")

    with tabs[2]: # Historical Data
        st.subheader("View Historical Data")
        
        if 'data_last_loaded' in st.session_state and st.session_state.data_last_loaded is not None:
            last_loaded_time = st.session_state.data_last_loaded.tz_convert('Asia/Manila')
            st.info(f"🕒 Data displayed was fetched from the database on: {last_loaded_time.strftime('%B %d, %Y at %I:%M %p')}.")
        
        st.markdown("---")
        
        df_hist = st.session_state.historical_df.copy()
        if not df_hist.empty:
            all_years = sorted(df_hist['date'].dt.year.unique(), reverse=True)
            if all_years:
                sel_year = st.selectbox("Select Year:", options=all_years, key="hist_year")
                df_year = df_hist[df_hist['date'].dt.year == sel_year]
                all_months = sorted(df_year['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)
                if all_months:
                    sel_month_str = st.selectbox("Select Month:", options=all_months, key="hist_month")
                    sel_month_num = pd.to_datetime(sel_month_str, format='%B').month
                    filtered_df = df_year[df_year['date'].dt.month == sel_month_num]
                    
                    if filtered_df.empty:
                        st.info("No data for the selected period.")
                    else:
                        for index, row in filtered_df.iterrows():
                            render_historical_record(row)
        else:
            st.warning("No historical data found.")
            
    with tabs[3]: # Raw Data Inspector
        st.header("🔬 Raw Data Inspector")
        st.warning("This panel is for debugging purposes. It shows the raw data received from Firestore *before* any cleaning or processing is applied.")
        
        if 'raw_records' in st.session_state and st.session_state.raw_records:
            # Filter to find the relevant records
            problematic_records = []
            for r in st.session_state.raw_records:
                try:
                    # Make the check robust in case 'date' field is missing
                    record_date = r.get('date')
                    if record_date:
                        # Handle both datetime objects and strings
                        if isinstance(record_date, str):
                            record_date = datetime.fromisoformat(record_date)
                        if isinstance(record_date, datetime):
                            if record_date.month == 7 and record_date.day in [13, 14, 15] and record_date.year == 2025:
                                problematic_records.append(r)
                except Exception as e:
                    st.error(f"Could not parse a record: {r}. Error: {e}")

            if not problematic_records:
                st.info("Could not find raw records for July 13-15, 2025.")
            else:
                st.write("Displaying raw records for July 13-15, 2025:")
                for record in problematic_records:
                    st.markdown("---")
                    date_val = record.get('date', 'N/A')
                    st.write(f"**Record with date value:** `{date_val}`")
                    st.write(f"**Python data type of 'date' field:** `{type(date_val)}`")
                    st.json(record)
        else:
            st.info("No raw records loaded in session state. Please use the 'Force Refresh' button in the sidebar.")
