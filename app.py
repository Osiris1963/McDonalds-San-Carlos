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
from datetime import timedelta, date
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
            creds_dict = st.secrets.firebase_credentials.to_dict()
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
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
    if 'events_df' not in st.session_state: st.session_state.events_df = pd.DataFrame()

    defaults = {'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688"}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="1h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    docs = _db_client.collection(collection_name).stream()
    records = [doc.to_dict() for doc in docs]
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def remove_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
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
    if df_train.empty or len(df_train) < 15: return pd.DataFrame(), None
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    recurring_events = generate_recurring_local_events(df_train['date'].min(), df_train['date'].max() + timedelta(days=periods))
    all_manual_events = pd.concat([events_df.rename(columns={'date':'ds', 'activity_name':'holiday'}), recurring_events])
    model = Prophet(growth='linear', holidays=all_manual_events, weekly_seasonality=True, yearly_seasonality=len(df_train) >= 365, changepoint_prior_scale=0.15)
    model.add_country_holidays(country_name='PH')
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']], model

@st.cache_resource
def train_and_forecast_xgboost_tuned(historical_df, periods, target_col):
    df_train = historical_df.copy().dropna(subset=['date', target_col])
    if df_train.empty: return pd.DataFrame()
    df_featured = create_advanced_features(df_train)
    features = ['dayofweek', 'month', 'year', 'weekofyear'] + [f for f in [f'{target_col}_lag_7', f'{target_col}_rolling_mean_7'] if f in df_featured.columns]
    df_featured.dropna(subset=features, inplace=True)
    X, y = df_featured[features], df_featured[target_col]
    if X.empty: return pd.DataFrame()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    model.fit(X, y)
    
    future_preds = []
    history = df_featured.copy()
    for _ in range(periods):
        last_date = history['date'].max()
        next_date = last_date + timedelta(days=1)
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history, future_step_df], ignore_index=True)
        extended_featured = create_advanced_features(extended_history)
        X_future = extended_featured[features].tail(1)
        pred = model.predict(X_future)[0]
        future_preds.append({'ds': next_date, 'yhat': pred})
        history.loc[len(history)-1, target_col] = pred
        
    return pd.DataFrame(future_preds)

def train_and_forecast_with_meta_model(historical_df, events_df, target_col, forecast_horizon):
    """Orchestrates the training of base models and a meta-model for forecasting."""
    # 1. Split data for meta-model training to prevent data leakage
    split_index = int(len(historical_df) * 0.8)
    train_df = historical_df.iloc[:split_index]
    validation_df = historical_df.iloc[split_index:]
    
    # 2. Train base models on the training set and predict on the validation set
    with st.spinner(f"({target_col.upper()}) Training base models on 80% of data..."):
        prophet_val_preds, _ = train_and_forecast_prophet(train_df, events_df, len(validation_df), target_col)
        xgb_val_preds = train_and_forecast_xgboost_tuned(train_df, len(validation_df), target_col)

    if prophet_val_preds.empty or xgb_val_preds.empty:
        st.error(f"Failed to generate base model predictions for {target_col} meta-model.")
        return pd.DataFrame(), None

    # 3. Create the training set for the meta-model
    meta_train_df = pd.merge(prophet_val_preds, xgb_val_preds, on='ds', suffixes=('_prophet', '_xgb'))
    meta_train_df = pd.merge(meta_train_df, validation_df[['date', target_col]].rename(columns={'date': 'ds'}), on='ds')
    
    X_meta = meta_train_df[['yhat_prophet', 'yhat_xgb']]
    y_meta = meta_train_df[target_col]

    # 4. Train the meta-model
    with st.spinner(f"({target_col.upper()}) Training meta-model..."):
        meta_model = LinearRegression()
        meta_model.fit(X_meta, y_meta)

    # 5. Re-train base models on ALL historical data to make final future predictions
    with st.spinner(f"({target_col.upper()}) Retraining base models on all data for future forecast..."):
        prophet_future_preds, _ = train_and_forecast_prophet(historical_df, events_df, forecast_horizon, target_col)
        xgb_future_preds = train_and_forecast_xgboost_tuned(historical_df, forecast_horizon, target_col)

    if prophet_future_preds.empty or xgb_future_preds.empty:
        st.error(f"Failed to generate final future predictions from base models for {target_col}.")
        return pd.DataFrame(), None

    # 6. Combine future predictions and apply the trained meta-model
    future_df = pd.merge(prophet_future_preds, xgb_future_preds, on='ds', suffixes=('_prophet', '_xgb'))
    X_future_meta = future_df[['yhat_prophet', 'yhat_xgb']]
    future_df['yhat'] = meta_model.predict(X_future_meta)

    return future_df[['ds', 'yhat']], meta_model

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist, fcst, target):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['date'], y=hist[target], mode='lines+markers', name='Historical Actuals', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='Forecast', line=dict(color='#ffc72c', dash='dash')))
    title_text = f"{target.replace('_',' ').title()} Forecast"
    y_axis_title = title_text + ' (₱)' if 'atv' in target or 'sales' in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical', xaxis_title='Date', yaxis_title=y_axis_title,
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

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning.")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                base_df = st.session_state.historical_df.copy()
                base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                cleaned_df, _, _ = remove_outliers_iqr(base_df, column='base_sales')
                hist_df_with_atv = calculate_atv(cleaned_df)
                
                FORECAST_HORIZON = 15

                cust_f, meta_model_cust = train_and_forecast_with_meta_model(hist_df_with_atv, st.session_state.events_df, 'customers', FORECAST_HORIZON)
                atv_f, meta_model_atv = train_and_forecast_with_meta_model(hist_df_with_atv, st.session_state.events_df, 'atv', FORECAST_HORIZON)

                if meta_model_cust:
                    weights = meta_model_cust.coef_
                    st.success(f"Customer Meta-Model Weights: Prophet={weights[0]:.2f}, XGBoost={weights[1]:.2f}")
                if meta_model_atv:
                    weights = meta_model_atv.coef_
                    st.success(f"ATV Meta-Model Weights: Prophet={weights[0]:.2f}, XGBoost={weights[1]:.2f}")

                if not cust_f.empty and not atv_f.empty:
                    combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
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

    tabs = st.tabs(["🔮 Forecast Dashboard", "📜 Historical Data"])

    with tabs[0]:
        if not st.session_state.forecast_df.empty:
            future_forecast_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'].dt.date >= date.today()].copy()
            if future_forecast_df.empty:
                st.warning("Forecast contains no future dates.")
            else:
                disp_cols = {'ds': 'Date', 'forecast_customers': 'Predicted Customers', 'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)', 'weather': 'Predicted Weather'}
                display_df = future_forecast_df.rename(columns=disp_cols)
                final_cols_order = [v for k,v in disp_cols.items() if v in display_df.columns]
                st.markdown("#### Forecasted Values")
                st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}), use_container_width=True, height=560)
                
                st.markdown("#### Forecast Visualization")
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=future_forecast_df['ds'], y=future_forecast_df['forecast_sales'], mode='lines+markers', name='Sales Forecast', line=dict(color='#ffc72c')))
                fig_main.add_trace(go.Scatter(x=future_forecast_df['ds'], y=future_forecast_df['forecast_customers'], mode='lines+markers', name='Customer Forecast', yaxis='y2', line=dict(color='#c8102e')))
                fig_main.update_layout(title='15-Day Sales & Customer Forecast', yaxis=dict(title='Predicted Sales (₱)'), yaxis2=dict(title='Predicted Customers', overlaying='y', side='right'), legend=dict(x=0.01,y=0.99), paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
                st.plotly_chart(fig_main, use_container_width=True)

            with st.expander("🔬 View Full Model Diagnostic Chart"):
                st.info("This chart shows how the final meta-model forecast compares to historical data.")
                hist_atv = calculate_atv(st.session_state.historical_df.copy())
                cust_chart = plot_full_comparison_chart(hist_atv, st.session_state.forecast_df.rename(columns={'forecast_customers': 'yhat'}), 'customers')
                st.plotly_chart(cust_chart, use_container_width=True)
                atv_chart = plot_full_comparison_chart(hist_atv, st.session_state.forecast_df.rename(columns={'forecast_atv': 'yhat'}), 'atv')
                st.plotly_chart(atv_chart, use_container_width=True)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    with tabs[1]:
        st.subheader("View Historical Data")
        df_hist = st.session_state.historical_df.copy()
        if not df_hist.empty:
            all_years = sorted(df_hist['date'].dt.year.unique(), reverse=True)
            if all_years:
                sel_year = st.selectbox("Select Year:", options=all_years)
                df_year = df_hist[df_hist['date'].dt.year == sel_year]
                all_months = sorted(df_year['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)
                if all_months:
                    sel_month_str = st.selectbox("Select Month:", options=all_months)
                    sel_month_num = pd.to_datetime(sel_month_str, format='%B').month
                    filtered_df = df_year[df_year['date'].dt.month == sel_month_num]
                    
                    if filtered_df.empty:
                        st.info("No data for the selected period.")
                    else:
                        for _, row in filtered_df.iterrows():
                            render_historical_record(row)
        else:
            st.warning("No historical data found.")
