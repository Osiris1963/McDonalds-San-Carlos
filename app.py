import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Sales Forecaster",
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
            background: linear-gradient(45deg, #c8102e, #e01a37); /* Red gradient */
            color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
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
def initialize_state(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {
        'forecast_df': pd.DataFrame(), 
        'metrics': {}, 
        'forecast_components': pd.DataFrame(),
        'show_all_activities': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl="1h", show_spinner="Loading data from database...")
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
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    original_rows = len(df)
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed_rows = original_rows - len(cleaned_df)
    return cleaned_df, removed_rows, upper_bound

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    sales = pd.to_numeric(df['base_sales'], errors='coerce').fillna(0)
    customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'): atv = np.divide(sales, customers)
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600, show_spinner="Fetching weather forecast...")
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast"
        params={
            "latitude":10.48, "longitude":123.42,
            "daily":"weather_code,temperature_2m_max,precipitation_sum",
            "timezone":"Asia/Manila", "forecast_days":days,
        }
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather']=df['weather_code'].apply(map_weather_code)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch weather data. Error: {e}")
        return None

def map_weather_code(code):
    if code in [0, 1]: return "Sunny"
    if code in [2, 3]: return "Cloudy"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "Rainy"
    if code in [95, 96, 99]: return "Storm"
    return "Cloudy"

def generate_recurring_local_events(start_date,end_date):
    events=[];current_date=start_date
    while current_date<=end_date:
        if current_date.day in[15,30]:events.append({'holiday':'Payday','ds':current_date,'lower_window':0,'upper_window':1})
        if current_date.month==7 and current_date.day==1:events.append({'holiday':'San Carlos Charter Day','ds':current_date,'lower_window':0,'upper_window':0})
        current_date+=timedelta(days=1)
    return pd.DataFrame(events)

# --- ADVANCED FEATURE ENGINEERING ---
def create_advanced_features(df, target_col):
    """Creates time series, lag, rolling window, and cyclical features."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    # Cyclical features
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Advanced Lag features
    for lag in [7, 14, 21]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Same-Day-of-Week Rolling Averages (e.g., avg of last 3 Mondays)
    df[f'{target_col}_rolling_mean_dow'] = df.groupby('dayofweek')[target_col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Standard Rolling features
    df[f'{target_col}_rolling_mean_7'] = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    df[f'{target_col}_rolling_std_7'] = df[target_col].shift(1).rolling(window=7, min_periods=1).std()
    
    # Interaction features (example)
    df['dayofweek_month_interaction'] = df['dayofweek'] * df['month']

    return df.reset_index()

# --- ADVANCED FORECASTING MODELS ---
@st.cache_resource(show_spinner="Training advanced forecasting models...")
def train_stacked_ensemble_and_forecast(_historical_df, _events_df, periods, target_col):
    """
    Trains a stacked ensemble model:
    1. Base Model 1: Prophet captures trends and seasonality.
    2. Base Model 2: XGBoost models the residuals (errors) of Prophet.
    3. Meta Model: Linear Regression learns to combine Prophet and XGBoost predictions.
    """
    # --- 1. Prepare Data ---
    df_train = _historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    if df_train.empty or len(df_train) < 30:
        st.error(f"Not enough data for {target_col} to train advanced models. Need at least 30 data points.")
        return pd.DataFrame(), None, None

    # --- 2. Train Prophet (Base Model 1) ---
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    manual_events = _events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_events = pd.concat([manual_events, recurring_events]).dropna(subset=['ds', 'holiday'])

    prophet_model = Prophet(
        holidays=all_events,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(df_train) >= 365,
        changepoint_prior_scale=0.1,
        changepoint_range=0.9,
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)

    future_dates = prophet_model.make_future_dataframe(periods=periods)
    prophet_forecast = prophet_model.predict(future_dates)
    
    # --- 3. Create Aligned Dataframe for Residual Modeling ---
    df_prophet_fit = prophet_forecast[prophet_forecast['ds'] <= df_prophet['ds'].max()]
    df_residuals = pd.merge(df_prophet, df_prophet_fit[['ds', 'yhat']], on='ds')
    df_residuals['residuals'] = df_residuals['y'] - df_residuals['yhat']
    
    df_train_with_target = df_train[['date', target_col]].copy()
    df_featured = create_advanced_features(df_train_with_target, target_col)
    
    # ROBUST ALIGNMENT: Merge feature data with residual data. Use an inner join
    # to ensure we only have rows that exist in both, which guarantees alignment.
    aligned_df = pd.merge(
        df_featured,
        df_residuals,
        left_on='date',
        right_on='ds',
        how='inner'
    ).dropna() # Drop NaNs created by lag/rolling features

    FEATURES = [col for col in aligned_df.columns if col in df_featured.columns and col not in ['date', target_col, 'residuals', 'ds', 'y', 'yhat']]
    
    xgb_model_res = None
    if not aligned_df.empty:
        X = aligned_df[FEATURES]
        y_res = aligned_df['residuals']
        
        xgb_model_res = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        xgb_model_res.fit(X, y_res)
    else:
        st.warning(f"Training data is empty for {target_col} after feature engineering. Skipping residual model.")
        return pd.DataFrame(), prophet_model, None

    # --- 4. Train Meta-Model (Stacking) ---
    # All data is now sourced from the perfectly aligned `aligned_df`
    prophet_in_sample_preds = aligned_df['yhat']
    xgb_in_sample_preds = xgb_model_res.predict(aligned_df[FEATURES])
    y_meta = aligned_df['y']

    X_meta = pd.DataFrame({
        'prophet_pred': prophet_in_sample_preds.values,
        'xgb_residual_pred': xgb_in_sample_preds
    })
    
    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)

    # --- 5. Generate Final Forecast ---
    final_forecast = prophet_forecast[['ds', 'yhat']].rename(columns={'yhat': 'prophet_pred'})
    
    full_hist_df = df_train[['date', target_col]].copy()
    future_df_template = pd.DataFrame({'date': pd.to_datetime(final_forecast['ds'][final_forecast['ds'] > full_hist_df['date'].max()])})
    combined_df = pd.concat([full_hist_df, future_df_template], ignore_index=True)
    
    combined_featured = create_advanced_features(combined_df, target_col)
    
    # Predict residuals for the entire history + future
    xgb_pred_features = combined_featured[combined_featured['date'].isin(final_forecast['ds'])].copy()
    
    # Get the features that the model was trained on
    final_features_for_prediction = [f for f in FEATURES if f in xgb_pred_features.columns]
    
    # Handle potential missing columns in future predictions
    for col in FEATURES:
        if col not in xgb_pred_features.columns:
            xgb_pred_features[col] = 0 # or some other default
            
    # Ensure column order is the same
    X_to_predict = xgb_pred_features[FEATURES].fillna(0) # Fill any NaNs in future rows
    predicted_residuals = xgb_model_res.predict(X_to_predict)
    
    xgb_preds_df = pd.DataFrame({'ds': xgb_pred_features['date'], 'xgb_residual_pred': predicted_residuals})
    final_forecast = pd.merge(final_forecast, xgb_preds_df, on='ds', how='left').fillna(0)

    X_final_meta = final_forecast[['prophet_pred', 'xgb_residual_pred']]
    final_forecast['yhat'] = meta_model.predict(X_final_meta)
    
    final_forecast['yhat'] = final_forecast['yhat'].clip(lower=0)
    
    return final_forecast[['ds', 'yhat']], prophet_model, xgb_model_res

# --- Firestore Data I/O ---
def add_to_firestore(db_client, collection_name, data):
    if db_client is None: return
    data['date'] = pd.to_datetime(data['date']).to_pydatetime()
    db_client.collection(collection_name).add(data)

def update_in_firestore(db_client, collection_name, doc_id, data):
    if db_client is None: return
    db_client.collection(collection_name).document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client is None: return
    db_client.collection(collection_name).document(doc_id).delete()

# --- Plotting and UI Functions ---
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_forecast_chart(df, title):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')))
    fig.add_trace(go.Scatter(x=df['ds'],y=df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')))
    fig.update_layout(
        title=title, xaxis_title='Date',
        yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),
        yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),
        legend=dict(x=0.01,y=0.99,orientation='h'), height=500,
        paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white'
    )
    return fig

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white',
            annotations=[{"text": "No data available for this period.", "xref": "paper", "yref": "paper", "showarrow": False}])
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash')))
    fig.update_layout(title=dict(text=title), xaxis_title='Date', yaxis_title=y_axis_title,
        legend=dict(font=dict(color='white')), height=450, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white')
    return fig

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

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()

if db:
    initialize_state(db)

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("Sales Forecaster")
        st.markdown("---")
        st.info("Forecasting with a Hybrid Stacked Ensemble Model (Prophet + XGBoost).")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Data cache cleared. Rerunning...")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data for reliable forecasting.")
            else:
                with st.spinner("🧠 Initializing Advanced Hybrid Forecast..."):
                    base_df = st.session_state.historical_df.copy()
                    
                    # --- Preprocessing ---
                    base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                    cleaned_df, removed_count, _ = remove_outliers_iqr(base_df, column='base_sales')
                    if removed_count > 0: st.warning(f"Removed {removed_count} outlier day(s).")
                    hist_df_with_atv = calculate_atv(cleaned_df)
                    ev_df = st.session_state.events_df.copy()
                    
                    FORECAST_HORIZON = 15
                    
                    # --- Run Models ---
                    cust_f, prophet_model_cust, _ = train_stacked_ensemble_and_forecast(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'customers')
                    atv_f, _, _ = train_stacked_ensemble_and_forecast(hist_df_with_atv, ev_df, FORECAST_HORIZON, 'atv')

                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                        
                        weather_df = get_weather_forecast()
                        if weather_df is not None:
                            combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                        
                        st.session_state.forecast_df = combo_f
                        
                        # --- Save Forecast Log for Accuracy Tracking ---
                        try:
                            with st.spinner("📝 Saving forecast log..."):
                                today_date = pd.to_datetime('today').normalize()
                                future_forecasts_to_log = combo_f[combo_f['ds'] > today_date]
                                for _, row in future_forecasts_to_log.iterrows():
                                    doc_id = f"{today_date.strftime('%Y-%m-%d')}_{pd.to_datetime(row['ds']).strftime('%Y-%m-%d')}"
                                    log_entry = {
                                        "generated_on": today_date, "forecast_for_date": pd.to_datetime(row['ds']),
                                        "predicted_sales": row['forecast_sales'], "predicted_customers": row['forecast_customers']
                                    }
                                    db.collection('forecast_log').document(doc_id).set(log_entry)
                            st.info("Forecast log saved successfully.")
                        except Exception as e:
                            st.error(f"Failed to save forecast log: {e}")
                        
                        # --- Store Prophet components for insights ---
                        if prophet_model_cust:
                            st.session_state.forecast_components = prophet_model_cust.predict(prophet_model_cust.history)
                            st.session_state.all_holidays = prophet_model_cust.holidays
                        
                        st.success("Advanced forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed. Check data and model configurations.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)

    # --- Main Content Tabs ---
    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Forecast Evaluator", "✍️ Add/Edit Data", "📅 Future Activities"])
    
    with tabs[0]: # Forecast Dashboard
        if not st.session_state.forecast_df.empty:
            today=pd.to_datetime('today').normalize()
            future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
            if not future_forecast_df.empty:
                disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                display_df=future_forecast_df.rename(columns=disp_cols)
                final_cols_order=[v for k,v in disp_cols.items()if v in display_df.columns]
                st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                st.markdown("#### Forecast Visualization");st.plotly_chart(plot_forecast_chart(future_forecast_df, '15-Day Sales & Customer Forecast'), use_container_width=True)
            else: st.warning("Forecast contains no future dates.")
        else: st.info("Click the 'Generate Forecast' button to begin.")
    
    with tabs[1]: # Forecast Insights
        st.info("The breakdown below is generated by the Prophet model component of the forecast.")
        if st.session_state.forecast_components.empty:
            st.info("Generate a forecast first to see the breakdown of its drivers.")
        else:
            future_components=st.session_state.forecast_components[st.session_state.forecast_components['ds']>=pd.to_datetime('today').normalize()].copy()
            if not future_components.empty:
                future_components['date_str']=future_components['ds'].dt.strftime('%A,%B %d,%Y')
                selected_date_str=st.selectbox("Select a day to analyze its forecast drivers:",options=future_components['date_str'])
                selected_date=future_components[future_components['date_str']==selected_date_str]['ds'].iloc[0]
                breakdown_fig,day_data=plot_forecast_breakdown(st.session_state.forecast_components,selected_date,st.session_state.all_holidays)
                st.plotly_chart(breakdown_fig,use_container_width=True)
            else: st.warning("No future dates available in the forecast components to analyze.")

    with tabs[2]: # Forecast Evaluator
        st.header("📈 Forecast Evaluator")
        st.info("This report compares actual results against the forecast that was generated **the day before** for a true measure of day-ahead accuracy.")

        def render_true_accuracy_content(days):
            try:
                # OPTIMIZED: Query only the last 40 days of logs
                from_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=40)
                log_docs = db.collection('forecast_log').where('generated_on', '>=', from_date).stream()
                
                log_records = [doc.to_dict() for doc in log_docs]
                if not log_records:
                    st.warning("No forecast logs found in the last 40 days."); raise StopIteration

                forecast_log_df = pd.DataFrame(log_records)
                forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.tz_localize(None)
                forecast_log_df['generated_on'] = pd.to_datetime(forecast_log_df['generated_on']).dt.tz_localize(None)

                # Filter for 1-day-ahead forecasts
                forecast_log_df = forecast_log_df[forecast_log_df['forecast_for_date'] - forecast_log_df['generated_on'] == timedelta(days=1)].copy()
                if forecast_log_df.empty:
                    st.warning("Not enough consecutive logs to calculate day-ahead accuracy."); raise StopIteration

                # Merge with historical actuals
                historical_actuals_df = st.session_state.historical_df[['date', 'sales', 'customers', 'add_on_sales']].copy()
                true_accuracy_df = pd.merge(historical_actuals_df, forecast_log_df, left_on='date', right_on='forecast_for_date', how='inner')
                if true_accuracy_df.empty:
                    st.warning("No matching historical data for logged forecasts."); raise StopIteration
                
                # Filter for the selected time period
                period_start_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=days)
                final_df = true_accuracy_df[true_accuracy_df['date'] >= period_start_date].copy()
                if final_df.empty:
                    st.warning(f"No forecast data in the last {days} days to evaluate."); raise StopIteration

                # Calculate and display metrics
                st.subheader(f"Accuracy Metrics for the Last {days} Days")
                final_df['adjusted_predicted_sales'] = final_df['predicted_sales'] - final_df['add_on_sales']
                sales_mae = mean_absolute_error(final_df['sales'], final_df['adjusted_predicted_sales'])
                cust_mae = mean_absolute_error(final_df['customers'], final_df['predicted_customers'])
                sales_mape = np.mean(np.abs((final_df['sales'] - final_df['adjusted_predicted_sales']) / final_df['sales'].replace(0, np.nan))) * 100
                cust_mape = np.mean(np.abs((final_df['customers'] - final_df['predicted_customers']) / final_df['customers'].replace(0, np.nan))) * 100

                col1, col2 = st.columns(2)
                col1.metric("Sales Accuracy (MAPE)", f"{100 - sales_mape:.2f}%", help="Mean Absolute Percentage Error")
                col1.metric("Sales MAE (Avg Error)", f"₱{sales_mae:,.2f}")
                col2.metric("Customer Accuracy (MAPE)", f"{100 - cust_mape:.2f}%")
                col2.metric("Customer MAE (Avg Error)", f"{cust_mae:,.0f} customers")

                st.markdown("---")
                st.subheader(f"Comparison Charts for the Last {days} Days")
                st.plotly_chart(plot_evaluation_graph(final_df, 'date', 'sales', 'adjusted_predicted_sales', 'Actual vs. Forecasted Sales', 'Sales (₱)'), use_container_width=True)
                st.plotly_chart(plot_evaluation_graph(final_df, 'date', 'customers', 'predicted_customers', 'Actual vs. Forecasted Customers', 'Customers'), use_container_width=True)

            except StopIteration: pass 
            except Exception as e: st.error(f"An error occurred while building the report: {e}")

        eval_tab_7, eval_tab_30 = st.tabs(["Last 7 Days", "Last 30 Days"])
        with eval_tab_7: render_true_accuracy_content(7)
        with eval_tab_30: render_true_accuracy_content(30)

    with tabs[3]: # Add/Edit Data
        st.subheader("✍️ Add New Daily Record")
        with st.form("new_record_form",clear_on_submit=True, border=True):
            new_date=st.date_input("Date", date.today())
            c1, c2, c3 = st.columns(3)
            new_sales=c1.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
            new_customers=c2.number_input("Customer Count",min_value=0)
            new_addons=c3.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
            
            if st.form_submit_button("✅ Save Record", use_container_width=True):
                new_rec={"date":new_date, "sales":new_sales, "customers":new_customers, "add_on_sales":new_addons}
                add_to_firestore(db,'historical_data',new_rec)
                st.cache_data.clear(); st.success("Record added!"); time.sleep(1); st.rerun()

    with tabs[4]: # Future Activities
        def set_view_all(): st.session_state.show_all_activities = True
        def set_overview(): st.session_state.show_all_activities = False

        if st.session_state.get('show_all_activities'):
            st.button("⬅️ Back to Overview", on_click=set_overview)
            # Full view logic can be added here if needed
        else:
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.markdown("##### Add New Activity")
                with st.form("new_activity_form", clear_on_submit=True, border=True):
                    activity_name = st.text_input("Activity/Event Name")
                    activity_date = st.date_input("Date of Activity", min_value=date.today())
                    potential_sales = st.number_input("Potential Sales (₱)", min_value=0.0, format="%.2f")
                    if st.form_submit_button("✅ Save Activity", use_container_width=True):
                        if activity_name and activity_date:
                            new_activity = {"activity_name": activity_name, "date": activity_date, "potential_sales": float(potential_sales)}
                            add_to_firestore(db, 'future_activities', new_activity)
                            st.cache_data.clear(); st.success("Activity saved!"); time.sleep(1); st.rerun()
            with col2:
                st.markdown("##### Upcoming Activities")
                activities_df = st.session_state.events_df
                upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy()
                st.dataframe(upcoming_df, use_container_width=True, hide_index=True)

else:
    st.error("Failed to connect to Firestore. Please check your configuration and network.")
