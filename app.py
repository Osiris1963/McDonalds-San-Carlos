import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import numpy as np
import plotly.graph_objs as go
import time
import requests
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Suppress informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


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
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; }
        [data-testid="stSidebar"] { background-color: #252525; }
        .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out; border: none; padding: 10px 16px; }
        .stButton:has(button:contains("Generate")), .stButton:has(button:contains("Save")) > button { background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF; }
        .stButton:has(button:contains("Refresh")), .stButton:has(button:contains("View All")), .stButton:has(button:contains("Back to Overview")) > button { border: 2px solid #c8102e; background: transparent; color: #c8102e; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: #ffffff; }
        .st-expander { border: 1px solid #444 !important; border-radius: 10px; background-color: #252525; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization & C.R.U.D Functions ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = {
              "type": st.secrets.firebase_credentials.type, "project_id": st.secrets.firebase_credentials.project_id,
              "private_key_id": st.secrets.firebase_credentials.private_key_id, "private_key": st.secrets.firebase_credentials.private_key.replace('\\n', '\n'),
              "client_email": st.secrets.firebase_credentials.client_email, "client_id": st.secrets.firebase_credentials.client_id,
              "auth_uri": st.secrets.firebase_credentials.auth_uri, "token_uri": st.secrets.firebase_credentials.token_uri,
              "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url,
              "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

def add_to_firestore(db_client, collection_name, record, existing_df):
    date_to_check = pd.to_datetime(record['date']).normalize()
    if not existing_df.empty and 'date' in existing_df.columns and date_to_check in pd.to_datetime(existing_df['date']).normalize().values:
        st.error(f"A record for {date_to_check.strftime('%Y-%m-%d')} already exists.")
        return
    db_client.collection(collection_name).add(record)

def update_firestore_record(db_client, collection_name, doc_id, update_data):
    db_client.collection(collection_name).document(doc_id).update(update_data)

def delete_from_firestore(db_client, collection_name, doc_id):
    db_client.collection(collection_name).document(doc_id).delete()

# --- App State Management ---
def initialize_state(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {'forecast_df': pd.DataFrame(), 'prophet_models': {}, 'username': "Admin", 'show_all_activities': False}
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
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    for col in ['sales', 'customers', 'add_on_sales', 'potential_sales']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').interpolate().fillna(0)
    return df.sort_values(by='date').reset_index(drop=True)

def cap_outliers_iqr(df, column='sales'):
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    df_capped = df.copy()
    df_capped.loc[df_capped[column] > upper_bound, column] = upper_bound
    return df_capped

@st.cache_data
def calculate_atv(df):
    df_copy = df.copy()
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(base_sales, df_copy['customers'])
    df_copy['atv'] = np.nan_to_num(atv)
    return df_copy

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        params={"latitude":10.48, "longitude":123.42, "daily":"weather_code,temperature_2m_max", "timezone":"Asia/Manila", "forecast_days":days}
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json()['daily'])
        df.rename(columns={'time':'date', 'temperature_2m_max':'temp_max'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'temp_max']]
    except requests.exceptions.RequestException:
        return pd.DataFrame()

@st.cache_data
def create_advanced_features(df, weather_df=None):
    df['date'] = pd.to_datetime(df['date'])
    if weather_df is not None and not weather_df.empty:
        df = pd.merge(df, weather_df, on='date', how='left')
        df['temp_max'].ffill(inplace=True)
        df['temp_max'].bfill(inplace=True)
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['is_payday_window'] = df['date'].dt.day.isin([15, 30, 31, 1, 2]).astype(int)

    df = df.sort_values('date').reset_index(drop=True)
    for lag in [7, 14, 28]: # Weekly lags
        if 'sales' in df.columns: df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        if 'customers' in df.columns: df[f'customers_lag_{lag}'] = df['customers'].shift(lag)

    return df.fillna(0)

def check_performance_and_recalibrate(db, historical_df, degradation_threshold=0.98, short_term_days=7, long_term_days=30):
    try:
        log_docs = db.collection('forecast_log').stream()
        log_records = [doc.to_dict() for doc in log_docs]
        if not log_records: return False

        forecast_log_df = pd.DataFrame(log_records)
        forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.tz_localize(None)
        forecast_log_df['generated_on'] = pd.to_datetime(forecast_log_df['generated_on']).dt.tz_localize(None)
        day_ahead_logs = forecast_log_df[forecast_log_df['forecast_for_date'] - forecast_log_df['generated_on'] <= timedelta(days=1)].copy()
        if day_ahead_logs.empty: return False

        true_accuracy_df = pd.merge(historical_df, day_ahead_logs, left_on='date', right_on='forecast_for_date', how='inner')
        if len(true_accuracy_df) < long_term_days: return False 
        
        today = pd.to_datetime('today').normalize()
        long_term_df = true_accuracy_df[true_accuracy_df['date'] >= today - pd.Timedelta(days=long_term_days)]
        short_term_df = true_accuracy_df[true_accuracy_df['date'] >= today - pd.Timedelta(days=short_term_days)]

        if long_term_df.empty or short_term_df.empty: return False

        long_term_accuracy = 100 - (np.nanmean(np.abs((long_term_df['sales'] - long_term_df['predicted_sales']) / long_term_df['sales'].replace(0, np.nan))) * 100)
        short_term_accuracy = 100 - (np.nanmean(np.abs((short_term_df['sales'] - short_term_df['predicted_sales']) / short_term_df['sales'].replace(0, np.nan))) * 100)

        if short_term_accuracy < (long_term_accuracy * degradation_threshold):
            st.warning(f"🚨 Model accuracy degradation detected! Recalibrating...")
            st.cache_data.clear()
            time.sleep(2) 
            return True
    except Exception as e:
        st.warning(f"Could not perform accuracy check. Error: {e}")
    return False

# --- Core Forecasting Models (Simplified) ---
@st.cache_data
def train_and_forecast_prophet_day_specific(historical_df, periods, target_col, day_of_week):
    df_train = historical_df[historical_df['date'].dt.dayofweek == day_of_week].copy()
    if len(df_train) < 10: return pd.DataFrame(), None
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})
    prophet_model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    last_date = historical_df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7)
    future_day_specific = future_dates[future_dates.dayofweek == day_of_week][:periods]
    future_df = pd.DataFrame({'ds': future_day_specific})
    forecast = prophet_model.predict(future_df)
    return forecast[['ds', 'yhat', 'trend', 'yearly', 'holidays']], prophet_model

@st.cache_data
def train_and_forecast_xgb_day_specific(historical_df, periods, target_col, day_of_week, customer_forecast_df=None, weather_df=None):
    df_day = historical_df[historical_df['date'].dt.dayofweek == day_of_week].copy()
    if len(df_day) < 20: return pd.DataFrame()
    last_date = historical_df['date'].max()
    future_dates_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7)
    future_day_specific_dates = future_dates_range[future_dates_range.dayofweek == day_of_week][:periods]
    future_df_placeholders = pd.DataFrame({'date': future_day_specific_dates})
    combined_df = pd.concat([df_day, future_df_placeholders], ignore_index=True)
    combined_featured_df = create_advanced_features(combined_df, weather_df)
    if target_col == 'atv' and customer_forecast_df is not None:
        combined_featured_df = pd.merge(combined_featured_df, customer_forecast_df[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecast_customers'}), on='date', how='left')
        combined_featured_df['forecast_customers'].ffill(inplace=True).bfill(inplace=True)
    features = [f for f in combined_featured_df.columns if combined_featured_df[f].dtype in ['int64', 'float64'] and f not in ['sales', 'customers', 'atv', 'date', target_col]]
    train_df = combined_featured_df.dropna(subset=[target_col])
    predict_df = combined_featured_df[combined_featured_df[target_col].isna()]
    X_train, y_train = train_df[features], train_df[target_col]
    X_future = predict_df[features]
    if X_train.empty or X_future.empty: return pd.DataFrame()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42).fit(X_train, y_train)
    predictions = model.predict(X_future)
    return pd.DataFrame({'ds': predict_df['date'], 'yhat': predictions})

# --- Simplified Orchestration Function ---
def run_simplified_ensemble_pipeline(historical_df, target_col, periods, weather_df, customer_forecasts=None):
    all_forecasts = []
    prophet_models = {}
    for day_of_week in range(7):
        prophet_f, p_model = train_and_forecast_prophet_day_specific(historical_df, periods, target_col, day_of_week)
        xgb_f = train_and_forecast_xgb_day_specific(historical_df, periods, target_col, day_of_week, customer_forecasts, weather_df)
        if prophet_f.empty or xgb_f.empty: continue
        if p_model: prophet_models[day_of_week] = p_model
        merged_f = pd.merge(prophet_f.rename(columns={'yhat':'yhat_prophet'}), xgb_f.rename(columns={'yhat':'yhat_xgb'}), on='ds', how='inner')
        merged_f['yhat'] = (merged_f['yhat_prophet'] + merged_f['yhat_xgb']) / 2
        for col in ['trend', 'yearly', 'holidays']:
            if col in merged_f.columns:
                merged_f[col] = merged_f[col]
        all_forecasts.append(merged_f)
    if not all_forecasts: return pd.DataFrame(), {}
    final_forecast = pd.concat(all_forecasts).sort_values('ds').reset_index(drop=True)
    return final_forecast, prophet_models

def log_forecast_to_firestore(db_client, forecast_df):
    generated_on = pd.to_datetime('today')
    for _, row in forecast_df.iterrows():
        log_entry = {
            "forecast_for_date": row['ds'],
            "predicted_sales": row['forecast_sales'],
            "predicted_customers": row['forecast_customers'],
            "generated_on": generated_on
        }
        query = db_client.collection('forecast_log').where('forecast_for_date', '==', row['ds']).limit(1)
        docs = list(query.stream())
        if docs:
            db_client.collection('forecast_log').document(docs[0].id).set(log_entry)
        else:
            db_client.collection('forecast_log').add(log_entry)

# --- Plotting and UI Rendering ---
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_forecast_breakdown(prophet_model, forecast_df, selected_date):
    if not prophet_model:
        st.warning("Prophet model not available for this day.")
        return go.Figure()
    
    components_df = prophet_model.predict(pd.DataFrame({'ds': [selected_date]}))
    return prophet_model.plot_components(components_df)

def plot_evaluation_graph(df, date_col, actual_col, forecast_col, title, y_axis_title):
    if df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name='Actual', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode='lines+markers', name='Forecast', line=dict(color='#d62728', dash='dash')))
    fig.update_layout(title=title, yaxis_title=y_axis_title, paper_bgcolor='#1a1a1a', plot_bgcolor='#252525', font_color='white')
    return fig

def render_activity_card(row, db_client):
    with st.expander(f"**{pd.to_datetime(row['date']).strftime('%b %d')}**: {row['activity_name']}"):
        status, sales, doc_id = row['remarks'], row['potential_sales'], row['doc_id']
        st.markdown(f"**Status:** {status} | **Potential Sales:** ₱{sales:,.2f}")
        with st.form(key=f"update_form_{doc_id}", border=False):
            updated_sales = st.number_input("Sales (₱)", value=float(sales), key=f"sales_{doc_id}")
            updated_remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"], index=["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"].index(status), key=f"remarks_{doc_id}")
            c1, c2 = st.columns(2)
            if c1.form_submit_button("💾 Update", use_container_width=True):
                update_firestore_record(db_client, 'future_activities', doc_id, {"potential_sales": updated_sales, "remarks": updated_remarks})
                st.success("Activity updated!"); st.cache_data.clear(); time.sleep(1); st.rerun()
            if c2.form_submit_button("🗑️ Delete", use_container_width=True):
                delete_from_firestore(db_client, 'future_activities', doc_id)
                st.warning("Activity deleted."); st.cache_data.clear(); time.sleep(1); st.rerun()

def render_historical_record(row, db_client):
    with st.expander(f"{row['date'].strftime('%B %d, %Y')} - Sales: ₱{row.get('sales', 0):,.2f}"):
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            c1, c2 = st.columns(2)
            updated_sales = c1.number_input("Sales (₱)", value=float(row.get('sales', 0)), key=f"sales_{row['doc_id']}")
            updated_customers = c2.number_input("Customers", value=int(row.get('customers', 0)), key=f"cust_{row['doc_id']}")
            if c1.form_submit_button("💾 Update Record", use_container_width=True):
                update_firestore_record(db_client, 'historical_data', row['doc_id'], {'sales': updated_sales, 'customers': updated_customers})
                st.success("Record updated!"); st.cache_data.clear(); time.sleep(1); st.rerun()
            if c2.form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                st.warning("Record deleted."); st.cache_data.clear(); time.sleep(1); st.rerun()

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state(db)
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png", width=80)
        st.title(f"Welcome, {st.session_state['username']}")
        st.markdown("---")
        st.info("Forecasting with a Day-Specific Prophet + XGBoost Ensemble.")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Please provide at least 50 days of data.")
            else:
                if check_performance_and_recalibrate(db, st.session_state.historical_df):
                    st.info("Model recalibrated. Please click 'Generate Forecast' again.")
                else:
                    with st.spinner("🧠 Training Day-Specific Models (Prophet + XGBoost)..."):
                        base_df = st.session_state.historical_df.copy()
                        FORECAST_HORIZON = 15
                        weather_df = get_weather_forecast(days=FORECAST_HORIZON + 30)
                        
                        hist_df_with_atv = calculate_atv(cap_outliers_iqr(base_df, column='sales'))
                        
                        # Stage 1: Forecast Customers
                        cust_f, cust_models = run_simplified_ensemble_pipeline(hist_df_with_atv, 'customers', FORECAST_HORIZON, weather_df)
                        
                        # Check if the customer forecast is valid before proceeding
                        if cust_f.empty:
                            st.error("Forecast Failed: Could not generate a customer forecast. Ensure sufficient data exists for each day of the week.")
                        else:
                            # Stage 2: Forecast ATV (only if customer forecast was successful)
                            atv_f, atv_models = run_simplified_ensemble_pipeline(hist_df_with_atv, 'atv', FORECAST_HORIZON, weather_df, customer_forecasts=cust_f)
                            
                            if not atv_f.empty:
                                combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds', how='outer')
                                combo_f.interpolate(method='linear', limit_direction='both', inplace=True)
                                combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                                st.session_state.forecast_df = combo_f
                                st.session_state.prophet_models['customers'] = cust_models
                                st.session_state.prophet_models['atv'] = atv_models
                                log_forecast_to_firestore(db, combo_f)
                                st.success("Forecast generated!")
                            else:
                                st.error("Forecast Failed: Could not generate an ATV forecast after the customer forecast.")
        
        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical.csv", "text/csv", use_container_width=True)

    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "📈 Evaluator", "✍️ Add Data", "📅 Activities", "📜 History"])
    
    with tabs[0]: # Dashboard
        if not st.session_state.forecast_df.empty:
            df = st.session_state.forecast_df
            st.dataframe(df.rename(columns={'ds':'Date', 'forecast_customers':'Customers', 'forecast_atv':'Avg. Sale (₱)', 'forecast_sales':'Sales (₱)'}).set_index('Date'), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_sales'], name='Sales', line=dict(color='#ffc72c')))
            fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_customers'], name='Customers', yaxis='y2', line=dict(color='#c8102e')))
            fig.update_layout(title='15-Day Sales & Customer Forecast', yaxis_title='Sales (₱)', yaxis2=dict(title='Customers', overlaying='y', side='right'), paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Click 'Generate Forecast' to begin.")

    with tabs[1]: # Insights
        st.header("💡 Forecast Insights")
        if not st.session_state.forecast_df.empty:
            selected_date = st.selectbox("Select a date to analyze:", options=st.session_state.forecast_df['ds'].dt.date)
            selected_date = pd.to_datetime(selected_date)
            day_of_week = selected_date.dayofweek
            target_to_view = st.radio("View insights for:", ["Customers", "Average Transaction Value (ATV)"], horizontal=True)
            
            if target_to_view == "Customers" and day_of_week in st.session_state.prophet_models.get('customers', {}):
                model = st.session_state.prophet_models['customers'][day_of_week]
                fig = plot_forecast_breakdown(model, st.session_state.forecast_df, selected_date)
                st.pyplot(fig)
            elif target_to_view == "Average Transaction Value (ATV)" and day_of_week in st.session_state.prophet_models.get('atv', {}):
                model = st.session_state.prophet_models['atv'][day_of_week]
                fig = plot_forecast_breakdown(model, st.session_state.forecast_df, selected_date)
                st.pyplot(fig)
            else: st.warning("No insight components available for the selected day or target.")
        else: st.info("Generate a forecast to view insights.")

    with tabs[2]: # Evaluator
        st.header("📈 Forecast Evaluator")
        try:
            log_docs = db.collection('forecast_log').stream()
            log_records = [doc.to_dict() for doc in log_docs]
            if log_records:
                forecast_log_df = pd.DataFrame(log_records)
                forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.tz_localize(None)
                true_accuracy_df = pd.merge(st.session_state.historical_df, forecast_log_df, left_on='date', right_on='forecast_for_date', how='inner')
                
                days_to_eval = st.slider("Select evaluation period (days):", 7, 90, 30)
                final_df = true_accuracy_df[true_accuracy_df['date'] >= pd.to_datetime('today').normalize() - pd.Timedelta(days=days_to_eval)].copy()

                if not final_df.empty:
                    sales_mae = mean_absolute_error(final_df['sales'], final_df['predicted_sales'])
                    sales_mape = np.mean(np.abs((final_df['sales'] - final_df['predicted_sales']) / final_df['sales'].replace(0,1))) * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Sales Accuracy (MAPE)", f"{100-sales_mape:.2f}%")
                    c2.metric("Sales Average Error (MAE)", f"₱{sales_mae:,.2f}")
                    st.plotly_chart(plot_evaluation_graph(final_df, 'date', 'sales', 'predicted_sales', 'Actual vs. Forecasted Sales', 'Sales (₱)'), use_container_width=True)
                else: st.warning(f"No forecast data in the last {days_to_eval} days to evaluate.")
            else: st.warning("No forecast logs found.")
        except Exception as e: st.error(f"Could not build report: {e}")
        
    with tabs[3]: # Add Data
        st.subheader("✍️ Add New Daily Record")
        with st.form("new_record_form", clear_on_submit=True):
            new_date, new_sales, new_customers = st.date_input("Date", date.today()), st.number_input("Total Sales (₱)", 0.0, format="%.2f"), st.number_input("Customer Count", 0)
            if st.form_submit_button("✅ Save Record", use_container_width=True):
                add_to_firestore(db, 'historical_data', {"date":new_date, "sales":new_sales, "customers":new_customers}, st.session_state.historical_df)
                st.success("Record added!"); st.cache_data.clear(); time.sleep(1); st.rerun()

    with tabs[4]: # Activities
        st.subheader("📅 Future Activities")
        c1, c2 = st.columns([1,2])
        with c1.form("new_activity_form", clear_on_submit=True):
            activity_name = st.text_input("Activity/Event Name")
            activity_date = st.date_input("Date of Activity", min_value=date.today())
            potential_sales = st.number_input("Potential Sales (₱)", 0.0, format="%.2f")
            remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
            if st.form_submit_button("✅ Save Activity", use_container_width=True):
                if activity_name and activity_date:
                    add_to_firestore(db, 'future_activities', {"activity_name": activity_name, "date": pd.to_datetime(activity_date), "potential_sales": potential_sales, "remarks": remarks}, pd.DataFrame())
                    st.success(f"Activity '{activity_name}' saved!"); st.cache_data.clear(); time.sleep(1); st.rerun()
        
        with c2:
            upcoming_df = st.session_state.events_df[pd.to_datetime(st.session_state.events_df['date']).dt.date >= date.today()].copy()
            if upcoming_df.empty:
                st.info("No upcoming activities scheduled.")
            else:
                for _, row in upcoming_df.iterrows():
                    render_activity_card(row, db)

    with tabs[5]: # History
        st.subheader("📜 View & Edit Historical Data")
        df = st.session_state.historical_df.copy()
        if not df.empty:
            year = st.selectbox("Select Year:", sorted(df['date'].dt.year.unique(), reverse=True))
            month_str = st.selectbox("Select Month:", sorted(df[df['date'].dt.year == year]['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True))
            month = pd.to_datetime(month_str, format='%B').month
            filtered_df = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]
            for _, row in filtered_df.iterrows():
                render_historical_record(row, db)
