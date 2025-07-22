import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objs as go
import io
import time
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Suppress informational messages for cleaner output ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

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
        .block-container {
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        [data-testid="stSidebar"] {
            background-color: #252525;
            border-right: 1px solid #444;
            width: 320px !important;
        }
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
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- App State Management ---
def initialize_state(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {
        'forecast_df': pd.DataFrame(),
        'metrics': {},
        'name': "Store 688",
        'forecast_components': pd.DataFrame(),
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
        record['doc_id'] = doc.id # Capture the document ID
        records.append(record)
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
    return cleaned_df

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    return df

def create_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # --- NEW: Interaction Features ---
    df['weekend_month_interaction'] = df['is_weekend'] * df['month']
    df['ber_month'] = (df['month'] >= 9).astype(int)
    df['weekend_ber_interaction'] = df['is_weekend'] * df['ber_month']

    df = df.sort_values('date')
    if 'sales' in df.columns:
        df['sales_lag_7'] = df['sales'].shift(7)
    if 'customers' in df.columns:
        df['customers_lag_7'] = df['customers'].shift(7)
    if 'sales' in df.columns:
        df['sales_rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    return df

def generate_recurring_local_events(start_date, end_date):
    local_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [15, 30]:
            local_events.append({'holiday': 'Payday', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    return pd.DataFrame(local_events)

# --- Core Forecasting Models ---

def train_prophet_model(df_train, events_df, target_col):
    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    recurring_events = generate_recurring_local_events(df_train['date'].min(), df_train['date'].max() + timedelta(days=30))
    manual_events = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_events = pd.concat([manual_events, recurring_events]).dropna(subset=['ds', 'holiday'])

    prophet_model = Prophet(
        holidays=all_events,
        daily_seasonality=False,
        weekly_seasonality=False, # Disable default to add a custom one
        yearly_seasonality=len(df_train) >= 365,
        changepoint_prior_scale=0.15,
    )
    prophet_model.add_seasonality(name='weekly', period=7, fourier_order=20)
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    return prophet_model

def train_xgboost_model(df_train, target_col):
    df_featured = create_advanced_features(df_train.copy())
    features = [
        'dayofyear', 'month', 'year', 'weekofyear', 
        'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7',
        'is_weekend', 'dayofweek_sin', 'dayofweek_cos',
        'weekend_month_interaction', 'weekend_ber_interaction' # New interaction features
    ]
    features = [f for f in features if f in df_featured.columns and f != target_col]
    
    df_featured.dropna(subset=features, inplace=True)
    if df_featured.empty: return None, None

    X, y = df_featured[features], df_featured[target_col]
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, features

def train_lstm_model(df_train, target_col, sequence_length=14):
    df_featured = create_advanced_features(df_train.copy())
    lstm_features = [
        'dayofyear', 'month', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7',
        'is_weekend', 'dayofweek_sin', 'dayofweek_cos',
        'weekend_month_interaction', 'weekend_ber_interaction' # New interaction features
    ]
    lstm_features = [f for f in lstm_features if f in df_featured.columns]
    
    cols_for_scaling = [target_col] + lstm_features
    data = df_featured[cols_for_scaling].dropna()

    if len(data) < sequence_length + 1: return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0: return None, None, None

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
    
    return model, scaler, lstm_features

def forecast_recursive(model, initial_history, periods, target_col, model_type='xgb', features_list=None, scaler=None, sequence_length=14):
    history = initial_history.copy()
    predictions = []

    for _ in range(periods):
        last_date = history['date'].max()
        next_date = last_date + timedelta(days=1)
        
        future_step_df = pd.DataFrame([{'date': next_date, target_col: np.nan}])
        history = pd.concat([history, future_step_df], ignore_index=True)
        
        extended_featured = create_advanced_features(history)
        
        if model_type == 'xgb':
            X_future = extended_featured[features_list].tail(1)
            prediction = model.predict(X_future)[0]
        
        elif model_type == 'lstm':
            cols_for_scaling = [target_col] + features_list
            last_sequence_unscaled = extended_featured[cols_for_scaling].tail(sequence_length)
            
            last_sequence_scaled = scaler.transform(last_sequence_unscaled)
            input_seq = last_sequence_scaled.reshape(1, sequence_length, len(cols_for_scaling))
            
            predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
            
            dummy_for_inverse = np.zeros((1, len(cols_for_scaling)))
            dummy_for_inverse[0, 0] = predicted_scaled
            
            prediction = scaler.inverse_transform(dummy_for_inverse)[0, 0]

        predictions.append({'ds': next_date, 'yhat': prediction})
        
        history.loc[history.index.max(), target_col] = prediction

    return pd.DataFrame(predictions)

# --- Main Stacking Ensemble Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def generate_stacked_forecast(_historical_df, _events_df, periods, target_col):
    st.session_state.status_messages.append(f"Processing target: **{target_col.title()}**")
    
    tscv = TimeSeriesSplit(n_splits=2, test_size=periods)
    train_index, val_index = list(tscv.split(_historical_df))[-1]
    
    df_train = _historical_df.iloc[train_index]
    df_val = _historical_df.iloc[val_index]

    meta_features = pd.DataFrame(index=df_val.index)

    # 1. Prophet
    st.session_state.status_messages.append("Training Prophet...")
    prophet_model = train_prophet_model(df_train, _events_df, target_col)
    future_val_prophet = prophet_model.make_future_dataframe(periods=len(df_val))
    forecast_val_prophet = prophet_model.predict(future_val_prophet)
    meta_features['prophet_preds'] = forecast_val_prophet['yhat'].tail(len(df_val)).values
    
    # 2. XGBoost
    st.session_state.status_messages.append("Training XGBoost...")
    xgb_model, xgb_features = train_xgboost_model(df_train, target_col)
    if xgb_model:
        df_val_featured = create_advanced_features(pd.concat([df_train, df_val]))
        X_val = df_val_featured[xgb_features].tail(len(df_val))
        meta_features['xgb_preds'] = xgb_model.predict(X_val)
    else:
        meta_features['xgb_preds'] = 0

    # 3. LSTM
    st.session_state.status_messages.append("Training LSTM...")
    lstm_model, lstm_scaler, lstm_features = train_lstm_model(df_train, target_col)
    if lstm_model:
        cols_for_scaling = [target_col] + lstm_features
        full_featured_df = create_advanced_features(_historical_df)
        data_for_scaling = full_featured_df[cols_for_scaling].dropna()
        scaled_data = lstm_scaler.transform(data_for_scaling)
        
        X_full = []
        for i in range(len(scaled_data) - lstm_model.input_shape[1]):
            X_full.append(scaled_data[i:i + lstm_model.input_shape[1]])
        X_full = np.array(X_full)

        if len(X_full) >= len(df_val):
            X_val_lstm = X_full[-len(df_val):]
            preds_scaled = lstm_model.predict(X_val_lstm, verbose=0)
            dummy = np.zeros((len(preds_scaled), len(cols_for_scaling)))
            dummy[:, 0] = preds_scaled.flatten()
            lstm_predictions = lstm_scaler.inverse_transform(dummy)[:, 0]
            meta_features['lstm_preds'] = lstm_predictions
        else:
            meta_features['lstm_preds'] = 0
    else:
        meta_features['lstm_preds'] = 0

    # 4. Train Meta-Model
    st.session_state.status_messages.append("Training Meta-Model...")
    meta_model = LinearRegression()
    y_val = df_val[target_col]
    meta_model.fit(meta_features.fillna(0), y_val)

    # 5. Final Forecast
    st.session_state.status_messages.append("Generating final forecast...")
    final_prophet_model = train_prophet_model(_historical_df, _events_df, target_col)
    final_xgb_model, final_xgb_features = train_xgboost_model(_historical_df, target_col)
    final_lstm_model, final_lstm_scaler, final_lstm_features = train_lstm_model(_historical_df, target_col)
    
    future_df = final_prophet_model.make_future_dataframe(periods=periods)
    prophet_future_preds = final_prophet_model.predict(future_df)['yhat'].tail(periods)

    xgb_future_preds = pd.Series(dtype='float64')
    if final_xgb_model:
        xgb_future_preds = forecast_recursive(final_xgb_model, _historical_df, periods, target_col, model_type='xgb', features_list=final_xgb_features)['yhat']
    
    lstm_future_preds = pd.Series(dtype='float64')
    if final_lstm_model:
        lstm_future_preds = forecast_recursive(final_lstm_model, _historical_df, periods, target_col, model_type='lstm', features_list=final_lstm_features, scaler=final_lstm_scaler)['yhat']

    future_meta_features = pd.DataFrame({
        'prophet_preds': prophet_future_preds.values,
        'xgb_preds': xgb_future_preds.values if not xgb_future_preds.empty else np.nan,
        'lstm_preds': lstm_future_preds.values if not lstm_future_preds.empty else np.nan
    })

    future_meta_features = future_meta_features.apply(lambda x: x.fillna(x.mean()), axis=1)

    final_predictions = meta_model.predict(future_meta_features.fillna(0))
    
    forecast_df = pd.DataFrame({'ds': future_df['ds'].tail(periods), 'yhat': final_predictions})
    forecast_components = final_prophet_model.predict(future_df)
    
    return forecast_df, forecast_components

# --- Firestore Data I/O ---
def update_historical_record_in_firestore(db_client, doc_id, data):
    if db_client is None: return
    db_client.collection('historical_data').document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client is None: return
    db_client.collection(collection_name).document(doc_id).delete()

# --- Plotting & UI Functions ---
def plot_forecast_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_sales'], mode='lines+markers', name='Sales Forecast', line=dict(color='#ffc72c')))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_customers'], mode='lines+markers', name='Customer Forecast', yaxis='y2', line=dict(color='#c8102e')))
    fig.update_layout(
        title=title, xaxis_title='Date',
        yaxis=dict(title='Predicted Sales (₱)', color='#ffc72c'),
        yaxis2=dict(title='Predicted Customers', overlaying='y', side='right', color='#c8102e'),
        legend=dict(x=0.01, y=0.99, orientation='h'), height=500,
        paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white'
    )
    return fig

def plot_forecast_breakdown(components, selected_date):
    day_data = components[components['ds'] == selected_date].iloc[0]
    x_data, y_data, measure_data = ['Baseline Trend'], [day_data.get('trend', 0)], ["absolute"]
    
    effects = {'Day of Week': 'weekly', 'Time of Year': 'yearly', 'Holidays/Events': 'holidays'}
    for name, key in effects.items():
        if key in day_data and pd.notna(day_data[key]) and day_data[key] != 0:
            x_data.append(name); y_data.append(day_data[key]); measure_data.append('relative')
            
    x_data.append('Final Forecast'); y_data.append(day_data['yhat']); measure_data.append('total')
    
    fig = go.Figure(go.Waterfall(
        name="Breakdown", orientation="v", measure=measure_data, x=x_data,
        text=[f"{v:,.0f}" for v in y_data], y=y_data,
        increasing={"marker":{"color":"#2ca02c"}}, decreasing={"marker":{"color":"#d62728"}},
        totals={"marker":{"color":"#1f77b4"}}
    ))
    fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A, %B %d')}", showlegend=False, paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
    return fig, day_data

def render_historical_record(row, db_client):
    date_str = pd.to_datetime(row['date']).strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            edit_cols = st.columns(3)
            updated_sales = edit_cols[0].number_input("Sales (₱)", value=float(row.get('sales', 0)), format="%.2f", key=f"sales_{row['doc_id']}")
            updated_customers = edit_cols[1].number_input("Customers", value=int(row.get('customers', 0)), key=f"cust_{row['doc_id']}")
            updated_addons = edit_cols[2].number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales', 0)), format="%.2f", key=f"addon_{row['doc_id']}")
            
            btn_cols = st.columns(2)
            if btn_cols[0].form_submit_button("💾 Update Record", use_container_width=True):
                update_data = {'sales': updated_sales, 'customers': updated_customers, 'add_on_sales': updated_addons}
                update_historical_record_in_firestore(db_client, row['doc_id'], update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

            if btn_cols[1].form_submit_button("🗑️ Delete Record", use_container_width=True, type="primary"):
                delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                st.warning(f"Record for {date_str} deleted.")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()

if db:
    initialize_state(db)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("Sales & Customer AI")
        st.markdown("---")
        st.info("Forecasting with a Stacking Ensemble AI (Prophet + XGBoost + LSTM)")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.success("Cache cleared. Rerunning..."); time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 90:
                st.error("Please provide at least 90 days of data for reliable forecasting.")
            else:
                st.session_state.status_messages = []
                status_placeholder = st.empty()
                with st.spinner("🧠 Initializing Advanced AI Forecast..."):
                    status_text = st.empty()
                    base_df = st.session_state.historical_df.copy()
                    
                    # Calculate base_sales for training
                    hist_df_with_components = calculate_atv(base_df)
                    
                    ev_df = st.session_state.events_df.copy()
                    
                    FORECAST_HORIZON = 15
                    
                    # --- FORECAST 1: Customers ---
                    status_text.text("Running Customer Forecast...")
                    cust_f, cust_components = generate_stacked_forecast(hist_df_with_components.copy(), ev_df, FORECAST_HORIZON, 'customers')
                    
                    # --- FORECAST 2: Base Sales ---
                    status_text.text("Running Base Sales Forecast...")
                    sales_f, _ = generate_stacked_forecast(hist_df_with_components.copy(), ev_df, FORECAST_HORIZON, 'base_sales')
                    
                    status_text.text("Combining forecasts...")
                    if not cust_f.empty and not sales_f.empty:
                        # Combine the two primary forecasts
                        combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), sales_f.rename(columns={'yhat':'forecast_base_sales'}), on='ds')
                        
                        # Derive ATV from the primary forecasts
                        combo_f['forecast_atv'] = combo_f['forecast_base_sales'] / combo_f['forecast_customers'].replace(0, np.nan)
                        combo_f['forecast_atv'].fillna(0, inplace=True)

                        # Estimate total sales by adding an average add-on value
                        avg_add_on = hist_df_with_components['add_on_sales'].mean()
                        combo_f['forecast_sales'] = combo_f['forecast_base_sales'] + avg_add_on

                        st.session_state.forecast_df = combo_f
                        st.session_state.forecast_components = cust_components
                        st.success("Advanced AI forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed.")
                status_text.empty()

        st.markdown("---")
        st.download_button("📥 Download Forecast", st.session_state.forecast_df.to_csv(index=False), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", st.session_state.historical_df.to_csv(index=False), "historical_data.csv", "text/csv", use_container_width=True)

    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Add/Edit Data", "📜 Historical Data"])

    with tabs[0]: # Forecast Dashboard
        if not st.session_state.forecast_df.empty:
            today = pd.to_datetime('today').normalize()
            future_forecast_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'] >= today].copy()
            if future_forecast_df.empty:
                st.warning("Forecast contains no future dates.")
            else:
                disp_cols = {'ds':'Date', 'forecast_customers':'Predicted Customers', 'forecast_atv':'Predicted Avg Sale (₱)', 'forecast_sales':'Predicted Sales (₱)'}
                display_df = future_forecast_df.rename(columns=disp_cols)[list(disp_cols.values())]
                st.markdown("#### Forecasted Values")
                st.dataframe(display_df.set_index('Date').style.format({'Predicted Customers':'{:,.0f}', 'Predicted Avg Sale (₱)':'₱{:,.2f}', 'Predicted Sales (₱)':'₱{:,.2f}'}), use_container_width=True, height=560)
                st.markdown("#### Forecast Visualization")
                fig = plot_forecast_chart(future_forecast_df, '15-Day Sales & Customer Forecast')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    with tabs[1]: # Forecast Insights
        st.info("The breakdown below is from the Prophet model, showing the main drivers of the customer forecast.")
        if st.session_state.forecast_components.empty:
            st.info("Generate a forecast first to see insights.")
        else:
            future_components = st.session_state.forecast_components[st.session_state.forecast_components['ds'] >= pd.to_datetime('today').normalize()].copy()
            if not future_components.empty:
                future_components['date_str'] = future_components['ds'].dt.strftime('%A, %B %d, %Y')
                selected_date_str = st.selectbox("Select a day to analyze:", options=future_components['date_str'])
                selected_date = future_components[future_components['date_str'] == selected_date_str]['ds'].iloc[0]
                breakdown_fig, _ = plot_forecast_breakdown(st.session_state.forecast_components, selected_date)
                st.plotly_chart(breakdown_fig, use_container_width=True)
            else:
                st.warning("No future dates available in the forecast to analyze.")

    with tabs[2]: # Add/Edit Data
        st.subheader("✍️ Add New Daily Record")
        with st.form("new_record_form", clear_on_submit=True, border=True):
            new_date = st.date_input("Date", date.today())
            new_sales = st.number_input("Total Sales (₱)", min_value=0.0, format="%.2f")
            new_customers = st.number_input("Customer Count", min_value=0)
            new_addons = st.number_input("Add-on Sales (₱)", min_value=0.0, format="%.2f")
            if st.form_submit_button("✅ Save Record"):
                new_rec = {"date": pd.to_datetime(new_date), "sales": new_sales, "customers": new_customers, "add_on_sales": new_addons}
                db.collection('historical_data').add(new_rec)
                st.cache_data.clear()
                st.success("Record added! Refresh data to see changes."); time.sleep(1); st.rerun()
    
    with tabs[3]: # Historical Data
        st.subheader("View & Edit Historical Data")
        df = st.session_state.historical_df.copy()
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            all_years = sorted(df['date'].dt.year.unique(), reverse=True)
            if all_years:
                filter_cols = st.columns(2)
                selected_year = filter_cols[0].selectbox("Select Year:", options=all_years)
                df_year = df[df['date'].dt.year == selected_year]
                all_months = sorted(df_year['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)
                if all_months:
                    selected_month_str = filter_cols[1].selectbox("Select Month:", options=all_months)
                    selected_month_num = pd.to_datetime(selected_month_str, format='%B').month
                    filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].sort_values('date', ascending=False)
                    if not filtered_df.empty:
                        for _, row in filtered_df.iterrows():
                            render_historical_record(row, db)
                    else: st.info("No data for the selected month and year.")
                else: st.info(f"No data available for the year {selected_year}.")
            else: st.info("No historical data to display.")
        else: st.info("No historical data available.")

else:
    st.error("Failed to connect to the database. Please check your Firestore credentials in Streamlit Secrets.")

