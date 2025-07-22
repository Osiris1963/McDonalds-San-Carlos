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
            # Assumes secrets are set in Streamlit Cloud
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
    return cleaned_df

def calculate_atv(df):
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        atv = np.divide(df['base_sales'], df['customers'])
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

def create_advanced_features(df):
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
    recurring_events = generate_recurring_local_events(df_train['date'].min(), df_train['date'].max())
    manual_events = events_df.rename(columns={'date':'ds', 'activity_name':'holiday'})
    all_events = pd.concat([manual_events, recurring_events]).dropna(subset=['ds', 'holiday'])

    prophet_model = Prophet(
        holidays=all_events,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(df_train) >= 365,
        changepoint_prior_scale=0.15
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    return prophet_model

def train_xgboost_model(df_train, target_col):
    df_featured = create_advanced_features(df_train.copy())
    features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7']
    features = [f for f in features if f in df_featured.columns and f != target_col]
    
    df_featured.dropna(subset=features, inplace=True)
    if df_featured.empty: return None

    X, y = df_featured[features], df_featured[target_col]
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def train_lstm_model(df_train, target_col, sequence_length=14):
    df_featured = create_advanced_features(df_train.copy())
    features = ['dayofyear', 'dayofweek', 'month', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7']
    features = [f for f in features if f in df_featured.columns]
    
    data = df_featured[[target_col] + features].dropna()
    if len(data) < sequence_length + 1: return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0: return None, None

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
    
    return model, scaler

def forecast_recursive(model, initial_history, periods, target_col, model_type='xgb'):
    """Generic recursive forecasting for XGBoost and LSTM."""
    history = initial_history.copy()
    predictions = []

    for _ in range(periods):
        last_date = history['date'].max()
        next_date = last_date + timedelta(days=1)
        
        future_step_df = pd.DataFrame([{'date': next_date}])
        extended_history = pd.concat([history, future_step_df], ignore_index=True)
        extended_featured = create_advanced_features(extended_history)
        
        if model_type == 'xgb':
            features_list = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7']
            features_list = [f for f in features_list if f in extended_featured.columns and f != target_col]
            X_future = extended_featured[features_list].tail(1)
            prediction = model.predict(X_future)[0]
        
        elif model_type == 'lstm':
            features_list = ['dayofyear', 'dayofweek', 'month', 'sales_lag_7', 'customers_lag_7', 'sales_rolling_mean_7']
            features_list = [f for f in features_list if f in extended_featured.columns]
            
            # Use the scaler from training
            scaler = model.scaler_
            
            # Get the last sequence from the history
            last_sequence_unscaled = extended_featured[[target_col] + features_list].tail(model.input_shape[1] + 1)
            last_sequence_scaled = scaler.transform(last_sequence_unscaled)
            
            input_seq = last_sequence_scaled[:-1].reshape(1, model.input_shape[1], model.input_shape[2])
            
            # Predict the scaled value
            predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
            
            # Create a dummy array to inverse transform
            dummy_for_inverse = np.zeros((1, scaler.n_features_in_))
            dummy_for_inverse[0, 0] = predicted_scaled
            
            # Inverse transform to get the actual value
            prediction = scaler.inverse_transform(dummy_for_inverse)[0, 0]

        predictions.append({'ds': next_date, 'yhat': prediction})
        
        # Update history with the new prediction for the next loop
        new_row = extended_featured.tail(1).copy()
        new_row.loc[new_row.index, target_col] = prediction
        history = pd.concat([history, new_row.drop(columns=[col for col in new_row.columns if col not in history.columns], errors='ignore')], ignore_index=True)

    return pd.DataFrame(predictions)

# --- Main Stacking Ensemble Function ---
@st.cache_data(ttl=3600)
def generate_stacked_forecast(_historical_df, _events_df, periods, target_col):
    """
    Trains base models (Prophet, XGBoost, LSTM) and a meta-model (Linear Regression)
    for a sophisticated stacked ensemble forecast.
    """
    st.session_state.status_messages.append(f"Processing for target: **{target_col.title()}**")
    
    # Use TimeSeriesSplit for robust validation
    tscv = TimeSeriesSplit(n_splits=2, test_size=periods)
    train_index, val_index = list(tscv.split(_historical_df))[-1]
    
    df_train = _historical_df.iloc[train_index]
    df_val = _historical_df.iloc[val_index]

    meta_features = pd.DataFrame(index=df_val.index)

    # 1. Prophet Model
    st.session_state.status_messages.append("Training Prophet...")
    prophet_model = train_prophet_model(df_train, _events_df, target_col)
    future_val_prophet = prophet_model.make_future_dataframe(periods=len(df_val))
    forecast_val_prophet = prophet_model.predict(future_val_prophet)
    meta_features['prophet_preds'] = forecast_val_prophet['yhat'].tail(len(df_val)).values
    
    # 2. XGBoost Model
    st.session_state.status_messages.append("Training XGBoost...")
    xgb_model = train_xgboost_model(df_train, target_col)
    if xgb_model:
        df_val_featured = create_advanced_features(pd.concat([df_train, df_val]))
        features = [f for f in xgb_model.get_booster().feature_names if f in df_val_featured.columns]
        X_val = df_val_featured[features].tail(len(df_val))
        meta_features['xgb_preds'] = xgb_model.predict(X_val)
    else:
        meta_features['xgb_preds'] = 0 # Fallback

    # 3. LSTM Model
    st.session_state.status_messages.append("Training LSTM Deep Learning model...")
    lstm_model, lstm_scaler = train_lstm_model(df_train, target_col)
    if lstm_model:
        lstm_model.scaler_ = lstm_scaler # Attach scaler for recursive forecast
        
        # Create validation sequences
        full_data_scaled = lstm_scaler.transform(create_advanced_features(_historical_df)[[target_col] + [f for f in features if f in _historical_df.columns]].dropna())
        X_val_lstm = []
        for i in range(len(df_train) - lstm_model.input_shape[1], len(_historical_df) - lstm_model.input_shape[1]):
            X_val_lstm.append(full_data_scaled[i:i + lstm_model.input_shape[1]])
        X_val_lstm = np.array(X_val_lstm)
        
        if X_val_lstm.shape[0] > 0:
            preds_scaled = lstm_model.predict(X_val_lstm, verbose=0)
            dummy_for_inverse = np.zeros((len(preds_scaled), lstm_scaler.n_features_in_))
            dummy_for_inverse[:, 0] = preds_scaled.flatten()
            meta_features['lstm_preds'] = lstm_scaler.inverse_transform(dummy_for_inverse)[:, 0]
        else:
            meta_features['lstm_preds'] = 0
    else:
        meta_features['lstm_preds'] = 0 # Fallback

    # 4. Train Meta-Model
    st.session_state.status_messages.append("Training Meta-Model (Stacking)...")
    meta_model = LinearRegression()
    y_val = df_val[target_col]
    meta_model.fit(meta_features.fillna(0), y_val)

    # 5. Final Forecast Generation
    st.session_state.status_messages.append("Generating final blended forecast...")
    
    # Retrain base models on ALL data
    final_prophet_model = train_prophet_model(_historical_df, _events_df, target_col)
    final_xgb_model = train_xgboost_model(_historical_df, target_col)
    final_lstm_model, final_lstm_scaler = train_lstm_model(_historical_df, target_col)
    
    # Generate future predictions from each base model
    future_df = final_prophet_model.make_future_dataframe(periods=periods)
    prophet_future_preds = final_prophet_model.predict(future_df)['yhat'].tail(periods)

    xgb_future_preds = pd.Series(dtype='float64')
    if final_xgb_model:
        xgb_future_preds = forecast_recursive(final_xgb_model, _historical_df, periods, target_col, model_type='xgb')['yhat']
    
    lstm_future_preds = pd.Series(dtype='float64')
    if final_lstm_model:
        final_lstm_model.scaler_ = final_lstm_scaler
        lstm_future_preds = forecast_recursive(final_lstm_model, _historical_df, periods, target_col, model_type='lstm')['yhat']

    # Combine into a meta-feature DataFrame for the future
    future_meta_features = pd.DataFrame({
        'prophet_preds': prophet_future_preds.values,
        'xgb_preds': xgb_future_preds.values if not xgb_future_preds.empty else 0,
        'lstm_preds': lstm_future_preds.values if not lstm_future_preds.empty else 0
    })

    # Get final stacked prediction
    final_predictions = meta_model.predict(future_meta_features.fillna(0))
    
    forecast_df = pd.DataFrame({
        'ds': future_df['ds'].tail(periods),
        'yhat': final_predictions
    })

    # For breakdown chart, use the Prophet model's components
    prophet_components_model = train_prophet_model(_historical_df, _events_df, target_col)
    full_future = prophet_components_model.make_future_dataframe(periods=periods)
    forecast_components = prophet_components_model.predict(full_future)
    
    return forecast_df, forecast_components

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
        if key in day_data and pd.notna(day_data[key]):
            x_data.append(name)
            y_data.append(day_data[key])
            measure_data.append('relative')
            
    x_data.append('Final Forecast'); y_data.append(day_data['yhat']); measure_data.append('total')
    
    fig = go.Figure(go.Waterfall(
        name="Breakdown", orientation="v", measure=measure_data, x=x_data,
        text=[f"{v:,.0f}" for v in y_data], y=y_data,
        increasing={"marker":{"color":"#2ca02c"}}, decreasing={"marker":{"color":"#d62728"}},
        totals={"marker":{"color":"#1f77b4"}}
    ))
    fig.update_layout(title=f"Forecast Breakdown for {selected_date.strftime('%A, %B %d')}", showlegend=False, paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white')
    return fig, day_data

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
            st.success("Cache cleared. Rerunning...")
            time.sleep(1)
            st.rerun()

        if st.button("📈 Generate Forecast", use_container_width=True):
            if len(st.session_state.historical_df) < 90: # Need more data for LSTM
                st.error("Please provide at least 90 days of data for reliable forecasting.")
            else:
                st.session_state.status_messages = []
                status_placeholder = st.empty()

                with st.spinner("🧠 Initializing Advanced AI Forecast..."):
                    base_df = st.session_state.historical_df.copy()
                    base_df['base_sales'] = base_df['sales'] - base_df['add_on_sales']
                    cleaned_df = remove_outliers_iqr(base_df, column='base_sales')
                    hist_df_with_atv = calculate_atv(cleaned_df)
                    ev_df = st.session_state.events_df.copy()
                    
                    FORECAST_HORIZON = 15
                    
                    # --- Customer Forecast ---
                    status_placeholder.text("Running Customer Forecast...")
                    cust_f, cust_components = generate_stacked_forecast(hist_df_with_atv.copy(), ev_df, FORECAST_HORIZON, 'customers')
                    
                    # --- ATV Forecast ---
                    status_placeholder.text("Running Average Transaction Forecast...")
                    atv_f, _ = generate_stacked_forecast(hist_df_with_atv.copy(), ev_df, FORECAST_HORIZON, 'atv')
                    
                    status_placeholder.text("Combining forecasts...")
                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                        
                        st.session_state.forecast_df = combo_f
                        st.session_state.forecast_components = cust_components
                        
                        st.success("Advanced AI forecast generated successfully!")
                    else:
                        st.error("Forecast generation failed.")
                status_placeholder.empty()

        st.markdown("---")
        st.download_button("📥 Download Forecast", st.session_state.forecast_df.to_csv(index=False), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
        st.download_button("📥 Download Historical", st.session_state.historical_df.to_csv(index=False), "historical_data.csv", "text/csv", use_container_width=True)

    # --- Main Content Tabs ---
    tabs = st.tabs(["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Add/Edit Data"])

    with tabs[0]:
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
            st.info("Click the 'Generate Forecast' button in the sidebar to begin.")

    with tabs[1]:
        st.info("The breakdown below is generated by the Prophet model component, showing the main drivers of the customer forecast.")
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

    with tabs[2]:
        st.subheader("✍️ Add New Daily Record")
        with st.form("new_record_form", clear_on_submit=True, border=True):
            new_date = st.date_input("Date", date.today())
            new_sales = st.number_input("Total Sales (₱)", min_value=0.0, format="%.2f")
            new_customers = st.number_input("Customer Count", min_value=0)
            new_addons = st.number_input("Add-on Sales (₱)", min_value=0.0, format="%.2f")
            
            if st.form_submit_button("✅ Save Record"):
                new_rec = {
                    "date": pd.to_datetime(new_date),
                    "sales": new_sales,
                    "customers": new_customers,
                    "add_on_sales": new_addons
                }
                db.collection('historical_data').add(new_rec)
                st.cache_data.clear()
                st.success("Record added! Refresh data to include it in the next forecast.")
                time.sleep(1)
                st.rerun()

else:
    st.error("Failed to connect to the database. Please check your Firestore credentials in Streamlit Secrets.")

