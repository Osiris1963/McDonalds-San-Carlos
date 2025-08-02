# forecasting.py
# Final version with a specialist LSTM model for customer forecasting.
# Corrected to be robust against weather API failures.

import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Import TensorFlow for the LSTM model.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_processing import create_customer_features, create_atv_features, get_weather_data

def create_lstm_sequences(data, n_steps, features, target):
    """
    Transforms a time-series dataframe into sequences for LSTM training.
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x = data[features].iloc[i:end_ix].values
        seq_y = data[target].iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_customer_model(historical_df, periods):
    """
    Trains a weather-aware LSTM model to forecast customer counts.
    """
    # 1. Fetch complete weather history
    start_date = historical_df['date'].min() - timedelta(days=30)
    end_date = historical_df['date'].max() + timedelta(days=periods)
    weather_df = get_weather_data(start_date, end_date)

    # 2. Create base features
    df_featured = create_customer_features(historical_df, weather_df)
    
    # 3. Define features to be used by the LSTM
    # Start with a base list of features that are always present.
    features = [
        'dayofyear', 'month', 'is_weekend', 'customers_lag_7'
    ]
    
    # --- ROBUSTNESS FIX ---
    # Dynamically add weather features ONLY if they exist in the dataframe.
    weather_cols = ['weather_temp', 'weather_precip', 'weather_wind', 'weather_code']
    for col in weather_cols:
        if col in df_featured.columns:
            features.append(col)
    # --- END FIX ---
            
    target = 'customers'
    
    # 4. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    df_scaled = df_featured.copy()
    df_scaled[features] = scaler.fit_transform(df_featured[features])
    df_scaled[target] = scaler_target.fit_transform(df_featured[[target]])

    # 5. Create sequences based on weekly patterns
    n_steps = 4
    all_predictions = []
    
    for day in range(7):
        day_df = df_scaled[df_scaled['dayofweek'] == day]
        if len(day_df) < n_steps + 1:
            continue

        X, y = create_lstm_sequences(day_df, n_steps, features, target)
        
        if len(X) == 0: continue

        # 6. Build and Train the LSTM Model
        model = Sequential([
            LSTM(units=50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=1, verbose=0)

        # 7. Forecast for the future
        last_sequence = day_df[features].tail(n_steps).values
        current_batch = last_sequence.reshape((1, n_steps, len(features)))
        
        future_dates_for_day = pd.date_range(start=historical_df['date'].max() + timedelta(days=1), periods=periods)
        future_dates_for_day = future_dates_for_day[future_dates_for_day.dayofweek == day]
        
        for future_date in future_dates_for_day:
            pred = model.predict(current_batch, verbose=0)[0]
            all_predictions.append({'ds': future_date, 'forecast_customers_scaled': pred[0]})

    if not all_predictions:
        return pd.DataFrame()

    # 8. Combine and inverse scale the predictions
    forecast_df = pd.DataFrame(all_predictions)
    if 'forecast_customers_scaled' in forecast_df.columns and not forecast_df.empty:
        forecast_df['forecast_customers'] = scaler_target.inverse_transform(forecast_df[['forecast_customers_scaled']]).flatten()
    else:
        forecast_df['forecast_customers'] = 0

    return forecast_df[['ds', 'forecast_customers']]


def train_atv_model(historical_df, events_df, periods):
    """Trains a Prophet model to forecast ATV. (Unchanged)"""
    df_atv = create_atv_features(historical_df)
    df_prophet = df_atv[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
    last_hist_date = historical_df['date'].max()
    future_end_date = last_hist_date + timedelta(days=periods)
    payday_dates = []
    current_date = historical_df['date'].min()
    while current_date <= future_end_date:
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_dates.append(current_date)
        current_date += timedelta(days=1)
    paydays = pd.DataFrame({'holiday': 'payday_period', 'ds': pd.to_datetime(payday_dates), 'lower_window': 0, 'upper_window': 1})
    holidays = paydays
    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        holidays = pd.concat([paydays, user_events])
    model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast_df = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat']]
    forecast_df.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    return forecast_df, model


def generate_forecast(historical_df, events_df, periods=15):
    """Orchestrates the new LSTM + Prophet forecasting process."""
    if len(historical_df) < 60:
        return pd.DataFrame(), None
    
    cust_forecast_df = train_customer_model(historical_df, periods)
    atv_forecast_df, prophet_model_for_ui = train_atv_model(historical_df, events_df, periods)
    
    if cust_forecast_df.empty or atv_forecast_df.empty:
        return pd.DataFrame(), None
        
    final_df = pd.merge(cust_forecast_df, atv_forecast_df, on='ds', how='left').sort_values('ds')
    final_df['forecast_atv'].fillna(method='ffill', inplace=True)
    
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round().astype(int)
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    
    return final_df.head(periods), prophet_model_for_ui
