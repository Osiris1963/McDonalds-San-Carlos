# forecasting.py
# Implements a Hybrid Ensemble model: Prophet for trend, LightGBM for residuals.
# Corrected to handle column name mismatches between model stages.

import pandas as pd
from datetime import timedelta
import numpy as np
from prophet import Prophet
import lightgbm as lgb

from data_processing import create_hybrid_features, create_atv_features, get_weather_data

def train_customer_model(historical_df, events_df, periods):
    """
    Trains a Hybrid Ensemble model to forecast customer counts.
    """
    # --- Part 1: Train the Trend Specialist (Prophet) ---
    df_prophet = historical_df[['date', 'customers']].rename(columns={'date': 'ds', 'customers': 'y'})
    
    trend_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5
    )
    
    if events_df is not None and not events_df.empty:
        holidays = events_df[['date', 'activity_name']].rename(columns={'date': 'ds', 'activity_name': 'holiday'})
        trend_model.holidays = holidays
        
    trend_model.fit(df_prophet)
    
    future_dates = trend_model.make_future_dataframe(periods=periods)
    trend_forecast = trend_model.predict(future_dates)

    # --- Part 2: Train the Context Specialist (LightGBM on Residuals) ---
    
    df_residuals = pd.merge(historical_df, trend_forecast[['ds', 'yhat']], left_on='date', right_on='ds')
    df_residuals['residuals'] = df_residuals['customers'] - df_residuals['yhat']
    
    start_date = historical_df['date'].min()
    end_date = historical_df['date'].max() + timedelta(days=periods)
    weather_df = get_weather_data(start_date, end_date)
    df_featured = create_hybrid_features(df_residuals, weather_df)
    
    features = [
        'dayofweek', 'dayofyear', 'month', 'is_weekend',
        'customers_lag_7', 'customers_rolling_mean_4_weeks_same_day',
        'weather_temp', 'weather_precip', 'weather_wind'
    ]
    target = 'residuals'
    
    residual_model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, random_state=42)
    residual_model.fit(df_featured[features], df_featured[target])
    
    # --- ROBUSTNESS FIX: Standardize column name before creating features ---
    # Rename Prophet's 'ds' column to 'date' to match what create_hybrid_features expects.
    trend_forecast.rename(columns={'ds': 'date'}, inplace=True)
    # --- END FIX ---
    
    future_feature_df = create_hybrid_features(trend_forecast, weather_df)
    future_residuals = residual_model.predict(future_feature_df[features])
    
    # --- Part 3: Combine the Forecasts ---
    final_forecast = trend_forecast.copy()
    final_forecast['residual_forecast'] = future_residuals
    final_forecast['forecast_customers'] = final_forecast['yhat'] + final_forecast['residual_forecast']
    
    last_hist_date = historical_df['date'].max()
    # Use the original 'date' column (renamed from 'ds') for filtering
    return final_forecast[final_forecast['date'] > last_hist_date][['date', 'forecast_customers']].rename(columns={'date': 'ds'})


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
    # This function needs to return two values to match the call in generate_forecast
    return forecast_df, model 

def generate_forecast(historical_df, events_df, periods=15):
    """Orchestrates the new Hybrid Ensemble forecasting process."""
    if len(historical_df) < 30:
        return pd.DataFrame(), None
    
    # train_customer_model now only returns one dataframe
    cust_forecast_df = train_customer_model(historical_df, events_df, periods)
    atv_forecast_df, prophet_model_for_ui = train_atv_model(historical_df, events_df, periods)
    
    if cust_forecast_df.empty or atv_forecast_df.empty:
        return pd.DataFrame(), None
        
    final_df = pd.merge(cust_forecast_df, atv_forecast_df, on='ds', how='left').sort_values('ds')
    final_df['forecast_atv'].fillna(method='ffill', inplace=True)
    
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round().astype(int)
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    
    # We pass back the ATV model for visualization, as the customer model is now a hybrid
    return final_df.head(periods), prophet_model_for_ui
