# data_processing.py
# Prepares data for the new Hybrid Ensemble model.

import pandas as pd
import numpy as np
import requests
from datetime import timedelta

def get_weather_data(start_date, end_date):
    """Fetches historical and future weather data from the Open-Meteo API."""
    try:
        latitude = 10.48
        longitude = 123.42
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
               f"&daily=weathercode,temperature_2m_max,precipitation_sum,windspeed_10m_max"
               f"&timezone=Asia/Manila&start_date={start_str}&end_date={end_str}")
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        weather_df = pd.DataFrame(data['daily'])
        weather_df.rename(columns={
            'time': 'date', 'temperature_2m_max': 'weather_temp',
            'precipitation_sum': 'weather_precip', 'windspeed_10m_max': 'weather_wind',
            'weathercode': 'weather_code'
        }, inplace=True)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        return weather_df
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch weather data. Error: {e}")
        return pd.DataFrame()

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses raw data from a Firestore collection."""
    if db_client is None: return pd.DataFrame()
    try:
        docs = db_client.collection(collection_name).stream()
        records = [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"Error reading from Firestore: {e}")
        return pd.DataFrame()
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' not in df.columns: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    return df.sort_values(by='date').reset_index(drop=True)

def create_hybrid_features(df, weather_df):
    """
    Creates the full feature set for the Hybrid Ensemble model.
    """
    df_copy = df.copy()
    
    # Merge weather data
    if weather_df is not None and not weather_df.empty:
        df_copy = pd.merge(df_copy, weather_df, on='date', how='left')
        # Use forward fill and back fill to handle any missing weather points
        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)

    # --- Time-Based Features ---
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    
    # --- Weekly Pattern Features ---
    if 'customers' in df_copy.columns:
        df_copy['customers_lag_7'] = df_copy['customers'].shift(7)
        grouped = df_copy.groupby('dayofweek')['customers']
        df_copy['customers_rolling_mean_4_weeks_same_day'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
        )

    # Ensure all feature columns exist, even if weather failed
    weather_cols = ['weather_temp', 'weather_precip', 'weather_wind', 'weather_code']
    for col in weather_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0.0

    return df_copy.bfill()

def create_atv_features(df):
    """Feature engineering for ATV. (Unchanged)"""
    df_copy = df.copy()
    df_copy.sort_values(by='date', inplace=True)
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)
    return df_copy
