# data_processing.py
import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data from a Firestore collection, ensuring no duplicate dates."""
    if db_client is None:
        return pd.DataFrame()
    
    docs = db_client.collection(collection_name).stream()
    records = []
    for doc in docs:
        record = doc.to_dict()
        record['doc_id'] = doc.id
        records.append(record)
        
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' not in df.columns:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert(None)
        df['date'] = df['date'].dt.normalize()

    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_advanced_features(df, events_df):
    """
    Creates a rich feature set for a unified Gradient Boosting model from the full time series.
    """
    df_copy = df.copy()

    # --- Foundational Metrics (Cleansed) ---
    customers_safe = df_copy['customers'].replace(0, np.nan)
    
    # Calculate ATV based ONLY on base sales to remove outlier impact.
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    df_copy['atv'] = (base_sales / customers_safe)

    # --- Time-Based Features ---
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day'] = df_copy['date'].dt.day
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('int')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['time_idx'] = (df_copy['date'] - df_copy['date'].min()).dt.days

    # --- Cyclical & Event Features ---
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']

    # --- Handle Categorical Weather Data ---
    if 'weather' in df_copy.columns:
        df_copy['weather'] = df_copy['weather'].fillna('Unknown')
        weather_dummies = pd.get_dummies(df_copy['weather'], prefix='weather', dtype=int)
        df_copy = pd.concat([df_copy, weather_dummies], axis=1)
        df_copy.drop('weather', axis=1, inplace=True)

    # --- Advanced Time Series Features ---
    # Using a focused list of variables for lag/rolling features
    target_vars = ['sales', 'customers', 'atv']
    
    lag_days = [1, 2, 7, 14] 
    for var in target_vars:
        for lag in lag_days:
            df_copy[f'{var}_lag_{lag}'] = df_copy[var].shift(lag)

    windows = [7, 14]
    for var in target_vars:
        for w in windows:
            series_shifted = df_copy[var].shift(1)
            df_copy[f'{var}_rolling_mean_{w}'] = series_shifted.rolling(window=w, min_periods=1).mean()
            df_copy[f'{var}_rolling_std_{w}'] = series_shifted.rolling(window=w, min_periods=1).std()

    df_copy['dayofyear_sin'] = np.sin(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['dayofyear_cos'] = np.cos(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['weekofyear_sin'] = np.sin(2 * np.pi * df_copy['weekofyear'] / 52)
    df_copy['weekofyear_cos'] = np.cos(2 * np.pi * df_copy['weekofyear'] / 52)
    
    # --- External Regressors (Events & Holidays) ---
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date']], on='date', how='left', indicator='is_event')
        df_copy['is_event'] = (df_copy['is_event'] == 'both').astype(int)
    else:
        df_copy['is_event'] = 0

    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0
        
    return df_copy.fillna(0)
