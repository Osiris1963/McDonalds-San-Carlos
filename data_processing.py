# data_processing.py
# This module prepares data by creating specialized feature sets for different models.

import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """
    Loads and preprocesses raw data from a Firestore collection.
    Ensures data types are correct and there are no duplicate dates.
    """
    if db_client is None:
        print("Warning: Firestore client is not available.")
        return pd.DataFrame()
    
    try:
        docs = db_client.collection(collection_name).stream()
        records = [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"Error reading from Firestore collection {collection_name}: {e}")
        return pd.DataFrame()
        
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' not in df.columns:
        print("Error: 'date' column not found in the data.")
        return pd.DataFrame()

    # --- Data Cleaning and Type Conversion ---
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
            df[col] = 0 # Ensure column exists

    return df.sort_values(by='date').reset_index(drop=True)

def create_customer_features(df):
    """
    Feature engineering tailored for CUSTOMER COUNT prediction with LightGBM.
    Focuses on weekly seasonality and cyclical patterns.
    """
    df_copy = df.copy()
    df_copy.sort_values(by='date', inplace=True)

    # --- Time-Based Features ---
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('Int64')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)

    # --- Core Logic: Weekly Pattern Features ---
    if 'customers' in df_copy.columns:
        # Lag features based on weekly cycles
        df_copy['customers_lag_7'] = df_copy['customers'].shift(7)
        df_copy['customers_lag_14'] = df_copy['customers'].shift(14)
        
        # Rolling averages based on the same day of previous weeks
        grouped = df_copy.groupby('dayofweek')['customers']
        df_copy['customers_rolling_mean_4_weeks_same_day'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
        )
        df_copy['customers_rolling_std_4_weeks_same_day'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=1).std()
        )

    # Use backfill to handle NaNs created by initial lags/rolls without losing data
    return df_copy.bfill()

def create_atv_features(df):
    """
    Feature engineering tailored for AVERAGE TRANSACTION VALUE (ATV) prediction with Prophet.
    This function is simpler as Prophet handles seasonality internally. Its main job
    is to calculate the target variable 'atv'.
    """
    df_copy = df.copy()
    df_copy.sort_values(by='date', inplace=True)

    # --- Core ATV Calculation ---
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe)
    
    # Forward-fill is a robust strategy for ATV on days with no customers
    df_copy['atv'] = df_copy['atv'].fillna(method='ffill').fillna(0)

    return df_copy
