# data_processing.py
import pandas as pd
import numpy as np
from datetime import timedelta

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data from a Firestore collection."""
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
    
    # --- Robust Date Handling ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    # --- Numeric Type Conversion ---
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Sort and return ---
    return df.sort_values(by='date').reset_index(drop=True)

def create_features(df, events_df):
    """
    This is the new, robust feature engineering pipeline.
    It creates time-based, event-based, and lag/rolling features for all key metrics.
    """
    df_copy = df.copy()
    
    # --- 1. Calculate ATV (Average Transaction Value) ---
    # Use base sales for a more stable ATV
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan) # Avoid division by zero
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    # --- 2. Foundational Time Features ---
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype(int)
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek_num'] = df_copy['date'].dt.dayofweek # For cyclical features if needed

    # --- 3. CRITICAL: Day of Week (One-Hot Encoded) ---
    df_copy['dayofweek'] = df_copy['date'].dt.day_name()
    day_dummies = pd.get_dummies(df_copy['dayofweek'], prefix='day', drop_first=False) # Keep all days
    df_copy = pd.concat([df_copy, day_dummies], axis=1)

    # --- 4. CRITICAL: Payday Feature (Context-Aware) ---
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    # --- 5. Holiday & Events Features ---
    if events_df is not None and not events_df.empty:
        events_df['date'] = pd.to_datetime(events_df['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    # --- 6. Lag & Rolling Window Features (Momentum) ---
    # Shift by 1 to prevent data leakage for training
    shift_val = 1 
    
    targets_for_features = ['sales', 'customers', 'atv']
    for target in targets_for_features:
        if target in df_copy.columns:
            # Lag features (what happened last week, two weeks ago?)
            df_copy[f'{target}_lag_7'] = df_copy[target].shift(shift_val + 6)
            df_copy[f'{target}_lag_14'] = df_copy[target].shift(shift_val + 13)

            # Rolling features (what's the recent trend?)
            df_copy[f'{target}_rolling_mean_7'] = df_copy[target].shift(shift_val).rolling(window=7).mean()
            df_copy[f'{target}_rolling_std_7'] = df_copy[target].shift(shift_val).rolling(window=7).std()

    # --- 7. ROBUST FIX: Convert all boolean/uint8 columns to integer ---
    # This ensures all feature columns are compatible with LightGBM
    for col in df_copy.columns:
        if df_copy[col].dtype == 'bool' or df_copy[col].dtype == 'uint8':
            df_copy[col] = df_copy[col].astype(int)
            
    return df_copy.fillna(0)
