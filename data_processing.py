# Revised data_processing.py

import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data from a Firestore collection, ensuring no duplicate dates."""
    # This function is fine as-is, no changes needed from the previous version.
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
    
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_features(df, events_df):
    """
    Creates a rich feature set for a single, unified forecasting model,
    with a strong emphasis on weekly patterns and same-day historical performance.
    """
    df_copy = df.copy()
    df_copy.sort_values(by='date', inplace=True)

    # --- Core Value Calculation ---
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    # --- Time-Based Features ---
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek # Monday=0, Sunday=6
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('Int64')
    df_copy['year'] = df_copy['date'].dt.year

    # --- PRIMARY FEATURES: Weekly Lags & Same-Day Rolling Averages ---
    for target in ['sales', 'customers', 'atv']:
        if target in df_copy.columns:
            # Simple weekly lags
            df_copy[f'{target}_lag_7'] = df_copy[target].shift(7)
            df_copy[f'{target}_lag_14'] = df_copy[target].shift(14)
            
            # --- NEW: Same-Day Rolling Averages ---
            # This calculates the average of the last 4 weeks for that specific day of the week.
            # It is the direct implementation of your business logic.
            grouped = df_copy.groupby('dayofweek')[target]
            df_copy[f'{target}_rolling_mean_4_weeks_same_day'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
            )
            df_copy[f'{target}_rolling_std_4_weeks_same_day'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=4, min_periods=1).std()
            )

    # --- SECONDARY FEATURES: Global Event & Interaction Features ---
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']

    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    # We use .bfill() (backfill) here to handle the initial NaNs created by lagging and rolling features.
    # This is a reasonable strategy to avoid losing the first few weeks of data for training.
    return df_copy.bfill()
