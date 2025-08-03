# data_processing.py (Unchanged)
import pandas as pd
import numpy as np
from datetime import timedelta

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
    Creates a robust set of features for the forecasting models.
    """
    df_copy = df.copy().sort_values('date').reset_index(drop=True)
    df_copy['date'] = pd.to_datetime(df_copy['date'])

    # --- 1. Base and Time-based Features ---
    if 'sales' in df_copy.columns and 'customers' in df_copy.columns:
        base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
        customers_safe = df_copy['customers'].replace(0, np.nan)
        df_copy['atv'] = (base_sales / customers_safe)
    
    df_copy['dayofmonth'] = df_copy['date'].dt.day
    df_copy['is_payday_period'] = df_copy['dayofmonth'].apply(
        lambda x: 1 if x in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )

    # --- 2. Lag and Rolling Features ---
    if 'customers' in df_copy.columns:
        df_copy['customers_lag_7'] = df_copy['customers'].shift(7)
        df_copy['customers_rolling_mean_7'] = df_copy['customers'].shift(7).rolling(window=7, min_periods=1).mean()

    # --- 3. Event-based Features ---
    if events_df is not None and not events_df.empty:
        event_dates = set(pd.to_datetime(events_df['date']).dt.normalize())
        df_copy['is_event'] = df_copy['date'].dt.normalize().isin(event_dates).astype(int)
    else:
        df_copy['is_event'] = 0
        
    # --- 4. Final Cleanup ---
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(0, inplace=True)

    return df_copy
