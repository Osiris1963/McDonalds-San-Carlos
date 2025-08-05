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

    # --- Ensure data is regular ---
    if not df.empty:
        full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        regular_df = pd.DataFrame(full_date_range, columns=['date'])
        df = pd.merge(regular_df, df, on='date', how='left').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_features(df):
    """Creates time-series features from a dataframe."""
    df = df.copy()
    df['atv'] = (df['sales'] / df['customers'].replace(0, 1)).fillna(0)
    
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    df['is_payday_period'] = df['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    
    # Lag features - crucial for tree models
    for lag in [15, 16, 30]: # Lags must be >= forecast horizon
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        df[f'customers_lag_{lag}'] = df['customers'].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        df[f'sales_roll_mean_{window}'] = df['sales'].shift(15).rolling(window=window).mean()
        df[f'customers_roll_mean_{window}'] = df['customers'].shift(15).rolling(window=window).mean()
        
    return df.fillna(0)
