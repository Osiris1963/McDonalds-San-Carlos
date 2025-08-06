import pandas as pd
import numpy as np
from datetime import timedelta

def load_from_firestore(db_client, collection_name):
    if db_client is None:
        return pd.DataFrame()
    docs = db_client.collection(collection_name).stream()
    records = [doc.to_dict() | {'doc_id': doc.id} for doc in docs]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df.dropna(subset=['date'], inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    for col in ['sales', 'customers', 'add_on_sales']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.sort_values(by='date').reset_index(drop=True)

def create_features(df, events_df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['sales_momentum'] = df['sales'] - df['sales'].shift(1)
    df['customers_momentum'] = df['customers'] - df['customers'].shift(1)
    df['sales_rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['atv'] = (df['sales'] - df.get('add_on_sales', 0)) / df['customers'].replace(0, np.nan)
    df['atv'] = df['atv'].fillna(method='ffill').fillna(0)
    df['is_payday'] = df['date'].dt.day.isin([14,15,16,29,30,31,1,2]).astype(int)
    df['payday_weekend_interaction'] = df['is_payday'] * df['is_weekend']
    if events_df is not None and not events_df.empty:
        events_df['date'] = pd.to_datetime(events_df['date']).dt.normalize()
        df = pd.merge(df, events_df[['date', 'activity_name']], on='date', how='left')
        df['is_event'] = df['activity_name'].notna().astype(int)
        df.drop(columns=['activity_name'], inplace=True)
    else:
        df['is_event'] = 0
    return df.fillna(0)
