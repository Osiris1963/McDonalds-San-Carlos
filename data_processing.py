# data_processing.py
import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data, ensuring time-series integrity."""
    if db_client is None: return pd.DataFrame()
    docs = db_client.collection(collection_name).stream()
    records = []
    for doc in docs:
        record = doc.to_dict()
        record['doc_id'] = doc.id
        records.append(record)
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' not in df.columns:
        if 'event_date' in df.columns: df.rename(columns={'event_date': 'date'}, inplace=True)
        else: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'sales' in df.columns and 'customers' in df.columns:
        df['atv'] = (df['sales'] / df['customers'].replace(0, np.nan)).fillna(0)
    return df.reset_index(drop=True)

def create_advanced_features(df, events_df):
    """Generates direct-horizon features and SDLY anchors."""
    df_copy = df.copy().sort_values('date')

    # --- SDLY Anchor (Matches Mon to Mon, 364 days ago) ---
    df_copy['cust_sdly'] = df_copy['customers'].shift(364)
    
    # --- Temporal Features ---
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['is_payday'] = df_copy['date'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)

    # --- Momentum Features (Recent 2026 trends) ---
    for w in [7, 14, 30]:
        df_copy[f'cust_roll_{w}'] = df_copy['customers'].shift(1).rolling(w, min_periods=1).mean()

    # --- Event Integration ---
    if events_df is not None and not events_df.empty:
        ev_dates = pd.to_datetime(events_df['date']).dt.normalize()
        df_copy['is_event'] = df_copy['date'].isin(ev_dates).astype(int)
    else:
        df_copy['is_event'] = 0

    return df_copy.fillna(method='ffill').fillna(0)

def prepare_data_for_prophet(df, events_df):
    df_p = df[['date', 'atv']].copy().rename(columns={'date': 'ds', 'atv': 'y'})
    df_p['is_payday'] = df_p['ds'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
    df_p['is_weekend'] = (df_p['ds'].dt.dayofweek >= 5).astype(int)
    return df_p, ['is_payday', 'is_weekend']
