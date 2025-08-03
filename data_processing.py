# data_processing.py (Provides essential features for the new model)
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
    """Creates time-based, event-based, and advanced interaction features."""
    df_copy = df.copy().sort_values('date').reset_index(drop=True)

    # 1. Base Features (ATV, Time Components)
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('Int64')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek_num'] = df_copy['date'].dt.dayofweek
    df_copy['dayofmonth'] = df_copy['date'].dt.day

    # 2. Lag Features (Critical for the "Analog Week" model)
    for target in ['sales', 'customers', 'atv']:
        if target in df_copy.columns:
            df_copy[f'{target}_lag_7'] = df_copy[target].shift(7)
            df_copy[f'{target}_lag_14'] = df_copy[target].shift(14)

    # 3. Momentum Features (Rolling Statistics)
    df_copy['customers_rolling_mean_7'] = df_copy['customers'].shift(7).rolling(window=7, min_periods=1).mean()
    df_copy['customers_rolling_std_7'] = df_copy['customers'].shift(7).rolling(window=7, min_periods=1).std()
    
    # 4. Event & Payday Features
    df_copy['is_payday_period'] = df_copy['dayofmonth'].apply(
        lambda x: 1 if x in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    # 5. Event Horizon Features (Proximity)
    if events_df is not None and not events_df.empty:
        event_dates = pd.to_datetime(events_df['date']).dt.normalize().unique()
        event_dates = pd.Series(event_dates).sort_values()
        all_dates = pd.DataFrame({'date': pd.to_datetime(df_copy['date']).dt.normalize()})
        all_dates = all_dates.sort_values('date').drop_duplicates('date')
        event_dates_df = pd.DataFrame({'event_dt': event_dates})
        merged = pd.merge_asof(all_dates, event_dates_df, left_on='date', right_on='event_dt', direction='forward')
        all_dates['is_event'] = all_dates['date'].isin(event_dates).astype(int)
        df_copy = pd.merge(df_copy, all_dates[['date', 'is_event']], on='date', how='left')
    else:
        df_copy['is_event'] = 0

    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    # Final cleanup
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(0, inplace=True)
            
    return df_copy
