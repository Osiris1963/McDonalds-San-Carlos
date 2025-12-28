import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Securely loads and normalizes Firestore data for time-series analysis."""
    if db_client is None: return pd.DataFrame()
    docs = db_client.collection(collection_name).stream()
    records = [{'doc_id': doc.id, **doc.to_dict()} for doc in docs]
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    # Flexible date column detection
    date_col = 'event_date' if 'event_date' in df.columns and collection_name == 'future_activities' else 'date'
    
    # Standardize time: Strip timezones and normalize to midnight
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None).dt.normalize()
    df.dropna(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    # Type safety for numeric calculations
    for col in ['sales', 'customers', 'add_on_sales']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df.reset_index(drop=True)

def create_advanced_features(df, events_df):
    """Engineers complex features including Velocity and Day-of-Week Lags."""
    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 1. Day-of-Week Specific Averages: Captures 'Friday' spikes accurately
    if 'customers' in df.columns:
        df['dow_cust_avg'] = df.groupby('dayofweek')['customers'].transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
        )
        # 2. Sales Velocity: Detects if demand is accelerating toward a peak
        short_roll = df['customers'].shift(1).rolling(window=7, min_periods=1).mean()
        long_roll = df['customers'].shift(1).rolling(window=28, min_periods=1).mean()
        df['velocity'] = (short_roll / (long_roll + 1e-5))

    # 3. Cyclical Encoding: Helps AI understand time as a circle, not a linear number
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # 4. Event Proximity: High-accuracy flag for upcoming promotions
    if events_df is not None and not events_df.empty:
        event_dates = pd.to_datetime(events_df['date']).dt.normalize().unique()
        df['is_event'] = df['date'].isin(event_dates).astype(int)
    else:
        df['is_event'] = 0

    return df.fillna(0)
