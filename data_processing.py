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
    
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_features_for_tft(df, events_df):
    """
    Prepares the dataframe for the Temporal Fusion Transformer.
    This is simpler as TFT handles time-dependencies internally.
    """
    df_copy = df.copy()

    # Calculate ATV, ensuring we handle division by zero
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    # Time-based features
    df_copy['month'] = df_copy['date'].dt.month.astype(str)
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek.astype(str)
    df_copy['dayofmonth'] = df_copy['date'].dt.day.astype(str)
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype(str)
    
    # Payday flags
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)

    # Event flags from future_activities
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    # Day type flags from historical data
    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    # Add a time index for the model
    df_copy['time_idx'] = (df_copy['date'] - df_copy['date'].min()).dt.days

    # Add a dummy group ID as we only have one store
    df_copy['group'] = "store_1"
    
    # Ensure dtypes are correct for the model
    for col in ['is_payday_period', 'is_event', 'is_not_normal_day']:
        df_copy[col] = df_copy[col].astype(str)

    # Make sure target variables are float
    df_copy['customers'] = df_copy['customers'].astype(float)
    df_copy['atv'] = df_copy['atv'].astype(float)

    return df_copy.sort_values(by='time_idx').reset_index(drop=True)
