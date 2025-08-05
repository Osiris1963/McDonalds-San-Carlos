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

    return df.sort_values(by='date').reset_index(drop=True)

def prepare_data_for_nbeats(df, events_df):
    """
    Prepares the DataFrame for N-BEATS.
    This version now handles missing dates by creating a regular time series.
    """
    if df.empty:
        return pd.DataFrame()
        
    df_copy = df.copy()

    # --- THIS IS THE FIX: Regularize the date range ---
    # Create a complete date range from the first to the last date in the data.
    full_date_range = pd.date_range(start=df_copy['date'].min(), end=df_copy['date'].max(), freq='D')
    regular_df = pd.DataFrame(full_date_range, columns=['date'])

    # Merge the original data onto the complete date range.
    # Missing dates will now have NaN values for sales, customers, etc.
    df_copy = pd.merge(regular_df, df_copy, on='date', how='left')

    # Fill missing values with 0. This assumes a missing day had 0 sales/customers.
    fill_cols = ['sales', 'customers', 'add_on_sales']
    for col in fill_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)
    # ---------------------------------------------------

    # Calculate ATV (Average Transaction Value)
    customers_safe = df_copy['customers'].replace(0, 1) # Avoid division by zero
    df_copy['atv'] = df_copy['sales'] / customers_safe

    # --- Time-Based Features (Known in the future) ---
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day'] = df_copy['date'].dt.day

    # --- Event & Payday Features (Known in the future) ---
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )

    # Merge external events data
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date']], on='date', how='left', indicator='is_event_temp')
        df_copy['is_event'] = (df_copy['is_event_temp'] == 'both').astype(int)
        df_copy.drop('is_event_temp', axis=1, inplace=True)
    else:
        df_copy['is_event'] = 0

    # The model also requires a 'time_idx' and a 'group' identifier.
    # This now creates a guaranteed continuous index.
    df_copy['time_idx'] = (df_copy['date'] - df_copy['date'].min()).dt.days
    df_copy['group'] = 'main_store' # A dummy group for our single time series

    return df_copy.fillna(0)
