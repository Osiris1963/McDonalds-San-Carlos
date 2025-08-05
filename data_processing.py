# data_processing.py (Re-architected for Temporal Fusion Transformer)
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

    # Ensure all days are present, filling gaps
    if not df.empty:
        min_date, max_date = df['date'].min(), df['date'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        df = df.set_index('date').reindex(full_date_range).reset_index().rename(columns={'index': 'date'})
        df['sales'].fillna(0, inplace=True)
        df['customers'].fillna(0, inplace=True)
        # Forward fill categorical/event data where appropriate
        df[['day_type', 'day_type_notes']] = df[['day_type', 'day_type_notes']].ffill()


    return df.sort_values(by='date').reset_index(drop=True)

def prepare_data_for_tft(df, events_df):
    """
    Creates time-based features and prepares the DataFrame for PyTorch Forecasting's TimeSeriesDataSet.
    """
    df_copy = df.copy()

    # --- Time-based Feature Engineering ---
    df_copy['month'] = df_copy['date'].dt.month.astype(str)
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype(str)
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek.astype(str)
    df_copy['day'] = df_copy['date'].dt.day

    # --- Event & Holiday Features (Known in Advance) ---
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    df_copy['is_weekend'] = (df_copy['date'].dt.dayofweek >= 5).astype(int)

    # Merge external events
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

    # --- Crucial TFT Columns ---
    # Create a continuous time index
    df_copy['time_idx'] = (df_copy['date'] - df_copy['date'].min()).dt.days

    # Create a group ID (for when we have multiple stores in the future)
    df_copy['group_id'] = 'Store_1'
    
    # Calculate ATV, handle division by zero
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)
    
    # Convert categorical features to string type for the model
    for col in ['month', 'dayofweek', 'weekofyear']:
        df_copy[col] = df_copy[col].astype('category')

    return df_copy.fillna(0)
