# data_processing.py

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

def load_from_firestore(db_client, collection_name):
    """
    Loads and preprocesses data from a Firestore collection.
    (This function remains unchanged as it correctly fetches the data)
    """
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

def prepare_data_for_tft(historical_df, events_df, periods_to_forecast):
    """
    Transforms raw data into a TimeSeriesDataSet for the Temporal Fusion Transformer.
    This is the new core of our data preparation.
    """
    df = historical_df.copy()

    # --- Feature Engineering for TFT ---
    # The model learns from raw values, but calendar events are crucial.
    df['month'] = df['date'].dt.month.astype(str)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(str)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    df['is_payday_period'] = df['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )

    # Merge external events
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df = pd.merge(df, events_df_unique[['date', 'activity_name']], on='date', how='left')
        df['is_event'] = df['activity_name'].notna().astype(int)
        df.drop(columns=['activity_name'], inplace=True)
    else:
        df['is_event'] = 0

    # Handle 'day_type' feature
    if 'day_type' in df.columns:
        df['is_not_normal_day'] = (df['day_type'] == 'Not Normal Day').astype(int)
    else:
        df['is_not_normal_day'] = 0

    # Calculate ATV (Average Transaction Value)
    base_sales = df['sales'] - df.get('add_on_sales', 0)
    customers_safe = df['customers'].replace(0, np.nan)
    df['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)
    
    # --- TimeSeriesDataSet Requirements ---
    # 1. A single group ID for our one store
    df['group_id'] = "main_store"
    
    # 2. A continuous time index
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days

    # --- Create future data placeholder for the forecast horizon ---
    last_date = df['date'].max()
    future_data = pd.DataFrame({
        'date': pd.to_datetime([last_date + pd.DateOffset(days=x) for x in range(1, periods_to_forecast + 1)]),
        'group_id': 'main_store',
    })
    future_data['time_idx'] = (future_data['date'] - df['date'].min()).dt.days
    future_data['month'] = future_data['date'].dt.month.astype(str)
    future_data['dayofweek'] = future_data['date'].dt.dayofweek.astype(str)
    future_data['year'] = future_data['date'].dt.year
    future_data['is_payday_period'] = future_data['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    if events_df is not None and not events_df.empty:
         future_data = pd.merge(future_data, events_df[['date', 'activity_name']], on='date', how='left')
         future_data['is_event'] = future_data['activity_name'].notna().astype(int)
         future_data.drop(columns=['activity_name'], inplace=True)
    else:
        future_data['is_event'] = 0
    
    # Combine historical and future data for the dataset
    df_for_loader = pd.concat([df, future_data], ignore_index=True)
    df_for_loader.fillna(0, inplace=True) # Fill NaNs for targets in future data

    # --- Define the TimeSeriesDataSet ---
    # This object handles data scaling and formatting for the model internally.
    max_encoder_length = 60  # How many past days the model sees to make a prediction
    
    dataset = TimeSeriesDataSet(
        df_for_loader,
        time_idx="time_idx",
        target=["customers", "atv"],
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=periods_to_forecast,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["month", "dayofweek"],
        time_varying_known_reals=["time_idx", "year", "is_payday_period", "is_event"],
        time_varying_unknown_reals=["customers", "atv", "sales", "add_on_sales"],
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    return dataset
