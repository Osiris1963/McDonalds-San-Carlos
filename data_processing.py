# data_processing.py
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data from a Firestore collection. (No changes needed here)"""
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

def prepare_data_for_tft(df, events_df):
    """
    Takes raw dataframes and engineers features suitable for the TFT model.
    This replaces the old 'create_advanced_features' function.
    """
    df_copy = df.copy()

    # --- Basic Time Features ---
    df_copy['month'] = df_copy['date'].dt.month.astype(str)
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek.astype(str)
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype(int)

    # --- Event & Holiday Features (Known in Advance) ---
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    df_copy['is_weekend'] = (df_copy['date'].dt.dayofweek >= 5).astype(int)

    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date']], on='date', how='left', indicator='is_event_merge')
        df_copy['is_event'] = (df_copy['is_event_merge'] == 'both').astype(int)
        df_copy.drop('is_event_merge', axis=1, inplace=True)
    else:
        df_copy['is_event'] = 0

    # --- Mandatory TFT Columns ---
    # The time_idx is a continuous integer representing time steps
    df_copy['time_idx'] = (df_copy['date'] - df_copy['date'].min()).dt.days
    # The group_id identifies the time series. Here, we only have one.
    df_copy['group_id'] = "Store_1"

    # --- Target Variable ---
    # We will forecast customers. Sales will be derived from customers * predicted ATV.
    df_copy['customers'] = df_copy['customers'].astype(float)

    # Impute ATV where customers is 0 to avoid division by zero errors
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (df_copy['sales'] / customers_safe).fillna(method='ffill').fillna(0)

    return df_copy

def create_tft_dataset(data, max_encoder_length, max_prediction_length):
    """
    Creates the TimeSeriesDataSet object for PyTorch Forecasting.
    """
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="customers",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,  # How much history to use for prediction
        max_prediction_length=max_prediction_length, # How far to predict
        static_categoricals=["group_id"],
        # Features that change over time and are known in the future (e.g., holidays, day of week)
        time_varying_known_categoricals=["month", "dayofweek"],
        time_varying_known_reals=["is_payday_period", "is_weekend", "is_event", "dayofyear", "weekofyear"],
        # Features that change over time but are not known in the future (our targets and related values)
        time_varying_unknown_reals=["customers", "atv", "sales"],
        allow_missing_timesteps=True,
        target_normalizer=None # We will handle normalization manually if needed
    )
