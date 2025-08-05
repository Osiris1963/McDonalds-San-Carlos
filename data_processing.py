# data_processing.py (Re-engineered for Deep Learning Models)
import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """
    Loads and preprocesses data from a Firestore collection.
    Ensures data is sorted, de-duplicated, and correctly typed.
    """
    if db_client is None:
        return pd.DataFrame()
    
    try:
        docs = db_client.collection(collection_name).stream()
        records = [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"Error fetching from Firestore collection '{collection_name}': {e}")
        return pd.DataFrame()
        
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' not in df.columns:
        return pd.DataFrame()

    # --- Data Cleaning and Typing ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # Standardize to midnight, remove timezone info for consistency
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.reset_index(drop=True)

def create_features(df, events_df):
    """
    Engineers a rich feature set for a deep learning time series model.
    This includes time-based, event-based, and rolling statistical features.
    
    Args:
        df (pd.DataFrame): The core historical data with 'date', 'sales', 'customers'.
        events_df (pd.DataFrame): DataFrame with future events ('date', 'activity_name').

    Returns:
        pd.DataFrame: The original DataFrame augmented with many new feature columns.
    """
    df_copy = df.copy()

    # --- Target Variable Calculation ---
    # Calculate ATV (Average Transaction Value) safely
    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    # --- Time-Based Features ---
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('int')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek # Monday=0, Sunday=6

    # --- Event & Holiday Features ---
    # Payday periods
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    # User-defined events from Firestore
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    # 'Not Normal Day' flag from historical edits
    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    # --- Interaction & Cyclical Features ---
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']
    
    # Sine/Cosine transformation for cyclical features to help model understand proximity
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month']/12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month']/12)
    df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['dayofweek']/7)
    df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['dayofweek']/7)

    # --- Rolling Statistical Features (Crucial for Momentum/Volatility) ---
    targets_for_rolling = ['sales', 'customers', 'atv']
    windows = [7, 14]

    for target in targets_for_rolling:
        if target in df_copy.columns:
            for w in windows:
                # Shift by 1 to prevent data leakage (use data from T-1 to predict T)
                shifted = df_copy[target].shift(1)
                df_copy[f'{target}_rolling_mean_{w}d'] = shifted.rolling(window=w).mean()
                df_copy[f'{target}_rolling_std_{w}d'] = shifted.rolling(window=w).std()

    # Fill NaNs created by rolling features with forward fill, then backfill
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    return df_copy
