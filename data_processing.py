# data_processing.py
import pandas as pd
from datetime import timedelta

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data from a Firestore collection."""
    if db_client is None:
        return pd.DataFrame()
    
    docs = db_client.collection(collection_name).stream()
    records = [doc.to_dict() for doc in docs]
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # --- Robust Date Handling ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    # --- Numeric Type Conversion ---
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Sort and return ---
    return df.sort_values(by='date').reset_index(drop=True)


def create_features(df, events_df):
    """
    This is the new, robust feature engineering pipeline.
    It creates time-based, event-based, and lag/rolling features.
    """
    df_copy = df.copy()
    
    # --- 1. Foundational Time Features ---
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype(int)
    df_copy['year'] = df_copy['date'].dt.year

    # --- 2. CRITICAL: Day of Week (One-Hot Encoded) ---
    # This is the "apples-to-apples" fix you wanted.
    df_copy['dayofweek'] = df_copy['date'].dt.day_name()
    day_dummies = pd.get_dummies(df_copy['dayofweek'], prefix='day')
    df_copy = pd.concat([df_copy, day_dummies], axis=1)

    # --- 3. CRITICAL: Payday Feature (Context-Aware) ---
    # Recognizes the 15/30 kinsenas/katapusan cycle in the Philippines.
    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    # --- 4. Holiday & Events Features ---
    # Merge future activities and known PH holidays for a complete event calendar
    if events_df is not None and not events_df.empty:
        events_df['date'] = pd.to_datetime(events_df['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    # Let's also add 'day_type' from your data entry as a feature
    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)

    # --- 5. Lag & Rolling Window Features (Momentum) ---
    # Shift by the forecast horizon (15) + 1 to avoid data leakage for future predictions
    future_shift = 1 
    
    # Lag features
    df_copy['sales_lag_7'] = df_copy['sales'].shift(7 + future_shift)
    df_copy['sales_lag_15'] = df_copy['sales'].shift(15 + future_shift)
    df_copy['customers_lag_7'] = df_copy['customers'].shift(7 + future_shift)

    # Rolling features
    df_copy['sales_rolling_mean_7'] = df_copy['sales'].shift(future_shift).rolling(window=7, min_periods=1).mean()
    df_copy['sales_rolling_std_7'] = df_copy['sales'].shift(future_shift).rolling(window=7, min_periods=1).std()
    
    # --- 6. Calculate ATV ---
    df_copy['customers'] = df_copy['customers'].replace(0, pd.NA) # Avoid division by zero
    df_copy['atv'] = (df_copy['sales'] / df_copy['customers']).fillna(0)

    return df_copy
