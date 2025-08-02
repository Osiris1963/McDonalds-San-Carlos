# data_processing.py (Enhanced with Trend and Interaction Features)
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
    df_copy = df.copy()

    # --- ENHANCEMENT: Added new feature names to the drop list for idempotency ---
    feature_cols_to_drop = [
        'atv', 'month', 'dayofyear', 'weekofyear', 'year', 'dayofweek_num', 'dayofweek',
        'is_payday_period', 'is_event', 'is_not_normal_day', 'is_weekend', 'payday_weekend_interaction',
        'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday'
    ]
    # Add rolling features and new interactions to the drop list
    for col in df_copy.columns:
        if 'lag' in str(col) or 'rolling' in str(col) or 'payday_interaction' in str(col):
            feature_cols_to_drop.append(col)
    df_copy.drop(columns=[col for col in feature_cols_to_drop if col in df_copy.columns], inplace=True, errors='ignore')


    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('Int64')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek_num'] = df_copy['date'].dt.dayofweek

    df_copy['dayofweek'] = df_copy['date'].dt.day_name()
    day_dummies = pd.get_dummies(df_copy['dayofweek'], prefix='day', drop_first=False)
    df_copy = pd.concat([df_copy, day_dummies], axis=1)

    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    df_copy['is_weekend'] = (df_copy['date'].dt.dayofweek >= 5).astype(int) # Saturday=5, Sunday=6
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']
    
    # --- ENHANCEMENT: Deeper Interaction Features ---
    # This helps the model learn the specific amplified effect of a payday falling on a particular day of the week.
    days_of_week_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_str in days_of_week_str:
        col_name = f'day_{day_str}'
        if col_name in df_copy.columns:
            df_copy[f'payday_interaction_{day_str}'] = df_copy['is_payday_period'] * df_copy[col_name]
    # --- END ENHANCEMENT ---


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

    shift_val = 1 
    for target in ['sales', 'customers', 'atv']:
        if target in df_copy.columns:
            df_copy[f'{target}_lag_7'] = df_copy[target].shift(shift_val + 6)
            df_copy[f'{target}_lag_14'] = df_copy[target].shift(shift_val + 13)

    # --- ENHANCEMENT: Smarter Time-Based Features (Rolling Statistics) ---
    # "Trend is King". These features give the model a sense of momentum.
    # We use .shift(1) to ensure we're only using past data to create features for the current day (prevents data leakage).
    for target in ['sales', 'customers', 'atv']:
        if target in df_copy.columns:
            # Rolling means to capture recent trends over 1 and 2 weeks
            df_copy[f'{target}_rolling_mean_7'] = df_copy[target].shift(1).rolling(window=7, min_periods=1).mean()
            df_copy[f'{target}_rolling_mean_14'] = df_copy[target].shift(1).rolling(window=14, min_periods=1).mean()
            # Rolling standard deviation to capture recent volatility
            df_copy[f'{target}_rolling_std_7'] = df_copy[target].shift(1).rolling(window=7, min_periods=1).std()
    # --- END ENHANCEMENT ---

    for col in df_copy.columns:
        if df_copy[col].dtype == 'bool' or df_copy[col].dtype == 'uint8':
            df_copy[col] = df_copy[col].astype(int)
            
    return df_copy.fillna(0)
