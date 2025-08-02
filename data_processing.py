# data_processing.py (Final Version with Base Sales Logic)
import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data, creating a clean 'base_sales' column for modeling."""
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
    
    # --- NEW: Base Sales Calculation ---
    # Convert to numeric, filling missing values with 0
    df['sales'] = pd.to_numeric(df.get('sales'), errors='coerce').fillna(0)
    df['add_on_sales'] = pd.to_numeric(df.get('add_on_sales'), errors='coerce').fillna(0)
    
    # Rename original 'sales' to 'total_sales' for clarity
    df.rename(columns={'sales': 'total_sales'}, inplace=True)
    
    # Create 'base_sales' which is the target for our trend model
    df['base_sales'] = df['total_sales'] - df['add_on_sales']
    # --- END NEW ---
    
    numeric_cols = ['customers']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_features(df, events_df):
    """Creates features. NOTE: The target variable for sales modeling is 'base_sales'."""
    df_copy = df.copy()

    feature_cols_to_drop = [
        'atv', 'month', 'dayofyear', 'weekofyear', 'year', 'dayofweek_num', 'dayofweek',
        'is_payday_period', 'is_event', 'is_not_normal_day', 'is_weekend', 'payday_weekend_interaction',
        'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday'
    ]
    for col in df_copy.columns:
        if 'lag' in str(col) or 'rolling' in str(col):
            feature_cols_to_drop.append(col)
    df_copy.drop(columns=[col for col in feature_cols_to_drop if col in df_copy.columns], inplace=True, errors='ignore')

    # --- MODIFIED: ATV is now calculated from base_sales ---
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (df_copy['base_sales'] / customers_safe).fillna(method='ffill').fillna(0)

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
    
    df_copy['is_weekend'] = (df_copy['date'].dt.dayofweek >= 5).astype(int)
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']

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

    # Lag features now use 'base_sales'
    for target in ['base_sales', 'customers', 'atv']:
        if target in df_copy.columns:
            df_copy[f'{target}_lag_7'] = df_copy[target].shift(7)
            df_copy[f'{target}_lag_14'] = df_copy[target].shift(14)

    for col in df_copy.columns:
        if df_copy[col].dtype == 'bool' or df_copy[col].dtype == 'uint8':
            df_copy[col] = df_copy[col].astype(int)
            
    return df_copy.fillna(0)
