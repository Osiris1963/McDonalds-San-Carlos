# data_processing.py (Final Version with Feature Isolation)
import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data, creating a clean 'base_sales' column for modeling."""
    if db_client is None: return pd.DataFrame()
    
    docs = db_client.collection(collection_name).stream()
    records = [doc.to_dict() for doc in docs]
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    if 'date' not in df.columns: return df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    df['date'] = df['date'].dt.tz_localize(None).dt.normalize()

    if 'sales' in df.columns:
        df['add_on_sales'] = pd.to_numeric(df.get('add_on_sales'), errors='coerce').fillna(0)
        df['sales'] = pd.to_numeric(df.get('sales'), errors='coerce').fillna(0)
        df.rename(columns={'sales': 'total_sales'}, inplace=True)
        df['base_sales'] = df['total_sales'] - df['add_on_sales']

    if 'customers' in df.columns:
        df['customers'] = pd.to_numeric(df['customers'], errors='coerce').fillna(0)

    return df.sort_values(by='date').reset_index(drop=True)

def create_features_for_customers(df, events_df):
    """Creates a pure, non-financial feature set for customer forecasting."""
    df_copy = df.copy()
    
    # Date-based features
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('Int64')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek_num'] = df_copy['date'].dt.dayofweek
    df_copy['is_weekend'] = (df_copy['dayofweek_num'] >= 5).astype(int)
    
    # Contextual features
    df_copy['is_payday_period'] = df_copy['date'].apply(lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0).astype(int)
    
    if 'add_on_sales' in df_copy.columns:
        df_copy['is_addon_day'] = (df_copy['add_on_sales'] > 1000).astype(int)
    else:
        df_copy['is_addon_day'] = 0

    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date']], on='date', how='left', indicator=True)
        df_copy['is_event'] = (df_copy['_merge'] == 'both').astype(int)
        df_copy.drop(columns=['_merge'], inplace=True)
    else:
        df_copy['is_event'] = 0

    # Lag features of itself ONLY
    if 'customers' in df_copy.columns:
        df_copy['customers_lag_7'] = df_copy['customers'].shift(7)
        df_copy['customers_lag_14'] = df_copy['customers'].shift(14)
        df_copy['customers_lag_21'] = df_copy['customers'].shift(21)
            
    return df_copy.fillna(0)

def create_features_for_atv(df, events_df):
    """Creates a feature set for ATV, which can include customer-related features."""
    # Start with the pure customer features
    df_copy = create_features_for_customers(df, events_df)
    
    # Calculate ATV - this is the only place it should be calculated
    if 'base_sales' in df_copy.columns and 'customers' in df_copy.columns:
        customers_safe = df_copy['customers'].replace(0, np.nan)
        df_copy['atv'] = (df_copy['base_sales'] / customers_safe).fillna(method='ffill').fillna(0)

    # Add ATV's own lags
    if 'atv' in df_copy.columns:
        df_copy['atv_lag_7'] = df_copy['atv'].shift(7)
        df_copy['atv_lag_14'] = df_copy['atv'].shift(14)
        df_copy['atv_lag_21'] = df_copy['atv'].shift(21)

    return df_copy.fillna(0)
