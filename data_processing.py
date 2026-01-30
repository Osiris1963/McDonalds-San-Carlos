# data_processing.py - ENHANCED VERSION v10.0
# Key enhancements:
# 1. Contextual Window Analysis (SDLY ± 14 days)
# 2. 8-Week Recent Trend Analysis with weighted significance
# 3. Purchase power and sales contribution metrics
# 4. Support for self-correction learning

import pandas as pd
import numpy as np
from datetime import timedelta

def load_from_firestore(db_client, collection_name):
    """Loads and preprocesses data with integrity checks."""
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
    
    # Handle date column
    if 'date' not in df.columns:
        if 'event_date' in df.columns:
            df.rename(columns={'event_date': 'date'}, inplace=True)
        else:
            return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
    
    # Convert numeric columns
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate ATV (Average Transaction Value / Purchase Power)
    if 'sales' in df.columns and 'customers' in df.columns:
        df['atv'] = (df['sales'] / df['customers'].replace(0, np.nan)).fillna(0)
    
    return df.reset_index(drop=True)


def load_forecast_errors(db_client):
    """
    Load historical forecast errors for self-correction learning.
    Compares forecast_log entries against historical_data actuals.
    """
    if db_client is None:
        return pd.DataFrame()
    
    try:
        # Load forecast log
        forecast_docs = db_client.collection('forecast_log').stream()
        forecasts = []
        for doc in forecast_docs:
            record = doc.to_dict()
            record['date'] = doc.id
            forecasts.append(record)
        
        if not forecasts:
            return pd.DataFrame()
        
        forecast_df = pd.DataFrame(forecasts)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Load actuals
        hist_df = load_from_firestore(db_client, 'historical_data')
        
        if hist_df.empty:
            return pd.DataFrame()
        
        # Merge to find errors
        merged = pd.merge(
            forecast_df,
            hist_df[['date', 'customers', 'sales', 'atv']],
            on='date',
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        if merged.empty:
            return pd.DataFrame()
        
        # Calculate errors
        merged['customer_error'] = merged['customers'] - merged['predicted_customers']
        merged['customer_pct_error'] = (merged['customer_error'] / merged['customers'].replace(0, np.nan) * 100).fillna(0)
        
        merged['sales_error'] = merged['sales'] - merged['predicted_sales']
        merged['sales_pct_error'] = (merged['sales_error'] / merged['sales'].replace(0, np.nan) * 100).fillna(0)
        
        merged['atv_error'] = merged['atv'] - merged['predicted_atv']
        
        # Add temporal features for pattern detection
        merged['dayofweek'] = merged['date'].dt.dayofweek
        merged['month'] = merged['date'].dt.month
        merged['is_payday'] = merged['date'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
        
        return merged
        
    except Exception as e:
        print(f"Error loading forecast errors: {e}")
        return pd.DataFrame()


def calculate_contextual_window_features(df, target_date, days_before=14, days_after=14):
    """
    Calculate features from the same period last year (± days window).
    
    For forecasting January 29, 2026:
    - Looks at January 15-28, 2025 (14 days before)
    - Looks at January 29, 2025 (exact SDLY)
    - Looks at January 30 - February 12, 2025 (14 days after)
    
    This captures the TREND around that period last year.
    """
    # Calculate SDLY date (364 days = 52 weeks, preserves day of week)
    sdly_date = target_date - timedelta(days=364)
    
    # Define the contextual window
    window_start = sdly_date - timedelta(days=days_before)
    window_end = sdly_date + timedelta(days=days_after)
    
    # Get data in the window
    window_data = df[(df['date'] >= window_start) & (df['date'] <= window_end)].copy()
    
    if window_data.empty:
        return {
            'sdly_customers': np.nan,
            'sdly_sales': np.nan,
            'sdly_atv': np.nan,
            'sdly_window_cust_mean': np.nan,
            'sdly_window_cust_std': np.nan,
            'sdly_window_sales_mean': np.nan,
            'sdly_window_atv_mean': np.nan,
            'sdly_trend_before': np.nan,
            'sdly_trend_after': np.nan,
            'sdly_momentum': np.nan,
            'sdly_cust_growth': np.nan,
            'sdly_sales_growth': np.nan
        }
    
    # Exact SDLY values
    sdly_exact = df[df['date'] == sdly_date]
    sdly_customers = sdly_exact['customers'].values[0] if len(sdly_exact) > 0 else np.nan
    sdly_sales = sdly_exact['sales'].values[0] if len(sdly_exact) > 0 else np.nan
    sdly_atv = sdly_exact['atv'].values[0] if len(sdly_exact) > 0 else np.nan
    
    # Window statistics
    sdly_window_cust_mean = window_data['customers'].mean()
    sdly_window_cust_std = window_data['customers'].std()
    sdly_window_sales_mean = window_data['sales'].mean()
    sdly_window_atv_mean = window_data['atv'].mean()
    
    # Trend analysis: before vs after the SDLY date
    before_data = window_data[window_data['date'] < sdly_date]
    after_data = window_data[window_data['date'] > sdly_date]
    
    # Trend before (was traffic increasing or decreasing leading up to SDLY?)
    if len(before_data) >= 3:
        before_data = before_data.sort_values('date')
        first_half = before_data.head(len(before_data)//2)['customers'].mean()
        second_half = before_data.tail(len(before_data)//2)['customers'].mean()
        sdly_trend_before = (second_half - first_half) / first_half if first_half > 0 else 0
    else:
        sdly_trend_before = 0
    
    # Trend after (did traffic increase or decrease after SDLY?)
    if len(after_data) >= 3:
        after_data = after_data.sort_values('date')
        first_half = after_data.head(len(after_data)//2)['customers'].mean()
        second_half = after_data.tail(len(after_data)//2)['customers'].mean()
        sdly_trend_after = (second_half - first_half) / first_half if first_half > 0 else 0
    else:
        sdly_trend_after = 0
    
    # Momentum: compare after period to before period
    before_mean = before_data['customers'].mean() if len(before_data) > 0 else np.nan
    after_mean = after_data['customers'].mean() if len(after_data) > 0 else np.nan
    sdly_momentum = after_mean / before_mean if before_mean and before_mean > 0 else 1.0
    
    # Growth rates (purchase power and sales contribution)
    if len(before_data) > 0 and len(after_data) > 0:
        sdly_cust_growth = (after_data['customers'].mean() - before_data['customers'].mean()) / before_data['customers'].mean() if before_data['customers'].mean() > 0 else 0
        sdly_sales_growth = (after_data['sales'].mean() - before_data['sales'].mean()) / before_data['sales'].mean() if before_data['sales'].mean() > 0 else 0
    else:
        sdly_cust_growth = 0
        sdly_sales_growth = 0
    
    return {
        'sdly_customers': sdly_customers,
        'sdly_sales': sdly_sales,
        'sdly_atv': sdly_atv,
        'sdly_window_cust_mean': sdly_window_cust_mean,
        'sdly_window_cust_std': sdly_window_cust_std,
        'sdly_window_sales_mean': sdly_window_sales_mean,
        'sdly_window_atv_mean': sdly_window_atv_mean,
        'sdly_trend_before': sdly_trend_before,
        'sdly_trend_after': sdly_trend_after,
        'sdly_momentum': sdly_momentum,
        'sdly_cust_growth': sdly_cust_growth,
        'sdly_sales_growth': sdly_sales_growth
    }


def calculate_recent_trend_features(df, target_date, weeks=8):
    """
    Analyze the last 8 weeks of data with WEIGHTED significance.
    Recent weeks get MORE weight than older weeks.
    
    Weights: Week 1 (most recent) = 8x, Week 2 = 7x, ... Week 8 = 1x
    """
    end_date = target_date - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(weeks=weeks)
    
    recent_data = df[(df['date'] > start_date) & (df['date'] <= end_date)].copy()
    
    if recent_data.empty or len(recent_data) < 7:
        return {
            'recent_weighted_cust_mean': np.nan,
            'recent_weighted_sales_mean': np.nan,
            'recent_weighted_atv_mean': np.nan,
            'recent_trend_slope': np.nan,
            'recent_acceleration': np.nan,
            'recent_volatility': np.nan,
            'recent_momentum_4w': np.nan,
            'recent_momentum_2w': np.nan,
            'recent_dow_factor': np.nan,
            'recent_purchase_power_trend': np.nan,
            'week_1_avg_cust': np.nan,
            'week_2_avg_cust': np.nan,
            'week_4_avg_cust': np.nan,
            'week_8_avg_cust': np.nan
        }
    
    recent_data = recent_data.sort_values('date')
    
    # Calculate weeks ago for each record
    recent_data['days_ago'] = (end_date - recent_data['date']).dt.days
    recent_data['weeks_ago'] = (recent_data['days_ago'] // 7) + 1
    recent_data['weeks_ago'] = recent_data['weeks_ago'].clip(1, weeks)
    
    # Weight: more recent = higher weight
    # Week 1 gets weight 8, Week 8 gets weight 1
    recent_data['weight'] = weeks + 1 - recent_data['weeks_ago']
    
    # Weighted means
    total_weight = recent_data['weight'].sum()
    recent_weighted_cust_mean = (recent_data['customers'] * recent_data['weight']).sum() / total_weight
    recent_weighted_sales_mean = (recent_data['sales'] * recent_data['weight']).sum() / total_weight
    recent_weighted_atv_mean = (recent_data['atv'] * recent_data['weight']).sum() / total_weight
    
    # Weekly aggregations
    weekly_agg = recent_data.groupby('weeks_ago').agg({
        'customers': 'mean',
        'sales': 'mean',
        'atv': 'mean'
    }).reset_index()
    
    # Get specific week averages
    week_1_avg = weekly_agg[weekly_agg['weeks_ago'] == 1]['customers'].values
    week_2_avg = weekly_agg[weekly_agg['weeks_ago'] == 2]['customers'].values
    week_4_avg = weekly_agg[weekly_agg['weeks_ago'] == 4]['customers'].values
    week_8_avg = weekly_agg[weekly_agg['weeks_ago'] == 8]['customers'].values
    
    week_1_avg_cust = week_1_avg[0] if len(week_1_avg) > 0 else np.nan
    week_2_avg_cust = week_2_avg[0] if len(week_2_avg) > 0 else np.nan
    week_4_avg_cust = week_4_avg[0] if len(week_4_avg) > 0 else np.nan
    week_8_avg_cust = week_8_avg[0] if len(week_8_avg) > 0 else np.nan
    
    # Trend slope (linear regression over time)
    recent_data['day_index'] = range(len(recent_data))
    if len(recent_data) >= 7:
        # Simple linear regression
        x = recent_data['day_index'].values
        y = recent_data['customers'].values
        slope = np.polyfit(x, y, 1)[0]
        recent_trend_slope = slope
    else:
        recent_trend_slope = 0
    
    # Acceleration (is the trend speeding up or slowing down?)
    if len(weekly_agg) >= 4:
        first_half_trend = weekly_agg[weekly_agg['weeks_ago'] >= 5]['customers'].mean() if len(weekly_agg[weekly_agg['weeks_ago'] >= 5]) > 0 else 0
        second_half_trend = weekly_agg[weekly_agg['weeks_ago'] <= 4]['customers'].mean() if len(weekly_agg[weekly_agg['weeks_ago'] <= 4]) > 0 else 0
        recent_acceleration = (second_half_trend - first_half_trend) / first_half_trend if first_half_trend > 0 else 0
    else:
        recent_acceleration = 0
    
    # Volatility (coefficient of variation)
    recent_volatility = recent_data['customers'].std() / recent_data['customers'].mean() if recent_data['customers'].mean() > 0 else 0
    
    # Momentum: last 4 weeks vs previous 4 weeks
    last_4w = recent_data[recent_data['weeks_ago'] <= 4]['customers'].mean()
    prev_4w = recent_data[recent_data['weeks_ago'] > 4]['customers'].mean()
    recent_momentum_4w = last_4w / prev_4w if prev_4w and prev_4w > 0 else 1.0
    
    # Momentum: last 2 weeks vs weeks 3-4
    last_2w = recent_data[recent_data['weeks_ago'] <= 2]['customers'].mean()
    prev_2w = recent_data[(recent_data['weeks_ago'] > 2) & (recent_data['weeks_ago'] <= 4)]['customers'].mean()
    recent_momentum_2w = last_2w / prev_2w if prev_2w and prev_2w > 0 else 1.0
    
    # Day-of-week factor for target date
    target_dow = target_date.weekday()
    dow_data = recent_data[recent_data['date'].dt.dayofweek == target_dow]
    overall_mean = recent_data['customers'].mean()
    dow_mean = dow_data['customers'].mean() if len(dow_data) > 0 else overall_mean
    recent_dow_factor = dow_mean / overall_mean if overall_mean > 0 else 1.0
    
    # Purchase power trend (ATV trend)
    if len(weekly_agg) >= 2:
        recent_weeks_atv = recent_data[recent_data['weeks_ago'] <= 2]['atv'].mean()
        older_weeks_atv = recent_data[recent_data['weeks_ago'] > 2]['atv'].mean()
        recent_purchase_power_trend = recent_weeks_atv / older_weeks_atv if older_weeks_atv > 0 else 1.0
    else:
        recent_purchase_power_trend = 1.0
    
    return {
        'recent_weighted_cust_mean': recent_weighted_cust_mean,
        'recent_weighted_sales_mean': recent_weighted_sales_mean,
        'recent_weighted_atv_mean': recent_weighted_atv_mean,
        'recent_trend_slope': recent_trend_slope,
        'recent_acceleration': recent_acceleration,
        'recent_volatility': recent_volatility,
        'recent_momentum_4w': recent_momentum_4w,
        'recent_momentum_2w': recent_momentum_2w,
        'recent_dow_factor': recent_dow_factor,
        'recent_purchase_power_trend': recent_purchase_power_trend,
        'week_1_avg_cust': week_1_avg_cust,
        'week_2_avg_cust': week_2_avg_cust,
        'week_4_avg_cust': week_4_avg_cust,
        'week_8_avg_cust': week_8_avg_cust
    }


def calculate_yoy_comparison(df, target_date):
    """
    Calculate Year-over-Year comparison metrics.
    Compares current year performance to last year.
    """
    # This year's data (last 30 days)
    ty_end = target_date - timedelta(days=1)
    ty_start = ty_end - timedelta(days=30)
    this_year = df[(df['date'] >= ty_start) & (df['date'] <= ty_end)]
    
    # Last year's same period
    ly_end = ty_end - timedelta(days=364)
    ly_start = ty_start - timedelta(days=364)
    last_year = df[(df['date'] >= ly_start) & (df['date'] <= ly_end)]
    
    if this_year.empty or last_year.empty:
        return {
            'yoy_customer_growth': 1.0,
            'yoy_sales_growth': 1.0,
            'yoy_atv_growth': 1.0
        }
    
    ty_cust = this_year['customers'].mean()
    ly_cust = last_year['customers'].mean()
    yoy_customer_growth = ty_cust / ly_cust if ly_cust > 0 else 1.0
    
    ty_sales = this_year['sales'].mean()
    ly_sales = last_year['sales'].mean()
    yoy_sales_growth = ty_sales / ly_sales if ly_sales > 0 else 1.0
    
    ty_atv = this_year['atv'].mean()
    ly_atv = last_year['atv'].mean()
    yoy_atv_growth = ty_atv / ly_atv if ly_atv > 0 else 1.0
    
    return {
        'yoy_customer_growth': yoy_customer_growth,
        'yoy_sales_growth': yoy_sales_growth,
        'yoy_atv_growth': yoy_atv_growth
    }


def detect_special_dates(date_series):
    """Identify special dates that historically have unusual traffic patterns."""
    df = pd.DataFrame({'date': date_series})
    
    # Philippine holidays and special dates
    df['is_new_year'] = ((df['date'].dt.month == 1) & (df['date'].dt.day <= 2)).astype(int)
    df['is_christmas_season'] = ((df['date'].dt.month == 12) & (df['date'].dt.day >= 15)).astype(int)
    df['is_valentines'] = ((df['date'].dt.month == 2) & (df['date'].dt.day.isin([13, 14, 15]))).astype(int)
    df['is_holy_week'] = 0  # Will be filled by events_df
    
    # Payday patterns
    df['is_payday'] = df['date'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
    df['is_mid_month_pay'] = df['date'].dt.day.isin([14, 15, 16]).astype(int)
    df['is_end_month_pay'] = df['date'].dt.day.isin([29, 30, 31, 1, 2]).astype(int)
    
    # Weekend
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['is_friday'] = (df['date'].dt.dayofweek == 4).astype(int)
    df['is_monday'] = (df['date'].dt.dayofweek == 0).astype(int)
    
    return df.drop(columns=['date'])


def create_advanced_features(df, events_df, for_prediction=False):
    """
    Master feature engineering function with contextual window analysis.
    """
    result = df.copy().sort_values('date').reset_index(drop=True)
    
    # === TEMPORAL FEATURES ===
    result['year'] = result['date'].dt.year
    result['month'] = result['date'].dt.month
    result['day'] = result['date'].dt.day
    result['dayofweek'] = result['date'].dt.dayofweek
    result['weekofyear'] = result['date'].dt.isocalendar().week.astype(int)
    result['dayofyear'] = result['date'].dt.dayofyear
    result['quarter'] = result['date'].dt.quarter
    
    # Cyclical encoding
    result['dow_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    # === SPECIAL DATE FLAGS ===
    special_dates = detect_special_dates(result['date'])
    result = pd.concat([result, special_dates], axis=1)
    
    # === EVENT INTEGRATION ===
    if events_df is not None and not events_df.empty:
        ev_dates = pd.to_datetime(events_df['date']).dt.normalize()
        result['is_event'] = result['date'].isin(ev_dates).astype(int)
        
        # Days until/from event
        result['days_to_event'] = 99
        result['days_from_event'] = 99
        
        for idx, row in result.iterrows():
            future_events = ev_dates[ev_dates > row['date']]
            past_events = ev_dates[ev_dates <= row['date']]
            
            if len(future_events) > 0:
                result.loc[idx, 'days_to_event'] = (future_events.min() - row['date']).days
            if len(past_events) > 0:
                result.loc[idx, 'days_from_event'] = (row['date'] - past_events.max()).days
    else:
        result['is_event'] = 0
        result['days_to_event'] = 99
        result['days_from_event'] = 99
    
    # === BASIC LAG FEATURES ===
    if 'customers' in result.columns and result['customers'].notna().any():
        for lag in [1, 7, 14, 21, 28]:
            result[f'customers_lag_{lag}'] = result['customers'].shift(lag)
        
        # Rolling statistics (with min_lag=1 to prevent leakage)
        shifted = result['customers'].shift(1)
        for window in [7, 14, 30]:
            result[f'customers_roll_mean_{window}'] = shifted.rolling(window, min_periods=1).mean()
            result[f'customers_roll_std_{window}'] = shifted.rolling(window, min_periods=3).std()
    
    # === CONTEXTUAL WINDOW FEATURES (SDLY ± 14 days) ===
    # These are calculated per-row for each target date
    contextual_features = []
    for idx, row in result.iterrows():
        ctx = calculate_contextual_window_features(result, row['date'])
        contextual_features.append(ctx)
    
    ctx_df = pd.DataFrame(contextual_features)
    result = pd.concat([result.reset_index(drop=True), ctx_df], axis=1)
    
    # === 8-WEEK RECENT TREND FEATURES ===
    recent_features = []
    for idx, row in result.iterrows():
        recent = calculate_recent_trend_features(result, row['date'], weeks=8)
        recent_features.append(recent)
    
    recent_df = pd.DataFrame(recent_features)
    result = pd.concat([result.reset_index(drop=True), recent_df], axis=1)
    
    # === YOY COMPARISON ===
    yoy_features = []
    for idx, row in result.iterrows():
        yoy = calculate_yoy_comparison(result, row['date'])
        yoy_features.append(yoy)
    
    yoy_df = pd.DataFrame(yoy_features)
    result = pd.concat([result.reset_index(drop=True), yoy_df], axis=1)
    
    # === FILL MISSING VALUES ===
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].ffill().bfill().fillna(0)
    
    return result


def prepare_data_for_prophet(df, events_df):
    """Prepare data for Prophet model (ATV forecasting)."""
    df_p = df[['date', 'atv']].copy().rename(columns={'date': 'ds', 'atv': 'y'})
    
    df_p['is_payday'] = df_p['ds'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
    df_p['is_weekend'] = (df_p['ds'].dt.dayofweek >= 5).astype(int)
    df_p['is_friday'] = (df_p['ds'].dt.dayofweek == 4).astype(int)
    df_p['is_new_year'] = ((df_p['ds'].dt.month == 1) & (df_p['ds'].dt.day <= 2)).astype(int)
    df_p['is_christmas'] = ((df_p['ds'].dt.month == 12) & (df_p['ds'].dt.day >= 20)).astype(int)
    
    if events_df is not None and not events_df.empty:
        ev_dates = pd.to_datetime(events_df['date']).dt.normalize()
        df_p['is_event'] = df_p['ds'].isin(ev_dates).astype(int)
    else:
        df_p['is_event'] = 0
    
    regressors = ['is_payday', 'is_weekend', 'is_friday', 'is_new_year', 'is_christmas', 'is_event']
    
    return df_p, regressors


def calculate_historical_multipliers(df, events_df=None):
    """Calculate historical multipliers for special dates."""
    if df.empty or 'customers' not in df.columns:
        return {}
    
    df = df.copy()
    
    # Calculate baseline
    dow_baseline = df.groupby(df['date'].dt.dayofweek)['customers'].median()
    df['expected'] = df['date'].dt.dayofweek.map(dow_baseline)
    df['multiplier'] = df['customers'] / df['expected']
    
    multipliers = {}
    
    # New Year
    new_year = df[(df['date'].dt.month == 1) & (df['date'].dt.day <= 2)]
    if len(new_year) > 0:
        multipliers['new_year'] = new_year['multiplier'].median()
    
    # Christmas
    christmas = df[(df['date'].dt.month == 12) & (df['date'].dt.day >= 20)]
    if len(christmas) > 0:
        multipliers['christmas'] = christmas['multiplier'].median()
    
    # Payday
    payday = df[df['date'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2])]
    non_payday = df[~df['date'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2])]
    if len(payday) > 0 and len(non_payday) > 0:
        multipliers['payday'] = payday['customers'].median() / non_payday['customers'].median()
    
    # Events
    if events_df is not None and not events_df.empty:
        ev_dates = pd.to_datetime(events_df['date']).dt.normalize()
        event_days = df[df['date'].isin(ev_dates)]
        if len(event_days) > 0:
            multipliers['event'] = event_days['multiplier'].median()
    
    return multipliers


def get_feature_list(include_sdly=True, include_contextual=True, include_recent=True):
    """
    Returns the list of features for model training.
    Organized by feature groups.
    """
    base_features = [
        'month', 'dayofweek', 'weekofyear', 'quarter',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    
    special_date_features = [
        'is_payday', 'is_mid_month_pay', 'is_end_month_pay',
        'is_weekend', 'is_friday', 'is_monday',
        'is_new_year', 'is_christmas_season', 'is_valentines',
        'is_event', 'days_to_event', 'days_from_event'
    ]
    
    lag_features = [
        'customers_lag_1', 'customers_lag_7', 'customers_lag_14', 
        'customers_lag_21', 'customers_lag_28'
    ]
    
    rolling_features = [
        'customers_roll_mean_7', 'customers_roll_mean_14', 'customers_roll_mean_30',
        'customers_roll_std_7', 'customers_roll_std_14'
    ]
    
    # NEW: Contextual window features (SDLY ± 14 days)
    contextual_features = [
        'sdly_customers', 'sdly_sales', 'sdly_atv',
        'sdly_window_cust_mean', 'sdly_window_cust_std',
        'sdly_window_sales_mean', 'sdly_window_atv_mean',
        'sdly_trend_before', 'sdly_trend_after',
        'sdly_momentum', 'sdly_cust_growth', 'sdly_sales_growth'
    ] if include_contextual else []
    
    # NEW: 8-week recent trend features
    recent_features = [
        'recent_weighted_cust_mean', 'recent_weighted_sales_mean', 'recent_weighted_atv_mean',
        'recent_trend_slope', 'recent_acceleration', 'recent_volatility',
        'recent_momentum_4w', 'recent_momentum_2w', 'recent_dow_factor',
        'recent_purchase_power_trend',
        'week_1_avg_cust', 'week_2_avg_cust', 'week_4_avg_cust', 'week_8_avg_cust'
    ] if include_recent else []
    
    # NEW: YoY comparison
    yoy_features = [
        'yoy_customer_growth', 'yoy_sales_growth', 'yoy_atv_growth'
    ]
    
    all_features = (base_features + special_date_features + lag_features + 
                   rolling_features + contextual_features + recent_features + yoy_features)
    
    return all_features
