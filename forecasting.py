# forecasting.py (Definitive Version with All Imports and Logic)
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb  # Audited: XGBoost import is present.
from sklearn.linear_model import RidgeCV
from datetime import timedelta
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from data_processing import create_features_for_customers, create_features_for_atv

def generate_ph_holidays(start_date, end_date, events_df):
    """Generates a DataFrame of PH holidays for Prophet."""
    holidays_list = [
        {'holiday': 'New Year\'s Day', 'ds': '2025-01-01'}, {'holiday': 'Maundy Thursday', 'ds': '2025-04-17'},
        {'holiday': 'Good Friday', 'ds': '2025-04-18'}, {'holiday': 'Araw ng Kagitingan', 'ds': '2025-04-09'},
        {'holiday': 'Labor Day', 'ds': '2025-05-01'}, {'holiday': 'Independence Day', 'ds': '2025-06-12'},
        {'holiday': 'Ninoy Aquino Day', 'ds': '2025-08-21'}, {'holiday': 'National Heroes Day', 'ds': '2025-08-25'},
        {'holiday': 'All Saints\' Day', 'ds': '2025-11-01'}, {'holiday': 'Bonifacio Day', 'ds': '2025-11-30'},
        {'holiday': 'Feast of the Immaculate Conception', 'ds': '2025-12-08'}, {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'},
    ]
    ph_holidays = pd.DataFrame(holidays_list); ph_holidays['ds'] = pd.to_datetime(ph_holidays['ds'])
    payday_events = []
    if pd.notna(start_date):
        current_date = start_date
        while current_date <= end_date:
            if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
                payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
            current_date += timedelta(days=1)
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])
    if events_df is not None and not events_df.empty and 'date' in events_df.columns:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])
    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

def generate_forecast(historical_df, events_df, periods=15):
    """Main forecasting function with strict feature isolation."""
    df_featured_cust = create_features_for_customers(historical_df, events_df)
    df_featured_atv = create_features_for_atv(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    prophet_model_for_insights = None 
    last_historical_date = historical_df['date'].max()
    day_mapping = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

    for day_of_week in range(7):
        # --- Process Customers ---
        df_day_cust = df_featured_cust[df_featured_cust['date'].dt.dayofweek == day_of_week].copy()
        if len(df_day_cust) < 30: continue
        
        numeric_types = ['int64', 'int32', 'float64', 'uint8', 'int8']
        cust_features = [col for col in df_day_cust.columns if df_day_cust[col].dtype in numeric_types and col not in ['total_sales', 'base_sales', 'customers', 'atv', 'add_on_sales']]
        
        df_train_cust = df_day_cust.dropna(subset=cust_features + ['customers'])
        if len(df_train_cust) < 20: continue
        X_train_cust, y_train_cust = df_train_cust[cust_features], df_train_cust['customers']

        first_future_day = last_historical_date + timedelta(days=1)
        while first_future_day.weekday() != day_of_week:
            first_future_day += timedelta(days=1)
        freq_str = f'W-{day_mapping[day_of_week]}'
        future_dates_for_day = pd.date_range(start=first_future_day, periods=periods, freq=freq_str)
        if future_dates_for_day.empty: continue
        
        cust_fcst, prophet_model_cust = run_model_pipeline(df_train_cust, X_train_cust, y_train_cust, future_dates_for_day, cust_features, historical_df, events_df, is_customer_model=True)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            if prophet_model_cust: prophet_model_for_insights = prophet_model_cust

        # --- Process ATV ---
        df_day_atv = df_featured_atv[df_featured_atv['date'].dt.dayofweek == day_of_week].copy()
        if len(df_day_atv) < 30: continue

        atv_features = [col for col in df_day_atv.columns if df_day_atv[col].dtype in numeric_types and col not in ['total_sales', 'base_sales', 'customers', 'atv', 'add_on_sales']]
        df_train_atv = df_day_atv.dropna(subset=atv_features + ['atv'])
        if len(df_train_atv) < 20: continue
        X_train_atv, y_train_atv = df_train_atv[atv_features], df_train_atv['atv']
        
        atv_fcst, _ = run_model_pipeline(df_train_atv, X_train_atv, y_train_atv, future_dates_for_day, atv_features, historical_df, events_df, is_customer_model=False)
        if not atv_fcst.empty: all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts: return pd.DataFrame(), None 
    
    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds')
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds')
    if cust_forecast_final.empty or atv_forecast_final.empty: return pd.DataFrame(), None

    final_df = pd.merge(cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}), atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}), on='ds', how='inner')
    if final_df.empty: return pd.DataFrame(), None
        
    final_df['forecast_base_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    historical_df['dayofweek'] = historical_df['date'].dt.dayofweek
    avg_addons_by_day = historical_df.groupby('dayofweek')['add_on_sales'].mean().reset_index()
    final_df['dayofweek'] = final_df['ds'].dt.dayofweek
    final_df = pd.merge(final_df, avg_addons_by_day, on='dayofweek', how='left').fillna(0)
    
    final_df['forecast_sales'] = final_df['forecast_base_sales'] + final_df['add_on_sales']
    
    for col in ['forecast_sales', 'forecast_customers', 'forecast_atv']: final_df[col] = final_df[col].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].round()

    return final_df.sort_values('ds').reset_index(drop=True).head(periods), prophet_model_for_insights

def run_model_pipeline(df_train, X_train, y_train, future_dates, final_features, historical_df_full, events_df, is_customer_model):
    """Generic model pipeline for either customer or ATV, with strategy determined by flag."""
    use_time_weights = is_customer_model
    prophet_changepoint_scale = 0.15 if is_customer_model else 0.05

    sample_weights = None
    if use_time_weights:
        time_since_last_obs = (df_train['date'].max() - df_train['date']).dt.days
        sample_weights = np.power(0.995, time_since_last_obs)

    df_prophet = df_train[['date']].rename(columns={'date': 'ds'}); df_prophet['y'] = y_train.values
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=prophet_changepoint_scale)
    prophet_model.fit(df_prophet)

    lgbm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=200, learning_rate=0.05, verbose=-1, seed=42)
    lgbm.fit(X_train, y_train, sample_weight=sample_weights)

    xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=5, seed=42)
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    X_meta = pd.DataFrame({'prophet': prophet_train_fcst['yhat'].values, 'lgbm': lgbm.predict(X_train), 'xgb': xgb.predict(X_train)})
    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    meta_learner.fit(X_meta, y_train, sample_weight=sample_weights)

    X_future = build_future_dataframe_generic(future_dates, historical_df_full, final_features, events_df, is_customer_model)
    if X_future.empty: return pd.DataFrame(), None

    X_meta_future = pd.DataFrame({'prophet': prophet_model.predict(pd.DataFrame({'ds': future_dates}))['yhat'].values, 'lgbm': lgbm.predict(X_future), 'xgb': xgb.predict(X_future)})
    primary_forecast = meta_learner.predict(X_meta_future)

    residuals = y_train - meta_learner.predict(X_meta)
    residual_model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100, learning_rate=0.05, verbose=-1, seed=123)
    residual_model.fit(X_train, residuals, sample_weight=sample_weights)
    final_forecast = primary_forecast + residual_model.predict(X_future)
    
    return pd.DataFrame({'ds': future_dates, 'yhat': final_forecast}), prophet_model

def build_future_dataframe_generic(future_dates, historical_df, final_features, events_df, is_customer_model):
    """Generic future dataframe builder."""
    if future_dates.empty: return pd.DataFrame()
    
    future_df_base = pd.DataFrame({'date': future_dates})

    if is_customer_model:
        future_df = create_features_for_customers(future_df_base, events_df)
    else:
        placeholder_df = pd.concat([historical_df, future_df_base], ignore_index=True)
        all_features_df = create_features_for_atv(placeholder_df, events_df)
        future_df = all_features_df[all_features_df['date'].isin(future_dates)].copy()

    for col in final_features:
        if col not in future_df.columns: future_df[col] = 0
            
    future_df = future_df[final_features]
    future_df.fillna(method='ffill', inplace=True); future_df.fillna(0, inplace=True)
    return future_df
