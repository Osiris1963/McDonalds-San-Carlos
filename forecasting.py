# forecasting.py (Re-Architected with Dual-Pipeline Logic)
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from datetime import timedelta
import warnings
import numpy as np
import holidays

warnings.filterwarnings("ignore", category=FutureWarning)

from data_processing import create_features

def generate_ph_holidays(start_date, end_date, events_df):
    """Generates a DataFrame of Philippine holidays, payday windows, and custom events."""
    years_to_generate = np.arange(start_date.year, end_date.year + 2)
    ph_holidays_list = []
    for date, name in sorted(holidays.PH(years=years_to_generate).items()):
        ph_holidays_list.append({'holiday': name, 'ds': date})
    
    ph_holidays = pd.DataFrame(ph_holidays_list)
    ph_holidays['ds'] = pd.to_datetime(ph_holidays['ds'])

    payday_events = []
    current_date = start_date - timedelta(days=30) 
    while current_date <= end_date:
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

# --- This complex model is now ONLY for trend-based variables like 'customers' ---
def run_day_specific_models(df_day_featured, target, day_of_week, periods, events_df):
    if len(df_day_featured) < 20: 
        return pd.DataFrame(), None

    cols_to_exclude = ['sales', 'add_on_sales', 'customers', 'atv', 'date', 'base_sales']
    all_features = [col for col in df_day_featured.columns if df_day_featured[col].dtype in ['int64', 'float64', 'int32'] and col not in cols_to_exclude]
    
    constant_cols = [col for col in all_features if df_day_featured[col].nunique() < 2]
    final_features = [f for f in all_features if f not in constant_cols]
    
    df_train = df_day_featured.dropna(subset=final_features + [target])
    
    last_date = df_day_featured['date'].max()
    future_date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7) 
    future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
    
    if len(future_dates) == 0:
        return pd.DataFrame(), None

    df_prophet = df_day_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(df_prophet)

    MIN_SAMPLES_FOR_TREE_MODELS = 15
    if len(df_train) < MIN_SAMPLES_FOR_TREE_MODELS:
        forecast_prophet = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
        return forecast_prophet[['ds', 'yhat']], prophet_model
        
    X_train = df_train[final_features]
    y_train = df_train[target]

    lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt'}
    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'seed': 42, 'n_jobs': -1}

    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = lgbm.predict(X_train)
    xgb_train_preds = xgb_model.predict(X_train)

    X_meta = pd.DataFrame({
        'prophet': prophet_train_fcst['yhat'].values,
        'lgbm': lgbm_train_preds,
        'xgb': xgb_train_preds
    })

    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    meta_learner.fit(X_meta, y_train)

    future_placeholder = pd.DataFrame({'date': future_dates})
    combined_df_for_features = pd.concat([df_day_featured, future_placeholder], ignore_index=True)
    combined_df_with_features = create_features(combined_df_for_features, events_df)
    
    X_future = combined_df_with_features[combined_df_with_features['date'].isin(future_dates)]
    X_future = X_future[final_features] 
    X_future.fillna(method='ffill', inplace=True) 
    X_future.fillna(0, inplace=True)

    prophet_future_fcst = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
    lgbm_future_preds = lgbm.predict(X_future)
    xgb_future_preds = xgb_model.predict(X_future)

    X_meta_future = pd.DataFrame({
        'prophet': prophet_future_fcst['yhat'].values,
        'lgbm': lgbm_future_preds,
        'xgb': xgb_future_preds
    })

    final_stacked_preds = meta_learner.predict(X_meta_future)
    day_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_stacked_preds})
    return day_forecast_df, prophet_model

# --- NEW: Dedicated model for stable variables like ATV ---
def generate_stable_atv_forecast(df_featured, periods):
    """
    Forecasts ATV based on the historical average for each day of the week.
    This model is simple, robust, and respects the "stable" nature of ATV.
    """
    if 'atv' not in df_featured.columns or df_featured.empty:
        return pd.DataFrame()

    # Calculate the mean ATV for each day of the week from historical data
    atv_by_day = df_featured.groupby(df_featured['date'].dt.dayofweek)['atv'].mean()

    # Create a future DataFrame
    last_date = df_featured['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['dayofweek'] = future_df['ds'].dt.dayofweek
    
    # Map the historical average to the future dates
    future_df['forecast_atv'] = future_df['dayofweek'].map(atv_by_day)
    
    # Forward-fill for any day that might not have been in the history (unlikely but safe)
    future_df['forecast_atv'].fillna(method='ffill', inplace=True)
    future_df['forecast_atv'].fillna(method='bfill', inplace=True)

    return future_df[['ds', 'forecast_atv']]


def generate_forecast(historical_df, events_df, periods=15):
    """
    Main orchestration function with a DUAL-PIPELINE approach.
    - Uses a complex, trend-based model for Customers.
    - Uses a simple, stable model for ATV.
    """
    df_featured = create_features(historical_df, events_df)
    
    # --- PIPELINE 1: Customer Forecasting (Complex Trend Model) ---
    all_cust_forecasts = []
    prophet_model_for_insights = None 
    
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        cust_fcst, prophet_model_cust = run_day_specific_models(df_day, 'customers', day_of_week, periods, events_df)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            if prophet_model_cust:
                prophet_model_for_insights = prophet_model_cust
    
    if not all_cust_forecasts:
        return pd.DataFrame(), None 
    
    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)

    # --- PIPELINE 2: ATV Forecasting (Simple Stability Model) ---
    atv_forecast_final = generate_stable_atv_forecast(df_featured, periods=periods)

    if cust_forecast_final.empty or atv_forecast_final.empty:
        return pd.DataFrame(), None

    # --- Combine the forecasts from the two separate pipelines ---
    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')
    if final_df.empty:
        return pd.DataFrame(), None
        
    final_df['forecast_base_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_base_sales'] = final_df['forecast_base_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    # Note: prophet_model is now only for customer insights
    return final_df.head(periods), prophet_model_for_insights
