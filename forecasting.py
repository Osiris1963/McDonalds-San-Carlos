# forecasting.py (Definitive Version with Time-Weighted Learning)
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from datetime import timedelta
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from data_processing import create_features

def generate_ph_holidays(start_date, end_date, events_df):
    """Generates a DataFrame of PH holidays for Prophet, including user-provided future activities."""
    holidays_list = [
        {'holiday': 'New Year\'s Day', 'ds': '2025-01-01'},
        {'holiday': 'Maundy Thursday', 'ds': '2025-04-17'},
        {'holiday': 'Good Friday', 'ds': '2025-04-18'},
        {'holiday': 'Araw ng Kagitingan', 'ds': '2025-04-09'},
        {'holiday': 'Labor Day', 'ds': '2025-05-01'},
        {'holiday': 'Independence Day', 'ds': '2025-06-12'},
        {'holiday': 'Ninoy Aquino Day', 'ds': '2025-08-21'},
        {'holiday': 'National Heroes Day', 'ds': '2025-08-25'},
        {'holiday': 'All Saints\' Day', 'ds': '2025-11-01'},
        {'holiday': 'Bonifacio Day', 'ds': '2025-11-30'},
        {'holiday': 'Feast of the Immaculate Conception', 'ds': '2025-12-08'},
        {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'},
    ]
    ph_holidays = pd.DataFrame(holidays_list)
    ph_holidays['ds'] = pd.to_datetime(ph_holidays['ds'])

    payday_events = []
    if pd.notna(start_date):
        current_date = start_date
        while current_date <= end_date:
            if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
                payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
            current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    if events_df is not None and not events_df.empty:
        if 'date' in events_df.columns:
            user_events = events_df[['date', 'activity_name']].copy()
            user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
            user_events['ds'] = pd.to_datetime(user_events['ds'])
            all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

def build_future_dataframe(future_dates, historical_df, final_features, events_df):
    """Builds a dataframe with all necessary features for a given set of future dates."""
    if future_dates.empty:
        return pd.DataFrame()

    future_df = pd.DataFrame({'date': future_dates})
    future_df = create_features(future_df, events_df)
    
    historical_indexed = historical_df.set_index('date')
    
    for col in final_features:
        if 'lag' in col:
            target_col, _, lag_days_str = col.partition('_lag_')
            lag_days = int(lag_days_str)
            source_dates = future_df['date'] - timedelta(days=lag_days)
            future_df[col] = historical_indexed.reindex(source_dates)[target_col].values

    for col in final_features:
        if col not in future_df.columns:
            future_df[col] = 0
            
    future_df = future_df[final_features]
    future_df.fillna(method='ffill', inplace=True)
    future_df.fillna(0, inplace=True)
    
    return future_df

def run_day_specific_pipeline(df_train, X_train, y_train, future_dates, final_features, historical_df_full, events_df):
    """Runs the full two-stage model pipeline with time-weighted learning."""
    # --- NEW: Time-Weighted Learning ---
    # Create weights that give more importance to recent data.
    # The most recent date in the training set gets the highest weight.
    time_since_last_obs = (df_train['date'].max() - df_train['date']).dt.days
    decay_factor = 0.995 # This factor can be tuned, but 0.995 is a strong starting point.
    sample_weights = np.power(decay_factor, time_since_last_obs)
    # --- END NEW ---

    # --- Stage 1: Primary Model ---
    df_prophet = df_train[['date']].rename(columns={'date': 'ds'})
    df_prophet['y'] = y_train.values
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    
    prophet_model = Prophet(
        holidays=prophet_holidays,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.25
    )
    prophet_model.fit(df_prophet)

    lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    # Pass the sample_weights to the fit method
    lgbm.fit(X_train, y_train, sample_weight=sample_weights)

    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'seed': 42, 'n_jobs': -1}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    # Pass the sample_weights to the fit method
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = lgbm.predict(X_train)
    xgb_train_preds = xgb_model.predict(X_train)
    X_meta = pd.DataFrame({'prophet': prophet_train_fcst['yhat'].values, 'lgbm': lgbm_train_preds, 'xgb': xgb_train_preds})
    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    # Pass the sample_weights to the meta-learner as well
    meta_learner.fit(X_meta, y_train, sample_weight=sample_weights)

    X_future = build_future_dataframe(future_dates, historical_df_full, final_features, events_df)

    if X_future.empty:
        return pd.DataFrame(), None

    prophet_future_fcst = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
    lgbm_future_preds = lgbm.predict(X_future)
    xgb_future_preds = xgb_model.predict(X_future)
    
    X_meta_future = pd.DataFrame({'prophet': prophet_future_fcst['yhat'].values, 'lgbm': lgbm_future_preds, 'xgb': xgb_future_preds})
    
    primary_forecast = meta_learner.predict(X_meta_future)
    
    # --- Stage 2: Residual Model ---
    primary_train_preds = meta_learner.predict(X_meta)
    residuals = y_train - primary_train_preds
    
    residual_lgbm_params = {'objective': 'regression_l1', 'n_estimators': 100, 'learning_rate': 0.05, 'verbose': -1, 'seed': 123}
    residual_model = lgb.LGBMRegressor(**residual_lgbm_params)
    residual_model.fit(X_train, residuals, sample_weight=sample_weights) # Also weight the residual model
    predicted_residuals = residual_model.predict(X_future)

    final_forecast = primary_forecast + predicted_residuals
    day_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_forecast})
    
    return day_forecast_df, prophet_model

def generate_forecast(historical_df, events_df, periods=15):
    """Main forecasting function with time-weighted learning."""
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    prophet_model_for_insights = None 
    
    last_historical_date = df_featured['date'].max()

    day_mapping = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        if len(df_day) < 30: continue

        all_features = [col for col in df_day.columns if df_day[col].dtype in ['int64', 'float64', 'int32'] and col not in ['total_sales', 'base_sales', 'customers', 'atv', 'date', 'add_on_sales']]
        constant_cols = [col for col in all_features if df_day[col].nunique() < 2]
        final_features = [f for f in all_features if f not in constant_cols]
        
        first_future_day = last_historical_date + timedelta(days=1)
        while first_future_day.weekday() != day_of_week:
            first_future_day += timedelta(days=1)
        
        freq_str = f'W-{day_mapping[day_of_week]}'
        future_dates_for_day = pd.date_range(start=first_future_day, periods=periods, freq=freq_str)

        if future_dates_for_day.empty: continue

        # --- Process Customers ---
        target_cust = 'customers'
        df_train_cust = df_day.dropna(subset=final_features + [target_cust])
        if len(df_train_cust) < 20: continue
        X_train_cust, y_train_cust = df_train_cust[final_features], df_train_cust[target_cust]
        
        cust_fcst, prophet_model_cust = run_day_specific_pipeline(df_train_cust, X_train_cust, y_train_cust, future_dates_for_day, final_features, df_featured, events_df)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            if prophet_model_cust: prophet_model_for_insights = prophet_model_cust

        # --- Process ATV ---
        target_atv = 'atv'
        df_train_atv = df_day.dropna(subset=final_features + [target_atv])
        if len(df_train_atv) < 20: continue
        X_train_atv, y_train_atv = df_train_atv[final_features], df_train_atv[target_atv]
        
        atv_fcst, _ = run_day_specific_pipeline(df_train_atv, X_train_atv, y_train_atv, future_dates_for_day, final_features, df_featured, events_df)
        if not atv_fcst.empty: all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts: return pd.DataFrame(), None 

    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds')
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds')
    
    if cust_forecast_final.empty or atv_forecast_final.empty: return pd.DataFrame(), None

    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')
    if final_df.empty: return pd.DataFrame(), None
        
    final_df['forecast_base_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    historical_df['dayofweek'] = historical_df['date'].dt.dayofweek
    avg_addons_by_day = historical_df.groupby('dayofweek')['add_on_sales'].mean().reset_index()
    final_df['dayofweek'] = final_df['ds'].dt.dayofweek
    final_df = pd.merge(final_df, avg_addons_by_day, on='dayofweek', how='left').fillna(0)
    
    final_df['forecast_sales'] = final_df['forecast_base_sales'] + final_df['add_on_sales']
    
    for col in ['forecast_sales', 'forecast_customers', 'forecast_atv']:
        final_df[col] = final_df[col].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].round()

    return final_df.sort_values('ds').reset_index(drop=True).head(periods), prophet_model_for_insights
