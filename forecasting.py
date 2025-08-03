# forecasting.py (Upgraded with Trend Weighting)
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
    current_date = start_date
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

def run_day_specific_models(df_day_featured, target, day_of_week, periods, events_df):
    if len(df_day_featured) < 20: 
        return pd.DataFrame(), None

    all_features = [col for col in df_day_featured.columns if df_day_featured[col].dtype in ['int64', 'float64', 'int32'] and col not in ['sales', 'customers', 'atv', 'date']]
    constant_cols = [col for col in all_features if df_day_featured[col].nunique() < 2]
    final_features = [f for f in all_features if f not in constant_cols]
    
    df_train = df_day_featured.dropna(subset=final_features + [target])
    
    # --- NEW: Check if this is the customer model and create sample weights ---
    sample_weights = None
    if target == 'customers' and not df_train.empty:
        # 1. Calculate the age of each data point in days from the most recent record
        most_recent_date = df_train['date'].max()
        df_train['days_old'] = (most_recent_date - df_train['date']).dt.days
        
        # 2. Define a decay rate. This is a key hyperparameter.
        # A smaller value (e.g., 0.001) gives more weight to older data.
        # A larger value (e.g., 0.01) makes the model focus very heavily on the most recent data.
        # 0.005 is a balanced starting point, giving a half-life of ~138 days.
        decay_rate = 0.005 
        
        # 3. Apply exponential decay to create the weights
        sample_weights = np.exp(-decay_rate * df_train['days_old'])
        
        # We no longer need this column for training features
        df_train = df_train.drop(columns=['days_old'])
    # --- END NEW ---

    last_date = df_day_featured['date'].max()
    future_date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7) 
    future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
    
    if len(future_dates) == 0:
        return pd.DataFrame(), None

    # --- Base Model Training ---
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

    # --- Pre-tuned Hyperparameters (Result of Offline Optuna Study) ---
    lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt'}
    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'seed': 42, 'n_jobs': -1}

    lgbm = lgb.LGBMRegressor(**lgbm_params)
    # --- NEW: Pass weights to the fit method ---
    lgbm.fit(X_train, y_train, sample_weight=sample_weights)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    # --- NEW: Pass weights to the fit method ---
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # --- Stacking Ensemble ---
    # 1. Get base model predictions on the training data itself to train the meta-learner
    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = lgbm.predict(X_train)
    xgb_train_preds = xgb_model.predict(X_train)

    # 2. Create the meta-feature set
    X_meta = pd.DataFrame({
        'prophet': prophet_train_fcst['yhat'].values,
        'lgbm': lgbm_train_preds,
        'xgb': xgb_train_preds
    })

    # 3. Train the meta-learner (RidgeCV is robust and fast)
    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    # --- NEW: Pass weights to the meta-learner's fit method ---
    meta_learner.fit(X_meta, y_train, sample_weight=sample_weights)

    # 4. Get base model predictions on FUTURE data
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

    # 5. Create the future meta-feature set
    X_meta_future = pd.DataFrame({
        'prophet': prophet_future_fcst['yhat'].values,
        'lgbm': lgbm_future_preds,
        'xgb': xgb_future_preds
    })

    # 6. Make the final prediction using the meta-learner
    final_stacked_preds = meta_learner.predict(X_meta_future)
    
    day_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_stacked_preds})
    
    return day_forecast_df, prophet_model

def generate_forecast(historical_df, events_df, periods=15):
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    prophet_model = None 
    
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        
        cust_fcst, prophet_model_cust = run_day_specific_models(df_day, 'customers', day_of_week, periods, events_df)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            if prophet_model_cust:
                prophet_model = prophet_model_cust
            
        atv_fcst, _ = run_day_specific_models(df_day, 'atv', day_of_week, periods, events_df)
        if not atv_fcst.empty:
            all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts:
        return pd.DataFrame(), None 

    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds').reset_index(drop=True)
    
    if cust_forecast_final.empty or atv_forecast_final.empty:
        return pd.DataFrame(), None

    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')
    if final_df.empty:
        return pd.DataFrame(), None
        
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    return final_df.head(periods), prophet_model
