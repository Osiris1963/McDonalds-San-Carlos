# Proposed replacement for forecasting.py

import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from datetime import timedelta
import numpy as np
from data_processing import create_features # Assumes our new version

def generate_ph_holidays(start_date, end_date, events_df):
    # This function is fine as-is, no changes needed.
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
            payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 0})
        current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)


def train_and_predict_stacked_model(df_featured, target, periods, events_df):
    """Trains a single stacked ensemble model on the entire dataset."""
    
    # --- Define Features ---
    all_features = [col for col in df_featured.columns if df_featured[col].dtype in ['int64', 'float64', 'int32'] and col not in ['sales', 'customers', 'atv', 'date']]
    constant_cols = [col for col in all_features if df_featured[col].nunique(dropna=False) < 2]
    final_features = [f for f in all_features if f not in constant_cols]
    
    df_train = df_featured.dropna(subset=final_features + [target])

    X_train = df_train[final_features]
    y_train = df_train[target]

    last_date = df_featured['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)

    # --- Base Model 1: Prophet ---
    df_prophet = df_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_prophet)

    # --- Base Model 2 & 3: LGBM & XGBoost ---
    lgbm_params = {'objective': 'regression_l1', 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
    xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'seed': 42, 'n_jobs': -1}

    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)

    # --- Stacking Ensemble ---
    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = lgbm.predict(X_train)
    xgb_train_preds = xgb_model.predict(X_train)
    
    X_meta = pd.DataFrame({'prophet': prophet_train_fcst['yhat'].values, 'lgbm': lgbm_train_preds, 'xgb': xgb_train_preds})
    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    meta_learner.fit(X_meta, y_train)

    # --- Future Prediction ---
    future_df_placeholder = pd.DataFrame({'date': future_dates})
    # To create features for the future, we need to append it to the historical data, create features, then filter
    combined_for_future_features = pd.concat([df_featured, future_df_placeholder], ignore_index=True)
    combined_with_features = create_features(combined_for_future_features, events_df)
    
    X_future = combined_with_features[combined_with_features['date'].isin(future_dates)]
    X_future = X_future[final_features]
    X_future.fillna(method='ffill', inplace=True) # Fill any gaps from feature creation
    X_future.fillna(0, inplace=True)

    prophet_future_fcst = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
    lgbm_future_preds = lgbm.predict(X_future)
    xgb_future_preds = xgb_model.predict(X_future)

    X_meta_future = pd.DataFrame({'prophet': prophet_future_fcst['yhat'].values, 'lgbm': lgbm_future_preds, 'xgb': xgb_future_preds})
    final_preds = meta_learner.predict(X_meta_future)
    
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_preds})
    return forecast_df, prophet_model


def generate_forecast(historical_df, events_df, periods=15):
    """Generates a forecast using a single, unified model."""
    df_featured = create_features(historical_df, events_df)
    
    # Model for Customers
    cust_fcst_df, prophet_model = train_and_predict_stacked_model(df_featured, 'customers', periods, events_df)
    cust_fcst_df.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    
    # Model for ATV
    atv_fcst_df, _ = train_and_predict_stacked_model(df_featured, 'atv', periods, events_df)
    atv_fcst_df.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    # Combine Forecasts
    final_df = pd.merge(cust_fcst_df, atv_fcst_df, on='ds', how='inner')
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    # Clean up final predictions
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    return final_df.head(periods), prophet_model
