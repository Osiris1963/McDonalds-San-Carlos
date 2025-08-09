# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def generate_customer_forecast(historical_df, events_df, periods=15):
    """
    Generates a customer forecast using a LightGBM model with recency weighting.
    """
    # --- 1. Model Training ---
    df_featured = create_advanced_features(historical_df.copy(), events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'

    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'num_leaves': 31,
        'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    }
    
    start_weight = 0.2
    sample_weights = np.linspace(start_weight, 1.0, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST], sample_weight=sample_weights)

    # --- 2. Future Prediction ---
    last_date = historical_df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    future_df = pd.DataFrame({'date': future_dates})
    
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    combined_featured = create_advanced_features(combined_df, events_df)
    
    X_pred = combined_featured[FEATURES].iloc[-periods:]
    
    future_predictions = model_cust.predict(X_pred)
    
    # --- 3. Finalize and Return ---
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'forecast_customers': future_predictions
    })
    
    forecast_df['forecast_customers'] = forecast_df['forecast_customers'].clip(lower=0).round().astype(int)
    
    return forecast_df, model_cust


def generate_atv_forecast(historical_df, events_df, periods=15):
    """
    Generates an ATV (Average Transaction Value) forecast using Prophet.
    """
    # --- 1. Data Preparation for Training ---
    prophet_df, regressor_names = prepare_data_for_prophet(historical_df, events_df)
    
    # --- 2. Model Training ---
    model_atv = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        growth='linear'
    )
    
    for regressor in regressor_names:
        model_atv.add_regressor(regressor, mode='multiplicative')

    model_atv.fit(prophet_df)

    # --- 3. Future Prediction ---
    future = model_atv.make_future_dataframe(periods=periods, freq='D')
    
    # --- THIS IS THE CORRECTED SECTION ---
    # SENIOR DEV NOTE: This logic now correctly builds the required regressor columns 
    # directly onto Prophet's future dataframe, avoiding the previous KeyError.
    future['is_payday_period'] = future['ds'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)

    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique.rename(columns={'date': 'ds'}, inplace=True)
        events_df_unique['ds'] = pd.to_datetime(events_df_unique['ds']).dt.normalize()
        events_df_unique['is_event'] = 1
        
        future = pd.merge(future, events_df_unique[['ds', 'is_event']], on='ds', how='left').fillna(0)
        future['is_event'] = future['is_event'].astype(int)
    else:
        future['is_event'] = 0

    forecast = model_atv.predict(future)
    
    # --- 4. Finalize and Return ---
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    forecast_final['forecast_atv'] = forecast_final['forecast_atv'].clip(lower=0)
    
    return forecast_final, model_atv
