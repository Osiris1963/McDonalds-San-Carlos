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
    // SENIOR DEV NOTE //: This model is specialized for customer traffic, leveraging a complex
    feature set to capture non-linear relationships. The recursive forecasting is replaced
    by a more stable batch prediction on a future feature-engineered dataframe.
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
    
    # Create a combined df to generate features for the future dates
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
    // SENIOR DEV NOTE //: Prophet is ideal for this task. It excels at capturing seasonality
    (yearly, weekly) and holiday/event effects, which are primary drivers of ATV.
    This provides a more stable and interpretable forecast for this specific metric.
    """
    # --- 1. Data Preparation ---
    prophet_df, regressor_names = prepare_data_for_prophet(historical_df, events_df)
    
    # --- 2. Model Training ---
    model_atv = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative', # Sales often have multiplicative seasonality
        growth='linear'
    )
    
    for regressor in regressor_names:
        model_atv.add_regressor(regressor, mode='multiplicative')

    model_atv.fit(prophet_df)

    # --- 3. Future Prediction ---
    future = model_atv.make_future_dataframe(periods=periods, freq='D')
    
    # We must populate the regressor columns for the future dates
    future_prepared, _ = prepare_data_for_prophet(pd.DataFrame({'date': future['ds']}), events_df)
    future = pd.merge(future.drop(columns=regressor_names, errors='ignore'), future_prepared, on='ds')
    
    forecast = model_atv.predict(future)
    
    # --- 4. Finalize and Return ---
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    forecast_final['forecast_atv'] = forecast_final['forecast_atv'].clip(lower=0)
    
    return forecast_final, model_atv
