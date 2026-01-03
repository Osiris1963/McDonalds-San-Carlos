# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def calculate_forecast_bias(historical_df, db_client):
    """
    SENIOR DEV UPGRADE: The 'Human' Learning Loop.
    Compares previous predictions in Firestore to actual results to calculate a correction factor.
    """
    try:
        # Load the log of what we PREDICTED
        docs = db_client.collection('forecast_log').stream()
        log_records = [doc.to_dict() for doc in docs]
        if not log_records: return 1.0 # No history? No correction.

        log_df = pd.DataFrame(log_records)
        log_df['forecast_for_date'] = pd.to_datetime(log_df['forecast_for_date']).dt.normalize()
        
        # Merge with ACTUAL historical data
        comparison = pd.merge(
            log_df, 
            historical_df[['date', 'customers']], 
            left_on='forecast_for_date', 
            right_on='date'
        )

        if len(comparison) < 5: return 1.0 # Need at least 5 days to trust the bias
        
        # Calculate Bias: (Actual / Predicted)
        # If > 1, we are under-predicting. If < 1, we are over-predicting.
        bias = (comparison['customers'] / comparison['predicted_customers']).mean()
        return np.clip(bias, 0.85, 1.15) # Safety cap: don't adjust more than 15%
    except:
        return 1.0

def generate_customer_forecast(historical_df, events_df, db_client, periods=15):
    """
    UPGRADED: Direct Multi-Step Forecasting with Poisson Objective.
    """
    df_featured = create_advanced_features(historical_df, events_df)
    
    # We only train on dates where we have our target (customers)
    train_df = df_featured[df_featured['customers'] > 0].copy()
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    
    # SENIOR SETTINGS: Poisson is best for 'count' data like customers
    lgbm_params = {
        'objective': 'poisson', 
        'metric': 'rmse',
        'learning_rate': 0.03,
        'n_estimators': 1200,
        'num_leaves': 64,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df['customers'])

    # Prepare Future DataFrame for prediction
    last_date = historical_df['date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
    future_df = pd.DataFrame({'date': future_dates})
    
    # Combine hist + future to generate features for the future dates
    full_context = pd.concat([historical_df, future_df], ignore_index=True)
    full_featured = create_advanced_features(full_context, events_df)
    
    # Extract only the future rows for prediction
    X_future = full_featured[full_featured['date'] > last_date][FEATURES]
    
    preds = model_cust.predict(X_future)
    
    # APPLY SELF-LEARNING BIAS
    bias_factor = calculate_forecast_bias(historical_df, db_client)
    preds = preds * bias_factor

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'forecast_customers': np.round(np.clip(preds, 0, None)).astype(int)
    })
    
    return forecast_df, model_cust

def generate_atv_forecast(historical_df, events_df, periods=15):
    """
    Prophet handles the 'Macro' trends (Holidays/Seasonality) for Average Ticket Value.
    """
    prophet_df, regressor_names = prepare_data_for_prophet(historical_df, events_df)
    
    model_atv = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    for regressor in regressor_names:
        model_atv.add_regressor(regressor)

    model_atv.fit(prophet_df)

    future = model_atv.make_future_dataframe(periods=periods)
    
    # Add regressor logic to future df
    future['is_payday_period'] = future['ds'].apply(lambda x: 1 if x.day in [14,15,16,29,30,31,1,2] else 0)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    
    # Event mapping
    if events_df is not None and not events_df.empty:
        event_dates = pd.to_datetime(events_df['date']).dt.normalize().tolist()
        future['is_event'] = future['ds'].apply(lambda x: 1 if x in event_dates else 0)
    else:
        future['is_event'] = 0

    forecast = model_atv.predict(future)
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    
    return forecast_final, model_atv
