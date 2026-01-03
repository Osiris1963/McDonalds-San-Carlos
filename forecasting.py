# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def calculate_performance_metrics(historical_df, forecast_log_df):
    """
    SENIOR DEV ADDITION: Self-evaluation logic.
    Compares past predictions stored in Firestore with actual historical results.
    """
    if forecast_log_df.empty or historical_df.empty:
        return 1.0, 0.0  # Default multiplier (no change), 0 bias

    # Merge logs with actuals on date
    comparison = pd.merge(
        forecast_log_df, 
        historical_df[['date', 'customers', 'atv']], 
        left_on='forecast_for_date', 
        right_on='date'
    )

    if comparison.empty:
        return 1.0, 0.0

    # Calculate Bias: (Actual / Predicted)
    # If > 1, we are under-predicting. If < 1, we are over-predicting.
    cust_bias = (comparison['customers'] / comparison['predicted_customers']).mean()
    
    # Clip bias to prevent extreme swings (max 20% adjustment)
    cust_bias = np.clip(cust_bias, 0.8, 1.2)
    
    return cust_bias

def generate_customer_forecast(historical_df, events_df, forecast_log_df=None, periods=15):
    """
    Enhanced Recursive LGBM with Self-Learning Bias Correction.
    """
    # 1. Self-Learning Step: Calculate how much we missed last time
    bias_multiplier = 1.0
    if forecast_log_df is not None:
        bias_multiplier = calculate_performance_metrics(historical_df, forecast_log_df)

    # --- Model Training ---
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'

    # Optimization: Use Poisson regression for count data (customers)
    lgbm_params = {
        'objective': 'poisson', # Better for counting customers
        'metric': 'rmse',
        'n_estimators': 1200,
        'learning_rate': 0.03,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'seed': 42,
        'verbose': -1
    }
    
    # Weight recent data more heavily to "relearn" fast
    sample_weights = np.linspace(0.1, 1.0, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST], sample_weight=sample_weights)

    # --- Recursive Forecasting ---
    future_predictions_list = []
    history_for_recursion = historical_df.copy()
    last_date = history_for_recursion['date'].max()

    last_known_atv = history_for_recursion['atv'].iloc[-1] if not history_for_recursion.empty else 250

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        temp_df = pd.concat([history_for_recursion, pd.DataFrame([{'date': current_pred_date}])], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        pred_cust = model_cust.predict(X_pred)[0]
        
        # APPLY SELF-LEARNING BIAS
        pred_cust = pred_cust * bias_multiplier
        pred_cust = max(0, pred_cust)

        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'atv': last_known_atv,
            'sales': pred_cust * last_known_atv
        }
        future_predictions_list.append(new_row)
        history_for_recursion = pd.concat([history_for_recursion, pd.DataFrame([new_row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_predictions_list)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    forecast_df['forecast_customers'] = forecast_df['forecast_customers'].round().astype(int)
    
    return forecast_df[['ds', 'forecast_customers']], model_cust, bias_multiplier

def generate_atv_forecast(historical_df, events_df, periods=15):
    """
    Prophet ATV forecast with enhanced seasonality handling.
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
    future = model_atv.make_future_dataframe(periods=periods, freq='D')
    
    # Fill future regressors
    future['is_payday_period'] = future['ds'].apply(lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    
    # Event merging logic
    if events_df is not None and not events_df.empty:
        events_dates = pd.to_datetime(events_df['date']).dt.normalize()
        future['is_event'] = future['ds'].isin(events_dates).astype(int)
    else:
        future['is_event'] = 0

    forecast = model_atv.predict(future)
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    
    return forecast_final, model_atv
