# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def calculate_performance_metrics(historical_df, forecast_log_df):
    """
    Standardizes date types to prevent merge errors and calculates accuracy bias.
    """
    if forecast_log_df is None or forecast_log_df.empty or historical_df.empty:
        return 1.0

    # 1. Standardize Historical Dates
    hist = historical_df.copy()
    hist['date'] = pd.to_datetime(hist['date']).dt.normalize()
    if hist['date'].dt.tz is not None:
        hist['date'] = hist['date'].dt.tz_localize(None)

    # 2. Standardize Log Dates
    logs = forecast_log_df.copy()
    # Ensure we use the correct column name from the log schema
    date_col = 'forecast_for_date' if 'forecast_for_date' in logs.columns else 'ds'
    logs[date_col] = pd.to_datetime(logs[date_col]).dt.normalize()
    if logs[date_col].dt.tz is not None:
        logs[date_col] = logs[date_col].dt.tz_localize(None)
    
    logs = logs.rename(columns={date_col: 'date'})

    # 3. Robust Merge
    comparison = pd.merge(
        logs[['date', 'predicted_customers']], 
        hist[['date', 'customers']], 
        on='date',
        how='inner'
    )

    if comparison.empty:
        return 1.0

    # Calculate Bias from last 7 days: Actual / Predicted
    comparison = comparison.sort_values('date').tail(7)
    comparison = comparison[comparison['predicted_customers'] > 0]
    
    if comparison.empty:
        return 1.0
        
    cust_bias = (comparison['customers'] / comparison['predicted_customers']).mean()
    
    # Clip adjustment to +/- 20% for stability
    return np.clip(cust_bias, 0.8, 1.2)

def generate_customer_forecast(historical_df, events_df, forecast_log_df=None, periods=15):
    """
    Recursive LGBM with a Self-Learning Bias Multiplier.
    """
    # Calculate how much we missed in recent history 
    bias_multiplier = calculate_performance_metrics(historical_df, forecast_log_df)
    
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]

    # Poisson is mathematically better for modeling customer counts 
    lgbm_params = {
        'objective': 'poisson', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'verbose': -1, 'seed': 42
    }
    
    # Weight recent data higher to "relearn" fast 
    sample_weights = np.linspace(0.2, 1.0, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df['customers'], sample_weight=sample_weights)

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
        
        # Apply the self-calculated bias adjustment 
        pred_cust = max(0, pred_cust * bias_multiplier)

        new_row = {
            'date': current_pred_date, 'customers': pred_cust,
            'atv': last_known_atv, 'sales': pred_cust * last_known_atv
        }
        future_predictions_list.append(new_row)
        history_for_recursion = pd.concat([history_for_recursion, pd.DataFrame([new_row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_predictions_list)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    forecast_df['forecast_customers'] = forecast_df['forecast_customers'].round().astype(int)
    
    return forecast_df[['ds', 'forecast_customers']], model_cust, bias_multiplier

def generate_atv_forecast(historical_df, events_df, periods=15):
    """Prophet-based ATV forecasting."""
    prophet_df, regressor_names = prepare_data_for_prophet(historical_df, events_df)
    model_atv = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
    
    for regressor in regressor_names:
        model_atv.add_regressor(regressor)

    model_atv.fit(prophet_df)
    future = model_atv.make_future_dataframe(periods=periods, freq='D')
    future['is_payday_period'] = future['ds'].apply(lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    
    if events_df is not None and not events_df.empty:
        events_dates = pd.to_datetime(events_df['date']).dt.normalize()
        future['is_event'] = future['ds'].isin(events_dates).astype(int)
    else:
        future['is_event'] = 0

    forecast = model_atv.predict(future)
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    return forecast_final, model_atv
