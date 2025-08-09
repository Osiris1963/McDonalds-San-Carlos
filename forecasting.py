# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast by predicting customer traffic with LightGBM and applying a
    stable, context-aware historical Average Transaction Value (ATV).

    // SENIOR DEV RE-ENGINEERING NOTE //:
    This is the final, definitive architecture based on the insight that ATV is a stable metric.
    1.  The AI model now ONLY predicts customer traffic. The sales and ATV models are removed.
    2.  For each future day, we calculate a historical ATV based on its context (e.g., the average of all past Tuesdays for a future Tuesday).
    3.  A robust 14-day rolling average is used as a fallback if a specific context has no history.
    4.  Final sales are calculated as: (Predicted Customers) * (Historical Contextual ATV).
    This approach provides maximum stability and grounds the sales forecast in proven reality.
    """
    # --- 1. Model Training (Customer Model Only) ---

    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)

    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'

    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
        'boosting_type': 'gbdt',
    }

    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(
        train_df[FEATURES],
        train_df[TARGET_CUST],
        sample_weight=sample_weights
    )

    # --- 2. Prepare Historical Data for ATV Calculation ---

    # // SENIOR DEV RE-ENGINEERING NOTE //: Calculate the true historical ATV for all past data.
    # This will be our lookup source. We handle division by zero gracefully.
    history_df = historical_df.copy()
    customers_safe = history_df['customers'].replace(0, np.nan)
    base_sales = history_df['sales'] - history_df.get('add_on_sales', 0)
    history_df['historical_atv'] = (base_sales / customers_safe).fillna(0)
    history_df['dayofweek'] = history_df['date'].dt.dayofweek
    
    # Calculate a global fallback ATV based on the last 14 days of valid data
    recent_valid_atv = history_df[history_df['historical_atv'] > 0].tail(14)['historical_atv']
    fallback_atv = recent_valid_atv.mean() if not recent_valid_atv.empty else 0

    # --- 3. Recursive Forecasting with Stable ATV ---

    future_predictions = []
    recursive_history = history_df.copy()
    last_date = recursive_history['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([recursive_history, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        # --- Predict Customers (AI Task) ---
        pred_cust = max(0, model_cust.predict(X_pred)[0])

        # --- Calculate Stable ATV (Historical Logic Task) ---
        current_dayofweek = current_pred_date.dayofweek
        
        # Find all historical data for the same day of the week
        context_df = history_df[history_df['dayofweek'] == current_dayofweek]
        valid_context_atv = context_df[context_df['historical_atv'] > 0]['historical_atv']
        
        if not valid_context_atv.empty:
            stable_atv = valid_context_atv.mean()
        else:
            # If no history for this day (e.g., no past Tuesdays), use the 14-day fallback
            stable_atv = fallback_atv
            
        # --- Calculate Final Sales ---
        pred_sales = pred_cust * stable_atv

        # This new row is used to generate features for the *next* day's customer prediction
        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'sales': pred_sales,
            'atv': stable_atv,
            'add_on_sales': 0,
            'day_type': 'Forecast'
        }
        future_predictions.append(new_row)
        recursive_history = pd.concat([recursive_history, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 4. Finalize and Return ---
    
    final_forecast = pd.DataFrame(future_predictions)
    final_forecast.rename(columns={
        'date': 'ds', 'customers': 'forecast_customers',
        'atv': 'forecast_atv', 'sales': 'forecast_sales'
    }, inplace=True)
    
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
