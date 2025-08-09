# forecasting.py
import pandas as pd
import numpy as np  # Imported for weighting
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a two-stage, unified LightGBM model
    with a recursive strategy for multi-step forecasting.
    
    // SENIOR DEV NOTE //: Refactored to include sample weighting for recency bias.
    """
    # --- 1. Model Training ---
    
    # Create a rich feature set from all historical data
    df_featured = create_advanced_features(historical_df, events_df)
    
    # Drop initial rows where rolling features couldn't be computed
    # This ensures the model trains on high-quality, complete data
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    # Define features and targets
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'
    TARGET_ATV = 'atv'

    # --- THIS IS THE CORRECTED SECTION (FROM YOUR ORIGINAL) ---
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    # // SENIOR DEV NOTE //: Implementing sample weighting.
    # We create a weight array that gives more importance to recent data.
    # The weight increases linearly from a starting point (e.g., 0.2) to 1.0 for the most recent data point.
    # This forces the model to prioritize minimizing errors on recent history, making it more adaptive.
    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    # Train the Customer Forecasting Model with sample weights
    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(
        train_df[FEATURES],
        train_df[TARGET_CUST],
        sample_weight=sample_weights  # The key change is here
    )

    # Train the ATV Forecasting Model with the same weighting
    model_atv = lgb.LGBMRegressor(**lgbm_params)
    model_atv.fit(
        train_df[FEATURES],
        train_df[TARGET_ATV],
        sample_weight=sample_weights  # And here
    )

    # --- 2. Recursive Forecasting ---
    # // SENIOR DEV NOTE //: The recursive strategy is acceptable but be aware that errors can compound.
    # The sample weighting helps mitigate this by ensuring the initial predictions are based on the most relevant data.
    
    future_predictions = []
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([history_df, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        pred_cust = model_cust.predict(X_pred)[0]
        pred_atv = model_atv.predict(X_pred)[0]
        
        # Ensure predictions are non-negative
        pred_cust = max(0, pred_cust)
        pred_atv = max(0, pred_atv)

        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'atv': pred_atv,
            'sales': pred_cust * pred_atv,
            'add_on_sales': 0,
            'day_type': 'Forecast' # Add a type for clarity in the history
        }
        future_predictions.append(new_row)
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 3. Finalize and Return ---
    
    final_forecast = pd.DataFrame(future_predictions)
    final_forecast.rename(columns={
        'date': 'ds',
        'customers': 'forecast_customers',
        'atv': 'forecast_atv',
        'sales': 'forecast_sales'
    }, inplace=True)
    
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
