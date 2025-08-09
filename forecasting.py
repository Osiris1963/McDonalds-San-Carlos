# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a two-stage system with dedicated, specialized models.
    - The Customer Model remains untouched to preserve its excellent performance.
    - A new, robust Sales Model using Quantile Regression is introduced.
    - ATV is now a derived metric, calculated from the two primary forecasts.
    """
    # --- 1. Model Training ---
    
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    # // SENIOR DEV NOTE //: The feature set remains the same, but we will now train two distinct models
    # on two different targets, allowing each to specialize.
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'
    TARGET_SALES = 'sales' # We now target 'sales' directly.

    # --- Base LGBM Parameters ---
    # Parameters for the customer model remain as they were.
    lgbm_params_cust = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
        'boosting_type': 'gbdt',
    }

    # // SENIOR DEV NOTE //: These are the new parameters for our Sales Model.
    # The key change is 'objective': 'quantile' and 'alpha': 0.5.
    # This trains the model to predict the MEDIAN sales value, making it
    # extremely robust to outliers from unusually high or low sales days.
    lgbm_params_sales = {
        'objective': 'quantile', # Changed from 'regression_l1' to 'quantile'
        'alpha': 0.5,            # Set to 0.5 to predict the median
        'metric': 'quantile',
        'n_estimators': 1000, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'num_leaves': 31, 'verbose': -1,
        'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    }
    
    # Recency weighting remains a powerful tool for both models.
    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    # --- Train the Customer Model (Unchanged) ---
    model_cust = lgb.LGBMRegressor(**lgbm_params_cust)
    model_cust.fit(
        train_df[FEATURES],
        train_df[TARGET_CUST],
        sample_weight=sample_weights
    )

    # --- Train the new, robust Sales Model ---
    model_sales = lgb.LGBMRegressor(**lgbm_params_sales)
    model_sales.fit(
        train_df[FEATURES],
        train_df[TARGET_SALES], # Fit on TARGET_SALES
        sample_weight=sample_weights
    )

    # --- 2. Recursive Forecasting ---
    
    future_predictions = []
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([history_df, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        # Predict customers and sales independently
        pred_cust = model_cust.predict(X_pred)[0]
        pred_sales = model_sales.predict(X_pred)[0]
        
        pred_cust = max(0, pred_cust)
        pred_sales = max(0, pred_sales)

        # // SENIOR DEV NOTE //: ATV is now a derived metric, calculated from the two
        # more reliable primary forecasts. We handle the division-by-zero edge case.
        if pred_cust > 0:
            pred_atv = pred_sales / pred_cust
        else:
            pred_atv = 0

        # This new row will be used to generate features for the *next* day's forecast.
        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'sales': pred_sales,
            'atv': pred_atv, # Add the derived atv back into the loop history
            'add_on_sales': 0,
            'day_type': 'Forecast'
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
    
    # We still return model_cust for feature importance analysis as it's the primary driver.
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
