# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using the definitive, unified architecture:
    A Direct Multi-Output LightGBM engine.

    This approach trains a single model per forecast day (horizon) that learns
    to predict BOTH customers and base_sales simultaneously, capturing the complex
    interdependencies between them and eliminating all previous sources of error.
    """
    # --- 1. Data and Feature Preparation ---
    df_featured = create_advanced_features(historical_df, events_df)
    
    FEATURES = [col for col in df_featured.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes', 'base_sales'
    ]]
    
    # --- 2. Model Training: Direct Multi-Output Strategy ---
    # // SENIOR DEV NOTE //: This is the final, wisest architecture.
    # We train one model per horizon. Each model predicts two targets at once.
    # This is simpler, more robust, and lets the model learn the internal dynamics.
    
    models = {} # A single dictionary to hold our multi-output models.

    # Using the original, trusted parameters for the LGBM model.
    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    for h in range(1, periods + 1):
        train_df = df_featured.copy()
        
        # The target 'y' is now a 2D array containing both target columns.
        train_df['target_customers'] = train_df['customers'].shift(-h)
        train_df['target_base_sales'] = train_df['base_sales'].shift(-h)

        train_df.dropna(subset=['target_customers', 'target_base_sales'], inplace=True)

        X_train = train_df[FEATURES]
        y_train_multi = train_df[['target_customers', 'target_base_sales']]

        # LightGBM natively handles multi-output regression when 'y' has multiple columns.
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train_multi)
        models[h] = model

    # --- 3. Forecasting ---
    X_pred_features = df_featured[FEATURES].iloc[-1:]
    future_predictions = []
    
    avg_addon_per_customer = historical_df['add_on_sales'].sum() / historical_df['customers'].replace(0, 1).sum()
    last_date = historical_df['date'].max()

    for h in range(1, periods + 1):
        # The model's predict() method will return an array with two values: [pred_cust, pred_base_sales]
        prediction = models[h].predict(X_pred_features)[0]
        
        pred_cust = max(0, prediction[0])
        pred_base_sales = max(0, prediction[1])
        
        pred_atv = (pred_base_sales / pred_cust) if pred_cust > 0 else 0
        
        estimated_add_on_sales = pred_cust * avg_addon_per_customer
        pred_total_sales = pred_base_sales + estimated_add_on_sales
        
        new_row = {
            'ds': last_date + timedelta(days=h),
            'forecast_customers': pred_cust,
            'forecast_atv': pred_atv,
            'forecast_sales': pred_total_sales,
        }
        future_predictions.append(new_row)

    # --- 4. Finalize and Return ---
    final_forecast = pd.DataFrame(future_predictions)
    
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    # Return the model for the first horizon for feature importance analysis
    return final_forecast, models.get(1)
