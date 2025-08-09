# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast by training the monetary model on 'base_sales' to align
    with the business's specific definition of ATV.
    """
    # --- 1. Model Training ---
    
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    # // SENIOR DEV NOTE //: We update the FEATURES list to exclude our new target 'base_sales'
    # and the original 'sales' to prevent data leakage.
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes',
        'base_sales' # Add new target to exclusion list
    ]]
    TARGET_CUST = 'customers'
    TARGET_BASE_SALES = 'base_sales' # The new, correct target for the monetary model.

    lgbm_params_cust = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    lgbm_params_sales = {
        'objective': 'quantile', 'alpha': 0.5, 'metric': 'regression_l1',
        'n_estimators': 1000, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'num_leaves': 31, 'verbose': -1,
        'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    }
    
    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params_cust)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST], sample_weight=sample_weights)

    # Train the monetary model on the correct target: base_sales
    model_base_sales = lgb.LGBMRegressor(**lgbm_params_sales)
    model_base_sales.fit(train_df[FEATURES], train_df[TARGET_BASE_SALES], sample_weight=sample_weights)

    # // SENIOR DEV NOTE //: To reconstruct total sales, we need a way to estimate add-on sales.
    # We'll use a simple, stable historical ratio: average add-on sales per customer.
    # We calculate this from the training data to avoid looking at future data.
    safe_customers = train_df['customers'].replace(0, 1) # Avoid division by zero
    avg_addon_per_customer = train_df['add_on_sales'].sum() / safe_customers.sum()


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

        # Predict customers and BASE sales
        pred_cust = model_cust.predict(X_pred)[0]
        pred_base_sales = model_base_sales.predict(X_pred)[0]
        
        pred_cust = max(0, pred_cust)
        pred_base_sales = max(0, pred_base_sales)

        # Calculate ATV using the business-aligned logic. This will now be accurate.
        pred_atv = (pred_base_sales / pred_cust) if pred_cust > 0 else 0
        
        # Estimate add-on sales and reconstruct the total sales forecast for the dashboard.
        estimated_add_on_sales = pred_cust * avg_addon_per_customer
        pred_total_sales = pred_base_sales + estimated_add_on_sales

        # This new row will be used to generate features for the *next* day's forecast.
        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'sales': pred_total_sales,  # Use reconstructed total sales for history
            'base_sales': pred_base_sales, # Add base_sales to history
            'atv': pred_atv,
            'add_on_sales': estimated_add_on_sales, # Add estimated add-ons
            'day_type': 'Forecast'
        }
        future_predictions.append(new_row)
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 3. Finalize and Return ---
    
    final_forecast = pd.DataFrame(future_predictions)
    # The 'sales' column in the dataframe now correctly refers to the reconstructed total sales.
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
