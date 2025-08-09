# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a two-stage, unified LightGBM model
    with a recursive strategy for multi-step forecasting.
    
    // SENIOR DEV RE-ENGINEERING NOTE //:
    The core strategy has been changed. Instead of predicting 'customers' and 'atv' independently,
    we now predict 'customers' and 'sales'. The 'atv' is then derived from these two more stable
    predictions. This maintains the mathematical relationship (sales = customers * atv) and
    prevents the ATV forecast from becoming unrealistically high.
    The excellent customer model logic is preserved as requested.
    """
    # --- 1. Model Training ---
    
    # Create a rich feature set from all historical data
    df_featured = create_advanced_features(historical_df, events_df)
    
    # Drop initial rows where rolling features couldn't be computed
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    # Define features and targets
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'
    # // SENIOR DEV RE-ENGINEERING NOTE //: The second target is now 'sales', not 'atv'.
    TARGET_SALES = 'sales'

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
    
    # Implement sample weighting for recency bias
    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    # Train the Customer Forecasting Model (Unchanged, as requested)
    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(
        train_df[FEATURES],
        train_df[TARGET_CUST],
        sample_weight=sample_weights
    )

    # // SENIOR DEV RE-ENGINEERING NOTE //: We now train a Sales Forecasting Model instead of an ATV model.
    # It uses the same robust parameters and recency weighting.
    model_sales = lgb.LGBMRegressor(**lgbm_params)
    model_sales.fit(
        train_df[FEATURES],
        train_df[TARGET_SALES],
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

        # Predict customers and sales
        pred_cust = model_cust.predict(X_pred)[0]
        # // SENIOR DEV RE-ENGINEERING NOTE //: Predict 'sales' using the new model.
        pred_sales = model_sales.predict(X_pred)[0]
        
        # Ensure predictions are non-negative
        pred_cust = max(0, pred_cust)
        pred_sales = max(0, pred_sales)

        # // SENIOR DEV RE-ENGINEERING NOTE //: Derive ATV from the predictions.
        # This prevents division by zero and ensures logical consistency.
        if pred_cust > 0:
            pred_atv = pred_sales / pred_cust
        else:
            pred_atv = 0
        
        # This new row will be added to the history for the next day's feature creation
        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'sales': pred_sales,
            'atv': pred_atv, # We add the derived ATV for feature consistency in the next loop
            'add_on_sales': 0, # Assume no add-on sales for future dates
            'day_type': 'Forecast'
        }
        future_predictions.append(new_row)
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 3. Finalize and Return ---
    
    final_forecast = pd.DataFrame(future_predictions)
    # The columns in the final dataframe are renamed for the app
    final_forecast.rename(columns={
        'date': 'ds',
        'customers': 'forecast_customers',
        'atv': 'forecast_atv',
        'sales': 'forecast_sales'
    }, inplace=True)
    
    # Final cleaning and formatting
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    # The app uses the customer model for feature importance, so we return it as before.
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
