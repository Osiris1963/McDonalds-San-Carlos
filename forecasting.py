# forecasting.py
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a two-stage, unified LightGBM model
    with a recursive strategy for multi-step forecasting.
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

    # Define model parameters (a good starting point)
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    # Train the Customer Forecasting Model
    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST])

    # Train the ATV Forecasting Model
    model_atv = lgb.LGBMRegressor(**lgbm_params)
    model_atv.fit(train_df[FEATURES], train_df[TARGET_ATV])

    # --- 2. Recursive Forecasting ---
    
    future_predictions = []
    # Start with a full copy of historical data to generate features for future steps
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        # Create a placeholder for the day we want to predict
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        
        # Append placeholder to history to create features for it
        temp_df = pd.concat([history_df, future_placeholder], ignore_index=True)
        
        # Create features for this combined dataframe
        featured_for_pred = create_advanced_features(temp_df, events_df)
        
        # The last row contains the features for the day we need to predict
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        # Predict customers and ATV
        pred_cust = model_cust.predict(X_pred)[0]
        pred_atv = model_atv.predict(X_pred)[0]
        
        # Store the prediction
        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'atv': pred_atv,
            'sales': pred_cust * pred_atv,
            'add_on_sales': 0 # Assume 0 for future
        }
        future_predictions.append(new_row)
        
        # Append the completed prediction back to history_df.
        # This makes the prediction for day N available for the feature calculation of day N+1.
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
    
    # Clip and round for realistic business values
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    # Return the forecast and the customer model for feature importance plots
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
