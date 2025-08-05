# forecasting.py
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_features

def generate_direct_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a direct multi-model strategy.
    This trains a separate model for each day in the forecast horizon.
    """
    # --- 1. Feature Engineering ---
    # Note: events_df is not used in this model but is kept for API consistency
    df_featured = create_features(historical_df)
    
    # --- 2. Target Engineering ---
    # Create target columns for each day we want to predict
    for i in range(1, periods + 1):
        df_featured[f'cust_target_d{i}'] = df_featured['customers'].shift(-i)
        df_featured[f'atv_target_d{i}'] = df_featured['atv'].shift(-i)
    
    # Drop rows where targets are NaN (the last `periods` rows) and initial rows with NaN from feature creation
    df_train = df_featured.dropna()
    
    FEATURES = [col for col in df_train.columns if col not in ['date', 'doc_id', 'add_on_sales', 'sales', 'customers', 'atv'] and 'target' not in col]
    
    # --- 3. Train Models ---
    models_cust = {}
    models_atv = {}

    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 500,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }

    print("Training robust multi-model engine...")
    for i in range(1, periods + 1):
        target_cust = f'cust_target_d{i}'
        target_atv = f'atv_target_d{i}'
        
        # Train customer model for day i
        model_c = lgb.LGBMRegressor(**lgbm_params)
        model_c.fit(df_train[FEATURES], df_train[target_cust])
        models_cust[i] = model_c
        
        # Train ATV model for day i
        model_a = lgb.LGBMRegressor(**lgbm_params)
        model_a.fit(df_train[FEATURES], df_train[target_atv])
        models_atv[i] = model_a

    # --- 4. Generate Forecast ---
    # Get the very last row of featured data, which has the most recent info
    last_known_data = create_features(historical_df.tail(60)).iloc[-1:][FEATURES]
    
    future_predictions = []
    last_date = historical_df['date'].max()

    for i in range(1, periods + 1):
        pred_cust = models_cust[i].predict(last_known_data)[0]
        pred_atv = models_atv[i].predict(last_known_data)[0]
        
        new_row = {
            'ds': last_date + timedelta(days=i),
            'forecast_customers': pred_cust,
            'forecast_atv': pred_atv,
            'forecast_sales': pred_cust * pred_atv
        }
        future_predictions.append(new_row)

    # --- 5. Finalize ---
    final_forecast = pd.DataFrame(future_predictions)
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    return final_forecast
