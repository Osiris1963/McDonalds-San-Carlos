# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a hybrid engine:
    - Customers: Retains the robust Direct Multi-Step LightGBM strategy.
    - Base Sales: Uses Facebook's Prophet model to handle time-series decomposition
      and external regressors, providing a different and potentially more stable forecast.
    """
    # --- 1. Data and Feature Preparation ---
    df_featured = create_advanced_features(historical_df, events_df)
    
    # --- 2. Customer Model Training (UNCHANGED) ---
    # We keep the superior Direct Multi-Step strategy for the customer forecast.
    
    FEATURES_LGBM = [col for col in df_featured.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes', 'base_sales'
    ]]
    models_cust = {}
    lgbm_params_cust = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 500,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }

    for h in range(1, periods + 1):
        train_df_lgbm = df_featured.copy()
        train_df_lgbm['target_cust'] = train_df_lgbm['customers'].shift(-h)
        train_df_lgbm.dropna(subset=['target_cust'], inplace=True)
        
        model_cust = lgb.LGBMRegressor(**lgbm_params_cust)
        model_cust.fit(train_df_lgbm[FEATURES_LGBM], train_df_lgbm['target_cust'])
        models_cust[h] = model_cust

    # --- 3. Base Sales Model Training (NEW: PROPHET) ---
    
    # // SENIOR DEV NOTE //: We now build and train a single Prophet model for base_sales.
    # Prophet requires specific column names: 'ds' for date and 'y' for the target value.
    prophet_train_df = df_featured[['date', 'base_sales']].rename(columns={'date': 'ds', 'base_sales': 'y'})
    
    # Define which of our features can be used as "extra regressors" in Prophet.
    # These must be known for the future. Lag/rolling features cannot be used here.
    PROPHET_REGRESSORS = [
        'is_payday_period', 'is_weekend', 'payday_weekend_interaction',
        'dayofyear_sin', 'dayofyear_cos', 'weekofyear_sin', 'weekofyear_cos',
        'is_event', 'is_not_normal_day'
    ]
    
    # Add the regressor columns to the training data.
    for regressor in PROPHET_REGRESSORS:
        if regressor in df_featured.columns:
            prophet_train_df[regressor] = df_featured[regressor]

    # Initialize and configure the Prophet model
    model_base_sales_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative', # Sales often have multiplicative seasonality
        growth='linear'
    )
    
    # Add each regressor to the model
    for regressor in PROPHET_REGRESSORS:
        if regressor in prophet_train_df.columns:
            model_base_sales_prophet.add_regressor(regressor, mode='additive')

    # Fit the Prophet model on the entire history
    model_base_sales_prophet.fit(prophet_train_df.dropna())


    # --- 4. Forecasting ---
    
    # First, get all customer predictions from our LightGBM models
    lgbm_pred_features = df_featured[FEATURES_LGBM].iloc[-1:]
    customer_predictions = [models_cust[h].predict(lgbm_pred_features)[0] for h in range(1, periods + 1)]

    # Next, get all base_sales predictions from our Prophet model
    # Create the 'future' dataframe that Prophet needs for forecasting
    last_date = historical_df['date'].max()
    future_dates = [last_date + timedelta(days=h) for h in range(1, periods + 1)]
    future_df = pd.DataFrame({'ds': future_dates})

    # We must generate the regressor features for these future dates
    future_df_featured = create_advanced_features(pd.DataFrame({'date': future_dates}), events_df)
    for regressor in PROPHET_REGRESSORS:
         if regressor in future_df_featured.columns:
            future_df[regressor] = future_df_featured[regressor]

    # Predict future base_sales
    sales_forecast_prophet = model_base_sales_prophet.predict(future_df)
    base_sales_predictions = sales_forecast_prophet['yhat'].tolist()
    
    # --- 5. Combine and Finalize ---
    
    future_predictions = []
    avg_addon_per_customer = historical_df['add_on_sales'].sum() / historical_df['customers'].replace(0, 1).sum()

    for i in range(periods):
        pred_cust = max(0, customer_predictions[i])
        pred_base_sales = max(0, base_sales_predictions[i])
        
        pred_atv = (pred_base_sales / pred_cust) if pred_cust > 0 else 0
        
        estimated_add_on_sales = pred_cust * avg_addon_per_customer
        pred_total_sales = pred_base_sales + estimated_add_on_sales
        
        new_row = {
            'ds': last_date + timedelta(days=i + 1),
            'forecast_customers': pred_cust,
            'forecast_sales': pred_total_sales,
            'forecast_atv': pred_atv,
        }
        future_predictions.append(new_row)
        
    final_forecast = pd.DataFrame(future_predictions)
    
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    # We return the first customer model for feature importance, as it remains a key part of the system.
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], models_cust.get(1)
