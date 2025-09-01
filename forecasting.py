# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def generate_customer_forecast(historical_df, events_df, periods=15):
    """
    Generates a customer forecast using a LightGBM model with a recursive strategy and sample weighting.
    // SENIOR DEV NOTE //: This logic is restored from your original 'forecasting (4).py' file.
    It predicts one step at a time and uses that prediction as a feature for the next step.
    """
    # --- 1. Model Training ---
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'

    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'num_leaves': 31,
        'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    }
    
    start_weight = 0.2
    end_weight = 1.0
    sample_weights = np.linspace(start_weight, end_weight, len(train_df))

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST], sample_weight=sample_weights)

    # --- 2. Recursive Forecasting ---
    future_predictions_list = []
    history_for_recursion = historical_df.copy()
    last_date = history_for_recursion['date'].max()

    # Hold the last known ATV constant for generating features inside the loop
    last_known_atv = history_for_recursion['atv'].iloc[-1] if 'atv' in history_for_recursion and not history_for_recursion.empty else 0
    if last_known_atv == 0: # Fallback if last ATV is zero or missing
        atv_fallback = history_for_recursion[history_for_recursion['atv'] > 0]['atv'].mean()
        last_known_atv = atv_fallback if not pd.isna(atv_fallback) else 250 # Ultimate fallback

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([history_for_recursion, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        pred_cust = model_cust.predict(X_pred)[0]
        pred_cust = max(0, pred_cust)

        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'atv': last_known_atv, # Use placeholder ATV for feature generation
            'sales': pred_cust * last_known_atv, # Use placeholder sales
            'add_on_sales': 0,
        }
        future_predictions_list.append(new_row)
        history_for_recursion = pd.concat([history_for_recursion, pd.DataFrame([new_row])], ignore_index=True)

    # --- 3. Finalize and Return ---
    forecast_df = pd.DataFrame(future_predictions_list)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    forecast_df['forecast_customers'] = forecast_df['forecast_customers'].clip(lower=0).round().astype(int)
    
    return forecast_df[['ds', 'forecast_customers']], model_cust


def generate_atv_forecast(historical_df, events_df, periods=15):
    """
    Generates an ATV forecast using Prophet. This logic remains unchanged.
    """
    # --- 1. Data Preparation for Training ---
    prophet_df, regressor_names = prepare_data_for_prophet(historical_df, events_df)

    # --- START OF ROBUST FIX ---
    # 1. Surgically remove the renovation period from the training data.
    # This is a more robust solution than relying on NaN handling alone.
    renovation_start = pd.to_datetime('2024-08-20') # <-- CORRECTED DATE
    renovation_end = pd.to_datetime('2024-10-18')
    prophet_df_filtered = prophet_df[~prophet_df['ds'].between(renovation_start, renovation_end)].copy()
    
    # Keep NaN handling as a good practice for other potential zero-sales days
    prophet_df_filtered['y'] = prophet_df_filtered['y'].replace(0, np.nan)
    prophet_df_filtered.dropna(subset=['y'], inplace=True)


    # --- 2. Model Training with Stabilized Parameters ---
    # 2. Change seasonality_mode to 'additive' for better stability with data gaps.
    model_atv = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive', # CHANGED from 'multiplicative'
        growth='linear'
    )
    # --- END OF ROBUST FIX ---
    
    for regressor in regressor_names:
        model_atv.add_regressor(regressor, mode='additive') # Use additive mode for regressors as well

    # Fit the model on the FILTERED data
    model_atv.fit(prophet_df_filtered)

    # --- 3. Future Prediction ---
    future = model_atv.make_future_dataframe(periods=periods, freq='D')
    
    future['is_payday_period'] = future['ds'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    )
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)

    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique.rename(columns={'date': 'ds'}, inplace=True)
        events_df_unique['ds'] = pd.to_datetime(events_df_unique['ds']).dt.normalize()
        events_df_unique['is_event'] = 1
        
        future = pd.merge(future, events_df_unique[['ds', 'is_event']], on='ds', how='left').fillna(0)
        future['is_event'] = future['is_event'].astype(int)
    else:
        future['is_event'] = 0

    forecast = model_atv.predict(future)
    
    # --- 4. Finalize and Return ---
    forecast_final = forecast[['ds', 'yhat']].tail(periods)
    forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    forecast_final['forecast_atv'] = forecast_final['forecast_atv'].clip(lower=0)
    
    return forecast_final, model_atv
