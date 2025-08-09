# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a HYBRID strategy:
    1. Recursive model for the stable 'customers' target.
    2. Direct Multi-Step (DMS) models for the volatile 'atv' target.
    
    // SENIOR DEV NOTE //: This architecture is a definitive solution to the ATV
    // problem. It completely eliminates compounding error for ATV predictions while
    // preserving the excellent performance of the customer forecast.
    """
    # --- 1. Data and Feature Preparation ---
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)

    # Decoupled feature sets remain crucial.
    FEATURES_CUST = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    FEATURES_ATV = [col for col in FEATURES_CUST if 'customers_' not in col and 'sales_' not in col]

    TARGET_CUST = 'customers'
    TARGET_ATV = 'atv'

    # --- 2. Model Training ---
    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    sample_weights = np.linspace(0.2, 1.0, len(train_df))

    # --- MODEL 1: RECURSIVE CUSTOMER FORECASTING (The existing, excellent model) ---
    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(
        train_df[FEATURES_CUST],
        train_df[TARGET_CUST],
        sample_weight=sample_weights
    )

    # --- MODEL 2: DIRECT MULTI-STEP ATV FORECASTING (The new, robust architecture) ---
    # We train a separate model for each day in the forecast horizon.
    atv_models = {}
    for i in range(1, periods + 1):
        # The target is the ATV value 'i' days in the future.
        y = train_df[TARGET_ATV].shift(-i)
        
        # We need to remove the last 'i' rows where the target is NaN.
        X_train_atv = train_df[FEATURES_ATV].iloc[:-i]
        y_train_atv = y.iloc[:-i]
        weights_atv = sample_weights[:-i]

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train_atv, y_train_atv, sample_weight=weights_atv)
        atv_models[i] = model
        
    # --- 3. Hybrid Forecasting Execution ---
    future_predictions = []
    
    # Part A: Generate all customer predictions recursively first.
    cust_history_df = historical_df.copy()
    customer_preds = []
    for i in range(periods):
        current_pred_date = cust_history_df['date'].max() + timedelta(days=1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([cust_history_df, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred_cust = featured_for_pred[FEATURES_CUST].iloc[-1:]

        pred_cust = max(0, model_cust.predict(X_pred_cust)[0])
        customer_preds.append(pred_cust)
        
        # Append the completed prediction back to the customer history
        new_row = {'date': current_pred_date, 'customers': pred_cust, 'sales': 0, 'atv': 0}
        cust_history_df = pd.concat([cust_history_df, pd.DataFrame([new_row])], ignore_index=True)

    # Part B: Generate all ATV predictions directly, using the specialist models.
    # We use the features from the VERY LAST REAL day of data for all ATV forecasts.
    last_real_features = create_advanced_features(historical_df, events_df)[FEATURES_ATV].iloc[-1:]
    atv_preds = []
    for i in range(1, periods + 1):
        model = atv_models[i]
        pred_atv = max(0, model.predict(last_real_features)[0])
        atv_preds.append(pred_atv)

    # Part C: Combine the results.
    last_date = historical_df['date'].max()
    for i in range(periods):
        pred_date = last_date + timedelta(days=i + 1)
        pred_cust = customer_preds[i]
        pred_atv = atv_preds[i]
        
        new_row = {
            'date': pred_date,
            'customers': pred_cust,
            'atv': pred_atv,
            'sales': pred_cust * pred_atv,
        }
        future_predictions.append(new_row)

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
