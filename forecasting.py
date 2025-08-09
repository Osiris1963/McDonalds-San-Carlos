# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a two-stage, decoupled LightGBM model.
    
    // SENIOR DEV NOTE //: Re-engineered to use separate feature sets for customer 
    // and ATV models to prevent feature entanglement and improve ATV accuracy.
    """
    # --- 1. Data and Feature Preparation ---
    
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    # // SENIOR DEV NOTE //: Decoupling the feature sets.
    # The customer model uses a broad set of features as it was performing well.
    # The ATV model's features are now specialized, removing direct customer/sales volume
    # metrics to force it to focus purely on drivers of per-customer spending.
    
    # Features for the Customer Model (unchanged, as it works well)
    FEATURES_CUST = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    
    # Specialized features for the ATV Model
    FEATURES_ATV = [col for col in FEATURES_CUST if 'customers_' not in col and 'sales_' not in col]

    TARGET_CUST = 'customers'
    TARGET_ATV = 'atv'

    # --- 2. Model Training ---

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

    # Train the Customer Forecasting Model on its dedicated feature set
    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(
        train_df[FEATURES_CUST],
        train_df[TARGET_CUST],
        sample_weight=sample_weights
    )

    # Train the ATV Forecasting Model on its specialized, smaller feature set
    model_atv = lgb.LGBMRegressor(**lgbm_params)
    model_atv.fit(
        train_df[FEATURES_ATV],
        train_df[TARGET_ATV],
        sample_weight=sample_weights
    )

    # --- 3. Recursive Forecasting ---
    
    future_predictions = []
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([history_df, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        
        # The last row contains all possible features for the day we need to predict
        X_pred_full = featured_for_pred.iloc[-1:]

        # // SENIOR DEV NOTE //: Predict using the correct, decoupled feature sets for each model.
        pred_cust = model_cust.predict(X_pred_full[FEATURES_CUST])[0]
        pred_atv = model_atv.predict(X_pred_full[FEATURES_ATV])[0]
        
        # Ensure predictions are non-negative
        pred_cust = max(0, pred_cust)
        pred_atv = max(0, pred_atv)

        new_row = {
            'date': current_pred_date,
            'customers': pred_cust,
            'atv': pred_atv,
            'sales': pred_cust * pred_atv,
            'add_on_sales': 0,
            'day_type': 'Forecast'
        }
        future_predictions.append(new_row)
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 4. Finalize and Return ---
    
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
    
    # We return the customer model for insights as it's the primary driver
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
