import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_customer_forecast(historical_df, events_df, periods=15):
    """
    Advanced Self-Correcting Customer Forecast.
    Calculates historical bias and adjusts future predictions automatically.
    """
    df_featured = create_advanced_features(historical_df, events_df)
    # Use only rows where we have the target and rolling features
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14', 'customers']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET = 'customers'

    # --- Step 1: Dynamic Weighting (Newer data matters more) ---
    # We apply an exponential weight so the model 'learns' more from recent behavior
    weights = np.exp(np.linspace(-2, 0, len(train_df))) 

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=1500,
        learning_rate=0.03, # Slower learning for better generalization
        num_leaves=50,
        feature_fraction=0.9,
        lambda_l1=0.2,
        importance_type='gain'
    )
    
    model.fit(train_df[FEATURES], train_df[TARGET], sample_weight=weights)

    # --- Step 2: Self-Evaluation (The 'Human' check) ---
    # Forecast the last 7 days of known data to see how 'off' we are
    backtest_days = min(7, len(train_df))
    test_features = train_df[FEATURES].tail(backtest_days)
    test_actuals = train_df[TARGET].tail(backtest_days)
    test_preds = model.predict(test_features)
    
    # Calculate Mean Bias: (Actual / Predicted)
    # If > 1, model is underestimating. If < 1, model is overestimating.
    bias_factor = (test_actuals.sum() / test_preds.sum())
    # Dampen the bias factor to prevent over-correction (range 0.9 to 1.1)
    bias_factor = max(0.9, min(1.1, bias_factor))

    # --- Step 3: Recursive Forecasting with Correction ---
    future_predictions_list = []
    history_for_recursion = historical_df.copy()
    last_date = history_for_recursion['date'].max()
    last_known_atv = history_for_recursion['atv'].iloc[-1] if not history_for_recursion.empty else 250

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        temp_df = pd.concat([history_for_recursion, pd.DataFrame([{'date': current_pred_date}])], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[FEATURES].iloc[-1:]

        raw_pred = model.predict(X_pred)[0]
        # Apply the intelligence: Multiply by bias_factor
        corrected_pred = max(0, raw_pred * bias_factor)

        new_row = {
            'date': current_pred_date,
            'customers': corrected_pred,
            'atv': last_known_atv,
            'sales': corrected_pred * last_known_atv
        }
        future_predictions_list.append(new_row)
        history_for_recursion = pd.concat([history_for_recursion, pd.DataFrame([new_row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_predictions_list)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    
    # Return bias factor for UI display
    metrics = {"bias_factor": bias_factor}
    return forecast_df[['ds', 'forecast_customers']], model, metrics
