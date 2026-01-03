# optimizer.py
import pandas as pd
import numpy as np

def get_self_learning_bias(historical_df, forecast_log_df):
    """
    Independent auditor that calculates how much the AI missed.
    Uses 'Strict Coercion' to fix the Merge Error once and for all.
    """
    if forecast_log_df is None or forecast_log_df.empty or historical_df.empty:
        return 1.0

    try:
        # FORCE everything to strings first, then back to datetime to strip hidden metadata
        hist = historical_df.copy()
        hist['merge_date'] = pd.to_datetime(hist['date'].astype(str)).dt.strftime('%Y-%m-%d')

        logs = forecast_log_df.copy()
        # Find the date column regardless of what it was named in Firestore
        log_date_col = 'forecast_for_date' if 'forecast_for_date' in logs.columns else 'ds'
        logs['merge_date'] = pd.to_datetime(logs[log_date_col].astype(str)).dt.strftime('%Y-%m-%d')

        # Merge on strictly formatted string dates to avoid the ValueError
        comparison = pd.merge(
            logs[['merge_date', 'predicted_customers']], 
            hist[['merge_date', 'customers']], 
            on='merge_date',
            how='inner'
        )

        if comparison.empty:
            return 1.0

        # Calculate performance of the last 7 entries
        comparison = comparison.tail(7)
        # Avoid division by zero
        valid_comparison = comparison[comparison['predicted_customers'] > 0]
        
        if valid_comparison.empty:
            return 1.0

        # BIAS = (What actually happened / What AI predicted)
        actual_performance_ratio = (valid_comparison['customers'] / valid_comparison['predicted_customers']).mean()
        
        # Human-like limit: Don't let the AI swing more than 20% in one go
        return np.clip(actual_performance_ratio, 0.8, 1.2)
        
    except Exception as e:
        print(f"Auditor Error: {e}")
        return 1.0
