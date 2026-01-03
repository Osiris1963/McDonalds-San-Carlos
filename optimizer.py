# optimizer.py
import pandas as pd
import numpy as np
from datetime import timedelta

class ForecastOptimizer:
    def __init__(self, historical_df, forecast_log_df):
        self.history = historical_df
        self.log = forecast_log_df
        self.bias_factor = 1.0
        self.error_metrics = {}

    def calculate_self_effectiveness(self):
        """Compares past predictions to actual results to find the Bias."""
        if self.log.empty or self.history.empty:
            return 1.0, {"status": "Insufficient data for self-correction"}

        # Merge past forecasts with actual results based on date
        comparison = pd.merge(
            self.log, 
            self.history[['date', 'customers', 'sales']], 
            left_on='forecast_for_date', 
            right_on='date'
        )

        if comparison.empty:
            return 1.0, {"status": "No overlapping dates found for backtesting"}

        # Calculate Mean Bias (Actual / Predicted)
        # If > 1, we are under-predicting. If < 1, we are over-predicting.
        comparison['customer_bias'] = comparison['customers'] / comparison['predicted_customers'].replace(0, np.nan)
        
        # We focus on the last 7 to 14 days for a 'human-like' recent memory
        recent_bias = comparison.tail(14)['customer_bias'].mean()
        
        # Calculate Accuracy (MAPE)
        mape = np.mean(np.abs((comparison['customers'] - comparison['predicted_customers']) / comparison['customers'])) * 100
        
        self.bias_factor = recent_bias if not pd.isna(recent_bias) else 1.0
        self.error_metrics = {
            "mape": mape,
            "accuracy": max(0, 100 - mape),
            "bias_adjustment": self.bias_factor,
            "sample_size": len(comparison)
        }
        
        return self.bias_factor, self.error_metrics

    def apply_smart_adjustment(self, forecast_df):
        """Applies the calculated bias to the future forecast."""
        if self.bias_factor == 1.0:
            return forecast_df

        adjusted_df = forecast_df.copy()
        
        # Apply the nudge
        # We use a dampened adjustment (0.5 weight) to prevent over-correction/oscillation
        dampened_bias = 1 + ((self.bias_factor - 1) * 0.5)
        
        if 'forecast_customers' in adjusted_df.columns:
            adjusted_df['forecast_customers'] = (adjusted_df['forecast_customers'] * dampened_bias).round().astype(int)
        
        if 'forecast_sales' in adjusted_df.columns:
            # Sales are recalculated based on adjusted customers
            adjusted_df['forecast_sales'] = adjusted_df['forecast_customers'] * adjusted_df['forecast_atv']
            
        return adjusted_df
