# forecasting.py (Updated snippets)
from optimizer import ForecastOptimizer
from data_processing import load_from_firestore

def run_optimized_forecast(db_client, historical_df, events_df, periods=15):
    """
    The orchestrator: Runs predictions, calculates effectiveness, 
    and applies self-correction logic.
    """
    # 1. Fetch past performance from your 'forecast_log'
    # Use your existing load function to get the logs
    forecast_log_df = load_from_firestore(db_client, 'forecast_log')
    if not forecast_log_df.empty:
        forecast_log_df['forecast_for_date'] = pd.to_datetime(forecast_log_df['forecast_for_date']).dt.normalize()

    # 2. Generate Base Forecasts (LGBM)
    cust_df, cust_model = generate_customer_forecast(historical_df, events_df, periods)
    
    # 3. Generate ATV Forecasts (Prophet)
    atv_df, _ = generate_atv_forecast(historical_df, events_df, periods)
    
    # 4. Combine
    final_df = pd.merge(cust_df, atv_df, on='ds')
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    # 5. SELF-LEARNING STEP (The 'Human' Brain)
    optimizer = ForecastOptimizer(historical_df, forecast_log_df)
    bias, metrics = optimizer.calculate_self_effectiveness()
    
    # Apply the correction nudge
    optimized_df = optimizer.apply_smart_adjustment(final_df)
    
    return optimized_df, metrics, cust_model
