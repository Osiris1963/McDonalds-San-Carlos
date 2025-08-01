# forecasting.py
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
from datetime import timedelta

# --- MISSING IMPORT ADDED HERE ---
from data_processing import create_features

def generate_ph_holidays(start_date, end_date):
    """Generates a DataFrame of PH holidays for Prophet."""
    # In a real app, use a library like 'holidays' or a maintained list
    # For now, this is a placeholder. Add major holidays.
    holidays_list = [
        {'holiday': 'New Year', 'ds': '2025-01-01'},
        {'holiday': 'Good Friday', 'ds': '2025-04-18'},
        {'holiday': 'Labor Day', 'ds': '2025-05-01'},
        {'holiday': 'Independence Day', 'ds': '2025-06-12'},
        {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'},
    ]
    ph_holidays = pd.DataFrame(holidays_list)
    ph_holidays['ds'] = pd.to_datetime(ph_holidays['ds'])

    # Add Payday events for prophet
    payday_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [15, 30]:
            payday_events.append({'holiday': 'Payday', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    
    # Ensure no duplicate dates in holidays
    all_events = pd.concat([ph_holidays, pd.DataFrame(payday_events)]).drop_duplicates(subset=['ds'])
    return all_events

def generate_forecast(historical_df, events_df, periods=15):
    """
    Main forecasting function. Trains a Prophet and LightGBM model,
    then creates a weighted ensemble forecast.
    """
    # --- 1. Prepare Data ---
    df_featured = create_features(historical_df, events_df)

    # Define the target and the features for the tree model
    TARGETS = ['sales', 'customers']
    FEATURES = [
        'month', 'dayofyear', 'weekofyear', 'year',
        'is_payday_period', 'is_event', 'is_not_normal_day',
        'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday',
        'sales_lag_7', 'sales_lag_15', 'customers_lag_7',
        'sales_rolling_mean_7', 'sales_rolling_std_7'
    ]
    FEATURES = [f for f in FEATURES if f in df_featured.columns] # Ensure all features exist
    
    final_forecasts = {}

    # --- 2. Train and Forecast with LightGBM ---
    for target in TARGETS:
        df_train = df_featured.dropna(subset=FEATURES + [target])
        X = df_train[FEATURES]
        y = df_train[target]

        lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=31, verbosity=-1)
        lgbm.fit(X, y)

        # Create future dataframe for prediction
        last_date = df_featured['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        # Create a temporary DataFrame for future feature generation
        future_df_template = pd.DataFrame({'date': future_dates})
        
        # We need to create features for the future dates iteratively.
        # This is a simplified approach for a live app.
        temp_history = pd.concat([df_featured, future_df_template], ignore_index=True)
        future_with_features = create_features(temp_history, events_df)
        
        X_future = future_with_features[future_with_features['date'].isin(future_dates)][FEATURES].copy()
        
        # Handle NaNs in future features (e.g., lags) by back-filling then forward-filling
        X_future.fillna(method='bfill', inplace=True)
        X_future.fillna(method='ffill', inplace=True) # Fill any remaining
        
        predictions = lgbm.predict(X_future)
        
        lgbm_forecast = pd.DataFrame({'ds': future_dates, 'yhat_lgbm': predictions})
        final_forecasts[f'lgbm_{target}'] = lgbm_forecast

    # --- 3. Train and Forecast with Prophet ---
    for target in TARGETS:
        df_prophet = historical_df[['date', target]].rename(columns={'date': 'ds', target: 'y'})
        
        start_date = df_prophet['ds'].min()
        end_date = df_prophet['ds'].max() + timedelta(days=periods)
        
        prophet_holidays = generate_ph_holidays(start_date, end_date)
        
        model = Prophet(
            holidays=prophet_holidays,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False, # Daily seasonality is usually noise for daily sales
            changepoint_prior_scale=0.1, # Increased flexibility for changepoints
            holidays_prior_scale=15.0  # Give more weight to holidays
        )
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        prophet_forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'yhat_prophet'})
        final_forecasts[f'prophet_{target}'] = prophet_forecast.tail(periods)

    # --- 4. Create Weighted Ensemble ---
    sales_forecast = pd.merge(final_forecasts['lgbm_sales'], final_forecasts['prophet_sales'], on='ds')
    cust_forecast = pd.merge(final_forecasts['lgbm_customers'], final_forecasts['prophet_customers'], on='ds')

    # Define weights. Start with 60/40. Can be tuned later.
    LGBM_WEIGHT = 0.6
    PROPHET_WEIGHT = 0.4
    
    sales_forecast['forecast_sales'] = (sales_forecast['yhat_lgbm'] * LGBM_WEIGHT) + (sales_forecast['yhat_prophet'] * PROPHET_WEIGHT)
    cust_forecast['forecast_customers'] = (cust_forecast['yhat_lgbm'] * LGBM_WEIGHT) + (cust_forecast['yhat_prophet'] * PROPHET_WEIGHT)

    # --- 5. Combine and Finalize ---
    final_df = pd.merge(sales_forecast[['ds', 'forecast_sales']], cust_forecast[['ds', 'forecast_customers']], on='ds')
    final_df['forecast_atv'] = (final_df['forecast_sales'] / final_df['forecast_customers']).fillna(0)
    
    # Ensure forecasts are non-negative
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()

    return final_df
