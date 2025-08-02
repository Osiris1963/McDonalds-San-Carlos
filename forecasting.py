# forecasting.py
# Implements the two-model forecasting architecture: LightGBM for customers and Prophet for ATV.

import pandas as pd
from datetime import timedelta
import lightgbm as lgb
from prophet import Prophet

# Import our specialized feature engineering functions
from data_processing import create_customer_features, create_atv_features

def train_customer_model(historical_df, periods):
    """
    Trains a LightGBM model on weekly-pattern features to forecast customer counts.
    
    Args:
        historical_df (pd.DataFrame): The dataframe with historical sales data.
        periods (int): The number of future days to forecast.

    Returns:
        pd.DataFrame: A dataframe with future dates and predicted customer counts.
    """
    # 1. Create specialized features for the customer model
    df_featured = create_customer_features(historical_df)
    
    # 2. Define features and target
    features = [
        'dayofweek', 'dayofyear', 'month', 'weekofyear', 'year', 'is_weekend',
        'customers_lag_7', 'customers_lag_14',
        'customers_rolling_mean_4_weeks_same_day', 'customers_rolling_std_4_weeks_same_day'
    ]
    target = 'customers'
    
    # Drop rows where target is unknown (i.e., the future)
    df_train = df_featured.dropna(subset=[target])
    
    # 3. Train the LightGBM Model
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', # L1 is robust to outliers
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        seed=42
    )
    lgbm.fit(df_train[features], df_train[target],
             eval_set=[(df_train[features], df_train[target])],
             eval_metric='rmse',
             callbacks=[lgb.early_stopping(100, verbose=False)])

    # 4. Create future dataframe and predict
    last_date = historical_df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    future_df = pd.DataFrame({'date': future_dates})
    
    # Use the historical data to create a base for future feature generation
    full_df = pd.concat([historical_df, future_df], ignore_index=True)
    full_df_featured = create_customer_features(full_df)
    
    future_features = full_df_featured[full_df_featured['date'].isin(future_dates)]
    
    predictions = lgbm.predict(future_features[features])
    
    # 5. Format and return the forecast
    forecast_df = pd.DataFrame({'ds': future_dates, 'forecast_customers': predictions})
    return forecast_df


def train_atv_model(historical_df, events_df, periods):
    """
    Trains a Prophet model to forecast Average Transaction Value (ATV).
    
    Args:
        historical_df (pd.DataFrame): The dataframe with historical sales data.
        events_df (pd.DataFrame): Dataframe with future events/activities.
        periods (int): The number of future days to forecast.

    Returns:
        pd.DataFrame: A dataframe with future dates and predicted ATV.
        Prophet model object: For plotting components in the UI.
    """
    # 1. Prepare data for Prophet
    df_atv = create_atv_features(historical_df)
    df_prophet = df_atv[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
    
    # 2. Define holidays and special events (Payday)
    last_hist_date = historical_df['date'].max()
    future_end_date = last_hist_date + timedelta(days=periods)
    
    payday_dates = []
    current_date = historical_df['date'].min()
    while current_date <= future_end_date:
        # Payday has a strong effect on spending power
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_dates.append(current_date)
        current_date += timedelta(days=1)
        
    paydays = pd.DataFrame({
        'holiday': 'payday_period',
        'ds': pd.to_datetime(payday_dates),
        'lower_window': 0,
        'upper_window': 1, # Effect lasts for the day and the next day
    })

    # Combine with user-provided events if any
    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        holidays = pd.concat([paydays, user_events])
    else:
        holidays = paydays

    # 3. Train the Prophet Model
    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True, # Critical for respecting weekly ATV patterns
        daily_seasonality=False
    )
    model.fit(df_prophet)
    
    # 4. Predict future ATV
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # 5. Format and return the forecast
    forecast_df = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat']]
    forecast_df.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    
    return forecast_df, model


def generate_forecast(historical_df, events_df, periods=15):
    """
    Orchestrates the two-model forecasting process.
    
    Returns:
        pd.DataFrame: The final combined forecast.
        Prophet model object: To be passed to the UI for visualization.
    """
    if len(historical_df) < 30:
        print("Error: Not enough historical data. Need at least 30 days.")
        return pd.DataFrame(), None
        
    # --- Run the two separate model pipelines ---
    cust_forecast_df = train_customer_model(historical_df, periods)
    atv_forecast_df, prophet_model_for_ui = train_atv_model(historical_df, events_df, periods)
    
    # --- Combine the results ---
    if cust_forecast_df.empty or atv_forecast_df.empty:
        return pd.DataFrame(), None

    final_df = pd.merge(cust_forecast_df, atv_forecast_df, on='ds')
    
    # --- Final Calculations and Cleanup ---
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round().astype(int)
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)

    return final_df.head(periods), prophet_model_for_ui
