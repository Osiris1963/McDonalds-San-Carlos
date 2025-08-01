# forecasting.py (Re-architected with Day-Specific Models)
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from datetime import timedelta
import warnings

# Suppress verbose warnings from modeling libraries
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Import from our data processing module ---
from data_processing import create_features

def generate_ph_holidays(start_date, end_date, events_df):
    """Generates a DataFrame of PH holidays for Prophet, including user-provided future activities."""
    # Official Public Holidays for 2025 in the Philippines
    holidays_list = [
        {'holiday': 'New Year\'s Day', 'ds': '2025-01-01'},
        {'holiday': 'Maundy Thursday', 'ds': '2025-04-17'},
        {'holiday': 'Good Friday', 'ds': '2025-04-18'},
        {'holiday': 'Araw ng Kagitingan', 'ds': '2025-04-09'},
        {'holiday': 'Labor Day', 'ds': '2025-05-01'},
        {'holiday': 'Independence Day', 'ds': '2025-06-12'},
        {'holiday': 'Ninoy Aquino Day', 'ds': '2025-08-21'},
        {'holiday': 'National Heroes Day', 'ds': '2025-08-25'},
        {'holiday': 'All Saints\' Day', 'ds': '2025-11-01'},
        {'holiday': 'Bonifacio Day', 'ds': '2025-11-30'},
        {'holiday': 'Feast of the Immaculate Conception', 'ds': '2025-12-08'},
        {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'},
    ]
    ph_holidays = pd.DataFrame(holidays_list)
    ph_holidays['ds'] = pd.to_datetime(ph_holidays['ds'])

    # Generate payday windows, which are a powerful local factor
    payday_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    # Add custom user-defined events
    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

def run_day_specific_models(df_day_featured, target, day_of_week, periods, events_df):
    """
    Trains Prophet, LightGBM, XGBoost on data for only a *specific day of the week*
    and returns their combined forecast for that day.
    """
    if len(df_day_featured) < 20: # Need at least ~20 weeks of data for a stable model
        return pd.DataFrame(), None

    # --- Prepare Data ---
    FEATURES = [col for col in df_day_featured.columns if df_day_featured[col].dtype in ['int64', 'float64', 'int32'] and col != target]
    
    # Remove features that are constant for a specific day (e.g., 'day_Monday' is always 1 in the Monday model)
    constant_cols = [col for col in FEATURES if df_day_featured[col].nunique() == 1]
    FEATURES = [f for f in FEATURES if f not in constant_cols]
    
    df_train = df_day_featured.dropna(subset=FEATURES + [target])
    X = df_train[FEATURES]
    y = df_train[target]
    
    # --- Future DataFrame Logic ---
    last_date = df_day_featured['date'].max()
    # Generate enough dates to capture the desired number of specific weekdays
    future_date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7) 
    future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
    
    if len(future_dates) == 0:
        return pd.DataFrame(), None

    # --- Prophet Model (Day-Specific) ---
    df_prophet = df_day_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    # Weekly seasonality is false because we are already modeling a specific day
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(df_prophet)
    future_prophet_df = pd.DataFrame({'ds': future_dates})
    forecast_prophet = prophet_model.predict(future_prophet_df)
    
    # --- Tree-Based Models (LGBM, XGB) ---
    # To create future features correctly, we need to append future placeholders to the historical data
    # and then engineer features on the combined dataframe.
    future_placeholder = pd.DataFrame(index=future_dates)
    future_placeholder.index.name = 'date'
    
    # Combine historical data for the specific day with future placeholders
    combined_df_for_features = pd.concat([df_day_featured, future_placeholder])
    combined_df_for_features = create_features(combined_df_for_features, events_df)
    
    X_future = combined_df_for_features[combined_df_for_features.index.isin(future_dates)]
    X_future = X_future[FEATURES]
    X_future.fillna(method='ffill', inplace=True) # Fill any gaps with the last known value
    X_future.fillna(0, inplace=True) # Fill any remaining NaNs

    # --- LightGBM Model ---
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X, y)
    lgbm_preds = lgbm.predict(X_future)

    # --- XGBoost Model ---
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict(X_future)
    
    # --- Ensemble Forecast for the specific day ---
    weights = {'prophet': 0.3, 'lgbm': 0.4, 'xgb': 0.3}
    final_preds = (
        forecast_prophet['yhat'].values * weights['prophet'] +
        lgbm_preds * weights['lgbm'] +
        xgb_preds * weights['xgb']
    )
    
    day_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_preds})
    
    return day_forecast_df, prophet_model

def generate_forecast(historical_df, events_df, periods=15):
    """
    Main forecasting function. Implements the "Team of Specialists" approach.
    It builds seven independent models, one for each day of the week.
    """
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts = []
    all_atv_forecasts = []
    
    # --- Main Orchestration Loop ---
    # Iterate through each day of the week (0=Monday, 6=Sunday)
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        
        # Forecast Customers for this day
        cust_fcst, prophet_model = run_day_specific_models(df_day, 'customers', day_of_week, periods, events_df)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            
        # Forecast ATV for this day
        atv_fcst, _ = run_day_specific_models(df_day, 'atv', day_of_week, periods, events_df)
        if not atv_fcst.empty:
            all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts:
        return pd.DataFrame(), None # Return empty if no forecasts could be made

    # --- Combine forecasts from all 7 specialist models ---
    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds').reset_index(drop=True)
    
    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    # Combine customer and ATV forecasts to get the final sales forecast
    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds')
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    # Post-processing to ensure logical values (no negative sales/customers)
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    # The 'prophet_model' returned is just the one for the last day (Sunday), mainly for debugging/insight purposes.
    # The true forecast is the composite `final_df`.
    return final_df, prophet_model
