# forecasting.py
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from datetime import timedelta

# --- Import from our data processing module ---
from data_processing import create_features

def generate_ph_holidays(start_date, end_date, events_df):
    """Generates a DataFrame of PH holidays for Prophet, including user-provided future activities."""
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

    payday_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [15, 30]:
            payday_events.append({'holiday': 'Payday', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

def run_base_models(df_featured, target, periods, events_df):
    """Trains Prophet, LightGBM, XGBoost and returns their forecasts."""
    base_forecasts = {}
    
    FEATURES = [
        'month', 'dayofyear', 'weekofyear', 'year', 'dayofweek_num',
        'is_payday_period', 'is_event', 'is_not_normal_day',
        'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 
        'day_Thursday', 'day_Tuesday', 'day_Wednesday',
        'sales_lag_7', 'sales_lag_14', 'sales_rolling_mean_7', 'sales_rolling_std_7',
        'customers_lag_7', 'customers_lag_14', 'customers_rolling_mean_7', 'customers_rolling_std_7',
        'atv_lag_7', 'atv_lag_14', 'atv_rolling_mean_7', 'atv_rolling_std_7'
    ]
    valid_features = [f for f in FEATURES if f in df_featured.columns]

    df_train = df_featured.dropna(subset=valid_features + [target])
    X = df_train[valid_features]
    y = df_train[target]
    
    # --- Prophet Model ---
    last_date = df_featured['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    df_prophet = df_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    base_forecasts['prophet'] = forecast[['ds', 'yhat']].tail(periods)
    
    # --- ROBUST FIX: Build the future DataFrame for tree models from scratch ---
    X_future = pd.DataFrame(index=future_dates)
    
    # 1. Add date features
    X_future['month'] = X_future.index.month
    X_future['dayofyear'] = X_future.index.dayofyear
    X_future['weekofyear'] = X_future.index.isocalendar().week.astype(int)
    X_future['year'] = X_future.index.year
    X_future['dayofweek_num'] = X_future.index.dayofweek
    
    # 2. Add one-hot encoded day of week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_name in day_names:
        X_future[f'day_{day_name}'] = (X_future.index.day_name() == day_name).astype(int)

    # 3. Add event and payday features
    X_future['is_payday_period'] = [1 if d.day in [14,15,16,29,30,31,1,2] else 0 for d in X_future.index]
    if events_df is not None:
        future_event_dates = pd.to_datetime(events_df['date']).dt.date
        X_future['is_event'] = X_future.index.to_series().dt.date.isin(future_event_dates).astype(int)
    else:
        X_future['is_event'] = 0
    X_future['is_not_normal_day'] = 0 # Assume future days are normal unless specified

    # 4. Add lag and rolling features by broadcasting the last known value
    for col in valid_features:
        if 'lag' in col or 'rolling' in col:
            # Use the most recent value from the historical data
            X_future[col] = df_featured[col].iloc[-1]
            
    # Ensure column order matches the training set
    X_future = X_future[valid_features]
    # --- END OF FIX ---

    # --- LightGBM Model ---
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X, y)
    lgbm_preds = lgbm.predict(X_future)
    base_forecasts['lgbm'] = pd.DataFrame({'ds': future_dates, 'yhat': lgbm_preds})

    # --- XGBoost Model ---
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict(X_future)
    base_forecasts['xgb'] = pd.DataFrame({'ds': future_dates, 'yhat': xgb_preds})

    return base_forecasts, prophet_model

def train_stacked_ensemble(base_forecasts):
    """Combines base model forecasts using a weighted average."""
    final_df = None
    for name, fcst_df in base_forecasts.items():
        renamed_df = fcst_df.rename(columns={'yhat': f'yhat_{name}'})
        if final_df is None:
            final_df = renamed_df
        else:
            final_df = pd.merge(final_df, renamed_df, on='ds', how='outer')
    
    weights = {'prophet': 0.3, 'lgbm': 0.4, 'xgb': 0.3}
    final_df['yhat'] = (
        final_df['yhat_prophet'] * weights['prophet'] +
        final_df['yhat_lgbm'] * weights['lgbm'] +
        final_df['yhat_xgb'] * weights['xgb']
    )
    return final_df[['ds', 'yhat']]

def generate_forecast(historical_df, events_df, periods=15):
    """Main forecasting function using a hierarchical approach and ensemble models."""
    df_featured = create_features(historical_df, events_df)

    atv_base_forecasts, _ = run_base_models(df_featured, 'atv', periods, events_df)
    atv_forecast = train_stacked_ensemble(atv_base_forecasts)
    atv_forecast.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    cust_base_forecasts, prophet_model_cust = run_base_models(df_featured, 'customers', periods, events_df)
    cust_forecast = train_stacked_ensemble(cust_base_forecasts)
    cust_forecast.rename(columns={'yhat': 'forecast_customers'}, inplace=True)

    final_df = pd.merge(cust_forecast, atv_forecast, on='ds')
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    return final_df, prophet_model_cust
