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
    
    # --- ROBUST FIX: Explicitly define the list of features for the models ---
    # This prevents text-based columns from being included by mistake.
    FEATURES = [
        'month', 'dayofyear', 'weekofyear', 'year', 'dayofweek_num',
        'is_payday_period', 'is_event', 'is_not_normal_day',
        'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 
        'day_Thursday', 'day_Tuesday', 'day_Wednesday',
        'sales_lag_7', 'sales_lag_14', 'sales_rolling_mean_7', 'sales_rolling_std_7',
        'customers_lag_7', 'customers_lag_14', 'customers_rolling_mean_7', 'customers_rolling_std_7',
        'atv_lag_7', 'atv_lag_14', 'atv_rolling_mean_7', 'atv_rolling_std_7'
    ]
    # Ensure we only use features that actually exist in the dataframe
    valid_features = [f for f in FEATURES if f in df_featured.columns]
    # --- END OF FIX ---

    df_train = df_featured.dropna(subset=valid_features + [target])
    X = df_train[valid_features]
    y = df_train[target]
    
    # --- Future DataFrame ---
    last_date = df_featured['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    future_df_template = pd.DataFrame({'date': future_dates})
    temp_history = pd.concat([df_featured, future_df_template], ignore_index=True)
    future_with_features = create_features(temp_history, events_df)
    X_future = future_with_features[future_with_features['date'].isin(future_dates)][valid_features].copy()

    # --- Prophet Model ---
    df_prophet = df_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    base_forecasts['prophet'] = forecast[['ds', 'yhat']].tail(periods)
    
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
