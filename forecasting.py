# forecasting.py (Corrected with modern scikit-learn Pipeline)
import pandas as pd
from prophet import Prophet
from datetime import timedelta
import warnings
import numpy as np

# --- NEW: Import Pipeline and StandardScaler for the modern approach ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=FutureWarning)

from data_processing import create_features

def generate_ph_holidays(start_date, end_date, events_df):
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

    payday_events = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_events.append({'holiday': 'Payday Window', 'ds': current_date, 'lower_window': 0, 'upper_window': 1})
        current_date += timedelta(days=1)
    
    all_holidays = pd.concat([ph_holidays, pd.DataFrame(payday_events)])

    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        user_events['ds'] = pd.to_datetime(user_events['ds'])
        all_holidays = pd.concat([all_holidays, user_events])

    return all_holidays.drop_duplicates(subset=['ds']).reset_index(drop=True)

def run_day_specific_models(df_day_featured, target, day_of_week, periods, events_df):
    """
    Runs a specific forecasting model based on the target variable.
    - For 'customers', it uses the "Analog Week" model with a Scaler + Ridge pipeline.
    - For 'atv', it uses a conservative Prophet-only model.
    """
    if len(df_day_featured) < 20: 
        return pd.DataFrame(), None

    future_date_range = pd.date_range(start=df_day_featured['date'].max() + timedelta(days=1), periods=periods * 7) 
    future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
    if len(future_dates) == 0:
        return pd.DataFrame(), None

    model = None 

    # --- STRATEGY 1: CONSERVATIVE ATV FORECAST ---
    if target == 'atv':
        df_prophet = df_day_featured[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_prophet)
        
        future_df_prophet = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df_prophet)
        final_preds_df = forecast[['ds', 'yhat']]

    # --- STRATEGY 2: "ANALOG WEEK" CUSTOMER FORECAST ---
    elif target == 'customers':
        features = [
            'customers_lag_7', 'customers_rolling_mean_7',
            'is_payday_period', 'is_event'
        ]
        
        df_train = df_day_featured.dropna(subset=features + [target]).copy()

        if len(df_train) < 10: 
            return pd.DataFrame(), None
            
        X_train = df_train[features]
        y_train = df_train[target]
        
        # --- FIX: Use a Pipeline to combine scaling and regression ---
        # This replaces the old `Ridge(normalize=True)`
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        model.fit(X_train, y_train)
        
        future_placeholder = pd.DataFrame({'date': future_dates})
        combined_df_for_features = pd.concat([df_day_featured, future_placeholder], ignore_index=True)
        combined_df_with_features = create_features(combined_df_for_features, events_df)
    
        X_future = combined_df_with_features[combined_df_with_features['date'].isin(future_dates)][features]
        
        predictions = model.predict(X_future)
        final_preds_df = pd.DataFrame({'ds': future_dates, 'yhat': predictions})
        
    else:
        return pd.DataFrame(), None

    return final_preds_df, model


def generate_forecast(historical_df, events_df, periods=15):
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        
        cust_fcst, _ = run_day_specific_models(df_day, 'customers', day_of_week, periods, events_df)
        if not cust_fcst.empty:
            all_cust_forecasts.append(cust_fcst)
            
        atv_fcst, _ = run_day_specific_models(df_day, 'atv', day_of_week, periods, events_df)
        if not atv_fcst.empty:
            all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts:
        return pd.DataFrame(), None 

    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds').reset_index(drop=True)
    
    if cust_forecast_final.empty or atv_forecast_final.empty:
        return pd.DataFrame(), None

    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')
    if final_df.empty:
        return pd.DataFrame(), None
        
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    return final_df.head(periods), None
