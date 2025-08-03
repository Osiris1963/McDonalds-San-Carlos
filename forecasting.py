# forecasting.py (Hybrid Prophet-XGBoost with Optuna Tuning)
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import optuna
from datetime import timedelta
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Suppress Optuna's trial logs and other warnings
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    if len(df_day_featured) < 30: # Need more data for this advanced model
        return pd.DataFrame(), None

    future_date_range = pd.date_range(start=df_day_featured['date'].max() + timedelta(days=1), periods=periods * 7)
    future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
    if len(future_dates) == 0:
        return pd.DataFrame(), None

    # --- STRATEGY 1: CONSERVATIVE ATV FORECAST (Prophet Only) ---
    if target == 'atv':
        df_prophet = df_day_featured[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_prophet)
        future_df_prophet = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df_prophet)
        return forecast[['ds', 'yhat']], model

    # --- STRATEGY 2: HYBRID PROPHET-XGBOOST FOR CUSTOMERS ---
    elif target == 'customers':
        # 1. Train Prophet to get the baseline and components
        df_prophet = df_day_featured[['date', 'customers']].rename(columns={'date': 'ds', 'customers': 'y'})
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(df_prophet)
        
        # 2. Merge Prophet components back into the main dataframe
        df_forecast_on_hist = prophet_model.predict(df_prophet[['ds']])
        df_day_featured = pd.merge(df_day_featured, df_forecast_on_hist[['ds', 'yhat', 'trend', 'yearly']], left_on='date', right_on='ds', how='left')
        
        # 3. Calculate the residual (Prophet's error) - this is our new target for XGBoost
        df_day_featured['residual'] = df_day_featured['customers'] - df_day_featured['yhat']
        
        # 4. Define features for XGBoost
        features = [col for col in df_day_featured.columns if df_day_featured[col].dtype in ['int64', 'float64', 'int32'] and col not in ['sales', 'customers', 'atv', 'date', 'yhat', 'residual', 'ds']]
        df_train = df_day_featured.dropna(subset=features + ['residual'])
        X = df_train[features]
        y = df_train['residual']
        
        # 5. Tune XGBoost with Optuna
        def objective(trial):
            params = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0, 0.1),
                'lambda': trial.suggest_float('lambda', 0.1, 1.0),
                'alpha': trial.suggest_float('alpha', 0.1, 1.0),
                'seed': 42, 'n_jobs': -1
            }
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        
        # 6. Train final XGBoost model on all data with the best parameters
        final_xgb = xgb.XGBRegressor(objective='reg:squarederror', seed=42, n_jobs=-1, **best_params)
        final_xgb.fit(X, y)
        
        # 7. Make the final forecast
        future_df_prophet = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
        
        future_df_xgb = pd.DataFrame({'date': future_dates})
        future_df_xgb = pd.merge(future_df_xgb, future_df_prophet[['ds', 'trend', 'yearly']], left_on='date', right_on='ds')
        
        combined_for_features = pd.concat([df_day_featured, future_df_xgb], ignore_index=True)
        combined_with_features = create_features(combined_for_features, events_df)
        X_future = combined_with_features[combined_with_features['date'].isin(future_dates)][features]
        X_future = X_future.reindex(columns=X.columns, fill_value=0)

        future_residual_pred = final_xgb.predict(X_future)
        
        final_prediction = future_df_prophet['yhat'].values + future_residual_pred
        
        final_preds_df = pd.DataFrame({'ds': future_dates, 'yhat': final_prediction})
        return final_preds_df, final_xgb
    
    return pd.DataFrame(), None


def generate_forecast(historical_df, events_df, periods=15):
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        
        cust_fcst, _ = run_day_specific_models(df_day, 'customers', day_of_week, periods, events_df)
        if not cust_fcst.empty: all_cust_forecasts.append(cust_fcst)
            
        atv_fcst, _ = run_day_specific_models(df_day, 'atv', day_of_week, periods, events_df)
        if not atv_fcst.empty: all_atv_forecasts.append(atv_fcst)

    if not all_cust_forecasts or not all_atv_forecasts:
        return pd.DataFrame(), None 

    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds').reset_index(drop=True)
    
    if cust_forecast_final.empty or atv_forecast_final.empty:
        return pd.DataFrame(), None

    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')

    if final_df.empty: return pd.DataFrame(), None
        
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)

    return final_df.head(periods), None
