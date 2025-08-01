# forecasting.py
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from datetime import timedelta
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Import from our data processing module ---
from data_processing import create_features

# --- Import the specific EarlyStopping callback for XGBoost ---
from xgboost.callback import EarlyStopping as XGBEarlyStopping

# Suppress Optuna's trial logging to keep the console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def tune_model_hyperparameters(X, y, model_name='lgbm'):
    """Uses Optuna to find the best hyperparameters for a given model."""
    
    def objective(trial):
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        if model_name == 'lgbm':
            params = {
                'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            
            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(10, verbose=False)])
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)

        else: # xgb
            params = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42, 'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)

            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                # --- FINAL FIX: Use the explicit callback object for XGBoost ---
                early_stopping_callback = XGBEarlyStopping(rounds=10, save_best=True)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[early_stopping_callback], verbose=False)
                # --- END OF FIX ---

                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)
        
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, show_progress_bar=False) # Reduced trials for faster UI
    return study.best_params

def run_base_models(df_featured, target, periods, events_df):
    """Trains tuned Prophet, LightGBM, XGBoost and returns their forecasts."""
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
    
    last_date = df_featured['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    
    # --- Prophet Model ---
    df_prophet = df_featured[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    base_forecasts['prophet'] = forecast[['ds', 'yhat']].tail(periods)
    
    # --- Future DataFrame for Tree Models ---
    X_future = pd.DataFrame(index=future_dates)
    X_future['month'] = X_future.index.month
    X_future['dayofyear'] = X_future.index.dayofyear
    X_future['weekofyear'] = X_future.index.isocalendar().week.astype(int)
    X_future['year'] = X_future.index.year
    X_future['dayofweek_num'] = X_future.index.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_name in day_names:
        X_future[f'day_{day_name}'] = (X_future.index.day_name() == day_name).astype(int)
    X_future['is_payday_period'] = [1 if d.day in [14,15,16,29,30,31,1,2] else 0 for d in X_future.index]
    if events_df is not None and not events_df.empty:
        future_event_dates = pd.to_datetime(events_df['date']).dt.date
        X_future['is_event'] = X_future.index.to_series().dt.date.isin(future_event_dates).astype(int)
    else:
        X_future['is_event'] = 0
    X_future['is_not_normal_day'] = 0
    for col in valid_features:
        if 'lag' in col or 'rolling' in col:
            X_future[col] = df_featured[col].iloc[-1]
    X_future = X_future[valid_features]

    # --- Tuned LightGBM Model ---
    lgbm_best_params = tune_model_hyperparameters(X, y, 'lgbm')
    lgbm = lgb.LGBMRegressor(random_state=42, **lgbm_best_params)
    lgbm.fit(X, y)
    lgbm_preds = lgbm.predict(X_future)
    base_forecasts['lgbm'] = pd.DataFrame({'ds': future_dates, 'yhat': lgbm_preds})

    # --- Tuned XGBoost Model ---
    xgb_best_params = tune_model_hyperparameters(X, y, 'xgb')
    xgb_model = xgb.XGBRegressor(random_state=42, **xgb_best_params)
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict(X_future)
    base_forecasts['xgb'] = pd.DataFrame({'ds': future_dates, 'yhat': xgb_preds})

    return base_forecasts, prophet_model

def train_stacked_ensemble(base_forecasts):
    """Combines base model forecasts using a Linear Regression meta-learner."""
    final_df = None
    for name, fcst_df in base_forecasts.items():
        renamed_df = fcst_df.rename(columns={'yhat': f'yhat_{name}'})
        if final_df is None:
            final_df = renamed_df
        else:
            final_df = pd.merge(final_df, renamed_df, on='ds', how='outer')
    
    X_meta = final_df[[col for col in final_df.columns if 'yhat' in col]]
    y_meta = X_meta.mean(axis=1)

    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)
    
    final_df['yhat'] = meta_model.predict(X_meta)
    return final_df[['ds', 'yhat']]

def generate_forecast(historical_df, events_df, periods=15):
    """Main forecasting function using a hierarchical approach and tuned ensemble models."""
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
