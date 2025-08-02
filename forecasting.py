# forecasting.py (Final Version with Residual Stacking)
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from datetime import timedelta
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from data_processing import create_features

def run_primary_model(df_train, X_train, y_train, future_dates, final_features, events_df):
    """Trains the primary stacked ensemble and returns models and future predictions."""
    # --- 1. Train Base Models ---
    df_prophet = df_train[['date']].rename(columns={'date': 'ds'})
    df_prophet['y'] = y_train
    prophet_holidays = generate_ph_holidays(df_prophet['ds'].min(), future_dates.max(), events_df)
    prophet_model = Prophet(holidays=prophet_holidays, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(df_prophet)

    lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train)

    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'seed': 42, 'n_jobs': -1}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)

    # --- 2. Train Meta-Learner ---
    prophet_train_fcst = prophet_model.predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = lgbm.predict(X_train)
    xgb_train_preds = xgb_model.predict(X_train)
    X_meta = pd.DataFrame({'prophet': prophet_train_fcst['yhat'].values, 'lgbm': lgbm_train_preds, 'xgb': xgb_train_preds})
    meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 10))
    meta_learner.fit(X_meta, y_train)

    # --- 3. Generate Primary Forecast for the Future ---
    future_placeholder = pd.DataFrame({'date': future_dates})
    combined_df_for_features = pd.concat([df_train, future_placeholder], ignore_index=True)
    combined_df_with_features = create_features(combined_df_for_features, events_df)
    X_future = combined_df_with_features[combined_df_with_features['date'].isin(future_dates)][final_features]
    X_future.fillna(method='ffill', inplace=True); X_future.fillna(0, inplace=True)

    prophet_future_fcst = prophet_model.predict(pd.DataFrame({'ds': future_dates}))
    lgbm_future_preds = lgbm.predict(X_future)
    xgb_future_preds = xgb_model.predict(X_future)
    X_meta_future = pd.DataFrame({'prophet': prophet_future_fcst['yhat'].values, 'lgbm': lgbm_future_preds, 'xgb': xgb_future_preds})
    
    primary_forecast = meta_learner.predict(X_meta_future)
    
    models = {'prophet': prophet_model, 'lgbm': lgbm, 'xgb': xgb_model, 'meta_learner': meta_learner}
    return primary_forecast, models

def run_residual_model(df_train, y_train, X_train, models, final_features, future_dates, events_df):
    """Trains a model to predict the errors of the primary model."""
    # --- 1. Calculate Historical Errors (Residuals) ---
    prophet_train_fcst = models['prophet'].predict(df_train[['date']].rename(columns={'date':'ds'}))
    lgbm_train_preds = models['lgbm'].predict(X_train)
    xgb_train_preds = models['xgb'].predict(X_train)
    X_meta_train = pd.DataFrame({'prophet': prophet_train_fcst['yhat'].values, 'lgbm': lgbm_train_preds, 'xgb': xgb_train_preds})
    primary_train_preds = models['meta_learner'].predict(X_meta_train)
    
    residuals = y_train - primary_train_preds

    # --- 2. Train a Model on the Errors ---
    # Use a simpler, faster model for residuals
    residual_lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 20, 'verbose': -1, 'n_jobs': -1, 'seed': 123}
    residual_model = lgb.LGBMRegressor(**residual_lgbm_params)
    residual_model.fit(X_train, residuals)

    # --- 3. Predict the Future Errors ---
    future_placeholder = pd.DataFrame({'date': future_dates})
    combined_df_for_features = pd.concat([df_train, future_placeholder], ignore_index=True)
    combined_df_with_features = create_features(combined_df_for_features, events_df)
    X_future = combined_df_with_features[combined_df_with_features['date'].isin(future_dates)][final_features]
    X_future.fillna(method='ffill', inplace=True); X_future.fillna(0, inplace=True)

    predicted_residuals = residual_model.predict(X_future)
    return predicted_residuals

def generate_forecast(historical_df, events_df, periods=15):
    """Main forecasting function using a two-stage Residual Stacking architecture."""
    df_featured = create_features(historical_df, events_df)
    
    all_cust_forecasts, all_atv_forecasts = [], []
    prophet_model_for_insights = None 
    
    for day_of_week in range(7):
        df_day = df_featured[df_featured['date'].dt.dayofweek == day_of_week].copy()
        if len(df_day) < 30: continue # Need more data for stable residual modeling

        all_features = [col for col in df_day.columns if df_day[col].dtype in ['int64', 'float64', 'int32'] and col not in ['total_sales', 'base_sales', 'customers', 'atv', 'date', 'add_on_sales']]
        constant_cols = [col for col in all_features if df_day[col].nunique() < 2]
        final_features = [f for f in all_features if f not in constant_cols]
        
        last_date = df_day['date'].max()
        future_date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods * 7) 
        future_dates = future_date_range[future_date_range.dayofweek == day_of_week][:periods]
        if len(future_dates) == 0: continue

        # --- Process Customers ---
        target_cust = 'customers'
        df_train_cust = df_day.dropna(subset=final_features + [target_cust])
        if len(df_train_cust) < 20: continue
        X_train_cust, y_train_cust = df_train_cust[final_features], df_train_cust[target_cust]
        
        primary_cust_fcst, cust_models = run_primary_model(df_train_cust, X_train_cust, y_train_cust, future_dates, final_features, events_df)
        predicted_cust_residuals = run_residual_model(df_train_cust, y_train_cust, X_train_cust, cust_models, final_features, future_dates, events_df)
        final_cust_fcst = primary_cust_fcst + predicted_cust_residuals
        all_cust_forecasts.append(pd.DataFrame({'ds': future_dates, 'yhat': final_cust_fcst}))
        
        if cust_models.get('prophet'): prophet_model_for_insights = cust_models['prophet']

        # --- Process ATV ---
        target_atv = 'atv'
        df_train_atv = df_day.dropna(subset=final_features + [target_atv])
        if len(df_train_atv) < 20: continue
        X_train_atv, y_train_atv = df_train_atv[final_features], df_train_atv[target_atv]

        primary_atv_fcst, atv_models = run_primary_model(df_train_atv, X_train_atv, y_train_atv, future_dates, final_features, events_df)
        predicted_atv_residuals = run_residual_model(df_train_atv, y_train_atv, X_train_atv, atv_models, final_features, future_dates, events_df)
        final_atv_fcst = primary_atv_fcst + predicted_atv_residuals
        all_atv_forecasts.append(pd.DataFrame({'ds': future_dates, 'yhat': final_atv_fcst}))

    if not all_cust_forecasts or not all_atv_forecasts: return pd.DataFrame(), None 

    # --- Combine and Finalize ---
    cust_forecast_final = pd.concat(all_cust_forecasts).sort_values('ds').reset_index(drop=True)
    atv_forecast_final = pd.concat(all_atv_forecasts).sort_values('ds').reset_index(drop=True)
    
    if cust_forecast_final.empty or atv_forecast_final.empty: return pd.DataFrame(), None

    cust_forecast_final.rename(columns={'yhat': 'forecast_customers'}, inplace=True)
    atv_forecast_final.rename(columns={'yhat': 'forecast_atv'}, inplace=True)

    final_df = pd.merge(cust_forecast_final, atv_forecast_final, on='ds', how='inner')
    if final_df.empty: return pd.DataFrame(), None
        
    final_df['forecast_base_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    # --- Add back historical average of add-on sales ---
    historical_df['dayofweek'] = historical_df['date'].dt.dayofweek
    avg_addons_by_day = historical_df.groupby('dayofweek')['add_on_sales'].mean().reset_index()
    final_df['dayofweek'] = final_df['ds'].dt.dayofweek
    final_df = pd.merge(final_df, avg_addons_by_day, on='dayofweek', how='left').fillna(0)
    
    final_df['forecast_sales'] = final_df['forecast_base_sales'] + final_df['add_on_sales']
    
    # Clip to logical values
    for col in ['forecast_sales', 'forecast_customers', 'forecast_atv']:
        final_df[col] = final_df[col].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].round()

    return final_df.head(periods), prophet_model_for_insights
