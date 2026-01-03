# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet

def generate_customer_forecast(historical_df, events_df, periods=15):
    """Direct Multi-Step Forecasting: 15 models for 15 days."""
    full_featured = create_advanced_features(historical_df, events_df)
    
    FEATURES = ['month', 'dayofweek', 'is_payday', 'is_weekend', 'is_event', 
                'cust_sdly', 'cust_roll_7', 'cust_roll_14', 'cust_roll_30']
    
    last_date = historical_df['date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
    future_preds = []

    # Train a specific expert model for each day in the 15-day outlook
    for h in range(1, periods + 1):
        train_df = full_featured.copy()
        train_df['target'] = train_df['customers'].shift(-h)
        train_data = train_df.dropna(subset=['target', 'cust_sdly'])
        
        # Poisson objective handles high-traffic spikes better than linear regression
        model = lgb.LGBMRegressor(objective='poisson', n_estimators=300, learning_rate=0.03, verbose=-1)
        model.fit(train_data[FEATURES], train_data['target'])
        
        # Predict the specific horizon
        current_feat = create_advanced_features(pd.concat([historical_df, pd.DataFrame({'date': [last_date + timedelta(days=h)]})]), events_df)
        pred = model.predict(current_feat[FEATURES].tail(1))[0]
        future_preds.append(max(0, pred))

    forecast_df = pd.DataFrame({'ds': future_dates, 'forecast_customers': np.round(future_preds).astype(int)})
    return forecast_df, None

def generate_atv_forecast(historical_df, events_df, periods=15):
    """Prophet Ensemble for ATV to capture yearly inflation trends."""
    df_p, regressors = prepare_data_for_prophet(historical_df, events_df)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
    for reg in regressors: model.add_regressor(reg)
    model.fit(df_p)
    
    future = model.make_future_dataframe(periods=periods)
    future['is_payday'] = future['ds'].dt.day.isin([14,15,16,29,30,31,1,2]).astype(int)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    
    forecast = model.predict(future)
    atv_res = forecast[['ds', 'yhat']].tail(periods).rename(columns={'yhat': 'forecast_atv'})
    return atv_res, model
