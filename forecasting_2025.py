import pandas as pd
from darts import TimeSeries
from darts.models import PatchTST
from darts.dataprocessing.transformers import Scaler
import lightgbm as lgb
from data_processing import create_features

def forecast_customers(df, periods):
    ts = TimeSeries.from_dataframe(df, 'date', 'customers')
    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)
    model = PatchTST(input_chunk_length=30, output_chunk_length=periods, n_epochs=300)
    model.fit(ts_scaled)
    forecast = scaler.inverse_transform(model.predict(n=periods))
    return forecast.pd_dataframe().rename(columns={'customers': 'forecast_customers'})

def forecast_atv(df, periods):
    features = ['month', 'dayofweek', 'is_weekend', 'is_event', 'is_payday', 'payday_weekend_interaction']
    df = df.dropna(subset=['atv'])
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(df[features], df['atv'])
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=periods)
    future_df = pd.DataFrame({'date': future_dates})
    future_df = create_features(future_df, None)
    preds = model.predict(future_df[features])
    future_df['forecast_atv'] = preds
    return future_df[['date', 'forecast_atv']]

def generate_forecast_2025(historical_df, events_df, periods=15):
    df = create_features(historical_df, events_df)
    cust_forecast = forecast_customers(df, periods)
    atv_forecast = forecast_atv(df, periods)
    final_df = pd.merge(cust_forecast, atv_forecast, left_on='date', right_on='date')
    final_df['forecast_sales'] = final_df['forecast_customers'].round() * final_df['forecast_atv']
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    return final_df
