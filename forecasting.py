# forecasting.py
# Implements the state-of-the-art Temporal Fusion Transformer model.
# Corrected to handle strict data type requirements for categorical features.

import pandas as pd
from datetime import timedelta
import numpy as np
from prophet import Prophet

# --- New Imports for TFT ---
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE

from data_processing import create_tft_features, create_atv_features, get_weather_data

def train_customer_model(historical_df, periods):
    """
    Trains a Temporal Fusion Transformer model to forecast customer counts.
    """
    # 1. Fetch weather data for the entire historical and future period
    start_date = historical_df['date'].min()
    end_date = historical_df['date'].max() + timedelta(days=periods)
    weather_df = get_weather_data(start_date, end_date)

    # 2. Create features for the TFT model
    df_featured = create_tft_features(historical_df, weather_df)
    
    # --- ROBUSTNESS FIX: Enforce correct data types for categorical features ---
    categorical_cols = ["series", "month", "dayofweek", "weather_code"]
    for col in categorical_cols:
        if col in df_featured.columns:
            df_featured[col] = df_featured[col].astype(str)
    # --- END FIX ---

    # 3. Define the TimeSeriesDataSet
    max_encoder_length = 30
    max_prediction_length = periods

    training_cutoff = df_featured["time_idx"].max() - max_prediction_length

    # Define the list of known categorical features for the model
    time_varying_known_categoricals = [col for col in ["month", "dayofweek", "weather_code"] if col in df_featured.columns]

    training_data = TimeSeriesDataSet(
        df_featured[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="customers",
        group_ids=["series"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["series"],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=["time_idx", "weather_temp", "weather_precip", "weather_wind"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["customers"],
        target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
        allow_missing_timesteps=True
    )

    # 4. Create validation set and dataloaders
    validation_data = TimeSeriesDataSet.from_dataset(training_data, df_featured, predict=True, stop_randomization=True)
    batch_size = 16
    train_dataloader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_data.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # 5. Configure and Train the TFT Model
    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=SMAPE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    trainer.fit(tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # 6. Predict on the future data
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    raw_predictions = best_tft.predict(val_dataloader, return_index=True, fast=True)
    
    # 7. Format the output
    all_preds = []
    for i in range(len(raw_predictions.index)):
        date = raw_predictions.index.iloc[i]['date']
        prediction = raw_predictions.prediction.iloc[i][3] 
        all_preds.append({'ds': date, 'forecast_customers': prediction})

    forecast_df = pd.DataFrame(all_preds)
    forecast_df = forecast_df.sort_values('ds').drop_duplicates('ds').reset_index(drop=True)

    return forecast_df


def train_atv_model(historical_df, events_df, periods):
    """Trains a Prophet model to forecast ATV. (Unchanged)"""
    df_atv = create_atv_features(historical_df)
    df_prophet = df_atv[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
    last_hist_date = historical_df['date'].max()
    future_end_date = last_hist_date + timedelta(days=periods)
    payday_dates = []
    current_date = historical_df['date'].min()
    while current_date <= future_end_date:
        if current_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_dates.append(current_date)
        current_date += timedelta(days=1)
    paydays = pd.DataFrame({'holiday': 'payday_period', 'ds': pd.to_datetime(payday_dates), 'lower_window': 0, 'upper_window': 1})
    holidays = paydays
    if events_df is not None and not events_df.empty:
        user_events = events_df[['date', 'activity_name']].copy()
        user_events.rename(columns={'date': 'ds', 'activity_name': 'holiday'}, inplace=True)
        holidays = pd.concat([paydays, user_events])
    model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast_df = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat']]
    forecast_df.rename(columns={'yhat': 'forecast_atv'}, inplace=True)
    return forecast_df, model

def generate_forecast(historical_df, events_df, periods=15):
    """Orchestrates the new TFT + Prophet forecasting process."""
    if len(historical_df) < 90:
        print("TFT model requires at least 90 days of data.")
        return pd.DataFrame(), None
    
    cust_forecast_df = train_customer_model(historical_df, periods)
    atv_forecast_df, prophet_model_for_ui = train_atv_model(historical_df, events_df, periods)
    
    if cust_forecast_df.empty or atv_forecast_df.empty:
        return pd.DataFrame(), None
        
    final_df = pd.merge(cust_forecast_df, atv_forecast_df, on='ds', how='left').sort_values('ds')
    final_df['forecast_atv'].fillna(method='ffill', inplace=True)
    
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round().astype(int)
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    
    return final_df.head(periods), prophet_model_for_ui
