# forecasting.py
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer
from datetime import timedelta
import os

# --- Constants ---
ENCODER_LENGTH = 60  # How many days of history the model sees to make a prediction
BATCH_SIZE = 32
MAX_EPOCHS = 30 # Increased for better accuracy, but still reasonable for initial training

def _create_ts_dataset(data, target, periods):
    """Helper function to create a TimeSeriesDataSet."""
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=target,
        group_ids=["group"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=periods,
        time_varying_known_reals=["dayofweek", "month", "day", "is_payday_period", "is_event"],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus")
    )

def _train_model(dataset, model_path):
    """Helper function to train and save a single N-BEATS model."""
    dataloader = dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    
    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto", # Uses GPU if available
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[],
    )
    
    # Define the N-BEATS model
    model = NBeats.from_dataset(
        dataset,
        learning_rate=3e-2,
        weight_decay=1e-2,
        loss=MAE(),
        backcast_loss_ratio=0.5,
    )
    
    # Train the model
    trainer.fit(model, train_dataloaders=dataloader)
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    return model

def generate_nbeats_forecast(historical_df, events_df, periods=15, force_retrain=False):
    """
    Generates a forecast using two specialized N-BEATS models.
    It will load pre-trained models if they exist, or train new ones.
    """
    # Define paths for saved models
    CUST_MODEL_PATH = "customer_model.pt"
    ATV_MODEL_PATH = "atv_model.pt"

    # --- 1. Data Preparation ---
    df_prepared = prepare_data_for_nbeats(historical_df, events_df)

    # --- 2. Train or Load Customer Model ---
    dataset_cust = _create_ts_dataset(df_prepared, 'customers', periods)
    if not os.path.exists(CUST_MODEL_PATH) or force_retrain:
        print("Training new customer model...")
        model_cust = _train_model(dataset_cust, CUST_MODEL_PATH)
    else:
        print("Loading existing customer model...")
        model_cust = NBeats.load_from_checkpoint(CUST_MODEL_PATH) # Incorrect way, need to load state dict
        # Correct loading:
        model_cust = NBeats.from_dataset(dataset_cust)
        model_cust.load_state_dict(torch.load(CUST_MODEL_PATH))


    # --- 3. Train or Load ATV Model ---
    dataset_atv = _create_ts_dataset(df_prepared, 'atv', periods)
    if not os.path.exists(ATV_MODEL_PATH) or force_retrain:
        print("Training new ATV model...")
        model_atv = _train_model(dataset_atv, ATV_MODEL_PATH)
    else:
        print("Loading existing ATV model...")
        model_atv = NBeats.from_dataset(dataset_atv)
        model_atv.load_state_dict(torch.load(ATV_MODEL_PATH))


    # --- 4. Generate Future Predictions ---
    # Create a future dataframe with known features for the forecast period
    last_date = df_prepared['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    future_df = pd.DataFrame({'date': future_dates})

    # Add the same known features to the future_df
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['is_payday_period'] = future_df['date'].apply(lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0)
    
    # Handle future events
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        future_df = pd.merge(future_df, events_df_unique[['date']], on='date', how='left', indicator='is_event_temp')
        future_df['is_event'] = (future_df['is_event_temp'] == 'both').astype(int)
        future_df.drop('is_event_temp', axis=1, inplace=True)
    else:
        future_df['is_event'] = 0

    # The model needs the last `encoder_length` of historical data to predict the future
    encoder_data = df_prepared[lambda x: x.time_idx > x.time_idx.max() - ENCODER_LENGTH]
    
    # Combine historical encoder data with future data for prediction
    prediction_data = pd.concat([encoder_data, future_df], ignore_index=True)
    
    # Generate predictions (direct, multi-step forecast)
    pred_cust = model_cust.predict(prediction_data).numpy().flatten()
    pred_atv = model_atv.predict(prediction_data).numpy().flatten()
    
    # --- 5. Finalize and Return ---
    final_forecast = pd.DataFrame({
        'ds': future_dates,
        'forecast_customers': pred_cust,
        'forecast_atv': pred_atv
    })
    final_forecast['forecast_sales'] = final_forecast['forecast_customers'] * final_forecast['forecast_atv']
    
    # Clip and round for realistic business values
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    return final_forecast
