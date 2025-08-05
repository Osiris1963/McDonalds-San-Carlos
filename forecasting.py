# forecasting.py
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer
from datetime import timedelta
import os

from data_processing import prepare_data_for_nbeats

# --- Constants ---
ENCODER_LENGTH = 60  # How many days of history the model sees to make a prediction
BATCH_SIZE = 32
MAX_EPOCHS = 30 

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
    """
    Helper function to train and save a single N-BEATS model.
    This version uses the explicit NBeats constructor for robustness.
    """
    train_dataloader = dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[],
    )
    
    # --- THIS IS THE FIX ---
    # We replace the "smart" from_dataset helper with the explicit constructor.
    # This avoids potential assertion errors from internal heuristics on small datasets.
    model = NBeats(
        context_length=ENCODER_LENGTH,
        prediction_length=dataset.max_prediction_length,
        loss=MAE(),
        # Default architecture parameters for stability
        widths=[32, 512],
        backcast_loss_ratio=0.5
    )
    # ----------------------
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
    )
    
    torch.save(model.state_dict(), model_path)
    return model

def generate_nbeats_forecast(historical_df, events_df, periods=15, force_retrain=False):
    """
    Generates a forecast using two specialized N-BEATS models.
    It will load pre-trained models if they exist, or train new ones.
    """
    CUST_MODEL_PATH = "customer_model.pt"
    ATV_MODEL_PATH = "atv_model.pt"

    df_prepared = prepare_data_for_nbeats(historical_df, events_df)

    # --- Train or Load Customer Model ---
    dataset_cust = _create_ts_dataset(df_prepared, 'customers', periods)
    if not os.path.exists(CUST_MODEL_PATH) or force_retrain:
        print("Training new customer model...")
        model_cust = _train_model(dataset_cust, CUST_MODEL_PATH)
    else:
        print("Loading existing customer model...")
        # To load a model, we must first initialize it with the same architecture
        model_cust = NBeats(
            context_length=ENCODER_LENGTH,
            prediction_length=dataset_cust.max_prediction_length,
            loss=MAE(),
            widths=[32, 512],
            backcast_loss_ratio=0.5
        )
        model_cust.load_state_dict(torch.load(CUST_MODEL_PATH))


    # --- Train or Load ATV Model ---
    dataset_atv = _create_ts_dataset(df_prepared, 'atv', periods)
    if not os.path.exists(ATV_MODEL_PATH) or force_retrain:
        print("Training new ATV model...")
        model_atv = _train_model(dataset_atv, ATV_MODEL_PATH)
    else:
        print("Loading existing ATV model...")
        model_atv = NBeats(
            context_length=ENCODER_LENGTH,
            prediction_length=dataset_atv.max_prediction_length,
            loss=MAE(),
            widths=[32, 512],
            backcast_loss_ratio=0.5
        )
        model_atv.load_state_dict(torch.load(ATV_MODEL_PATH))

    # --- Generate Future Predictions ---
    # The model needs the last `encoder_length` of historical data to predict the future
    encoder_data = df_prepared[lambda x: x.time_idx > x.time_idx.max() - ENCODER_LENGTH]
    
    # Generate predictions from the models
    pred_cust_raw, _ = model_cust.predict(encoder_data, return_x=True)
    pred_atv_raw, _ = model_atv.predict(encoder_data, return_x=True)
    
    pred_cust = pred_cust_raw.numpy().flatten()
    pred_atv = pred_atv_raw.numpy().flatten()
    
    # --- Finalize and Return ---
    last_date = df_prepared['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]

    final_forecast = pd.DataFrame({
        'ds': future_dates,
        'forecast_customers': pred_cust,
        'forecast_atv': pred_atv
    })
    final_forecast['forecast_sales'] = final_forecast['forecast_customers'] * final_forecast['forecast_atv']
    
    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)
    
    return final_forecast
