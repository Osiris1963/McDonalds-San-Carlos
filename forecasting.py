# forecasting.py (Re-architected for Temporal Fusion Transformer)
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from data_processing import prepare_data_for_tft

def generate_tft_forecast(historical_df, events_df, periods=15):
    """
    Generates a multi-target forecast using the Temporal Fusion Transformer model.
    """
    if len(historical_df) < 60: # TFT needs a reasonable amount of history
        return pd.DataFrame(), None, None

    # --- 1. Prepare Data ---
    full_df = prepare_data_for_tft(historical_df, events_df)

    # --- 2. Define TimeSeriesDataSet ---
    # TFT requires a validation set that is right after the training set.
    max_date = full_df['date'].max()
    training_cutoff_date = max_date - timedelta(days=periods)
    training_cutoff_idx = full_df[full_df['date'] == training_cutoff_date]['time_idx'].iloc[0]

    # Define the dataset
    training_dataset = TimeSeriesDataSet(
        full_df[lambda x: x.time_idx <= training_cutoff_idx],
        time_idx="time_idx",
        target=["sales", "customers", "atv"],
        group_ids=["group_id"],
        max_encoder_length=45,  # How many days of history to use for prediction
        max_prediction_length=periods,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["month", "dayofweek", "weekofyear"],
        time_varying_known_reals=["time_idx", "day", "dayofyear", "year", "is_payday_period", "is_weekend", "is_event", "is_not_normal_day"],
        time_varying_unknown_reals=["sales", "customers", "atv"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # --- 3. Create Model ---
    # Create validation and full dataloaders
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, full_df, predict=True, stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
    validation_dataloader = validation_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)
    
    # Define the TFT model with QuantileLoss for probabilistic forecasting
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # Number of quantiles to predict
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # --- 4. Train Model ---
    # Using PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu", # or "gpu" if available
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    trainer.fit(tft, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader)

    # --- 5. Generate Forecast ---
    # Predict on the validation set to get future forecasts
    raw_predictions, x = tft.predict(validation_dataloader, mode="raw", return_x=True)

    # Format the output
    dates = full_df['date'][training_cutoff_idx + 1 : training_cutoff_idx + 1 + periods]
    
    forecast_df = pd.DataFrame({'ds': dates})
    
    # Extract quantiles for each target
    targets = ["sales", "customers", "atv"]
    for i, target in enumerate(targets):
        # Quantiles are at indices 0, 3, 6 (p10, p50, p90)
        forecast_df[f'forecast_{target}_p10'] = raw_predictions['prediction'][:, :, 0][:, i].numpy()
        forecast_df[f'forecast_{target}_p50'] = raw_predictions['prediction'][:, :, 3][:, i].numpy()
        forecast_df[f'forecast_{target}_p90'] = raw_predictions['prediction'][:, :, 6][:, i].numpy()

    # Clip predictions to be non-negative
    for col in forecast_df.columns:
        if 'forecast' in col:
            forecast_df[col] = forecast_df[col].clip(lower=0)
            if 'customers' in col:
                forecast_df[col] = forecast_df[col].round()

    return forecast_df, tft, x, raw_predictions
