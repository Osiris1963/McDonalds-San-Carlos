# forecasting.py

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, MultiLoss
from pytorch_forecasting.metrics import MAE, SMAPE
from data_processing import prepare_data_for_tft

def generate_forecast(historical_df, events_df, periods=15):
    """
    Orchestrates the entire deep learning forecast process.
    """
    # 1. Prepare data in the required format
    # The dataset object is the single source of truth for the model
    dataset = prepare_data_for_tft(historical_df, events_df, periods_to_forecast=periods)
    
    # 2. Create dataloaders for training
    dataloader = dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    # 3. Configure and train the model
    # We use a PyTorch Lightning Trainer for robust training
    trainer = pl.Trainer(
        max_epochs=25,  # Increased epochs for deep learning
        accelerator="auto", # Automatically uses GPU if available
        gradient_clip_val=0.1,
        limit_train_batches=50,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False, # Disable in Streamlit to avoid clutter
    )

    # Define the TFT model architecture
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        # As requested, we use a simple loss for point forecasts (not probabilistic)
        loss=MultiLoss([MAE(), MAE()]),
        optimizer="AdamW"
    )

    # Train the model
    trainer.fit(tft, train_dataloaders=dataloader)

    # 4. Generate predictions
    # We need the last `encoder_length` of data to predict the future
    encoder_data = dataset.filter(lambda x: x.time_idx > x.time_idx.max() - tft.hparams.max_encoder_length)
    
    # Create a dataloader for the prediction input
    pred_dataloader = encoder_data.to_dataloader(train=False, batch_size=1, num_workers=0)

    # Get raw predictions
    raw_predictions = tft.predict(pred_dataloader, mode="prediction", return_index=True)
    
    # The model predicts both 'customers' and 'atv'
    pred_customers = raw_predictions.output[0][:, 0].numpy()
    pred_atv = raw_predictions.output[0][:, 1].numpy()

    # Get the dates for the forecast
    forecast_dates = [
        historical_df['date'].max() + pd.DateOffset(days=x) 
        for x in range(1, periods + 1)
    ]

    # Assemble the final forecast DataFrame
    final_df = pd.DataFrame({
        'ds': forecast_dates,
        'forecast_customers': pred_customers,
        'forecast_atv': pred_atv,
    })
    
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    # Clip values to ensure they are non-negative
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    
    # Return both the forecast and the model for later inspection
    return final_df, tft, dataset
