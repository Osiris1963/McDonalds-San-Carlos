# forecasting.py
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from datetime import timedelta
from data_processing import prepare_data_for_tft, create_tft_dataset

def _parse_tft_predictions(prediction_output, last_date, periods):
    """Helper function to format TFT predictions into a clean DataFrame."""
    # The model returns predictions for all quantiles. We extract them.
    # The shape is (batch_size, time_steps, quantiles)
    preds = prediction_output.prediction[0]
    
    # Get the dates for the forecast horizon
    dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    
    # Create the DataFrame
    forecast_df = pd.DataFrame({
        'ds': dates,
        # Quantiles are at indices: 0 (p10), 1 (p25), 2 (p50), 3 (p75), 4 (p90)
        'customers_p10': preds[:, 0].numpy(),
        'customers_p50': preds[:, 2].numpy(), # p50 is the median forecast
        'customers_p90': preds[:, 4].numpy()
    })

    # Clip and round for realistic business values
    forecast_df['predicted_customers'] = forecast_df['customers_p50'].clip(lower=0).round().astype(int)
    
    return forecast_df

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using the Temporal Fusion Transformer.
    This replaces the old LightGBM recursive strategy.
    """
    # --- 1. Data Preparation ---
    df_prepared = prepare_data_for_tft(historical_df, events_df)
    
    # We use ~60 days of history to predict the next 15 days
    max_encoder_length = 60
    max_prediction_length = periods

    # --- 2. Create TFT Dataset and Dataloader ---
    training_dataset = create_tft_dataset(df_prepared, max_encoder_length, max_prediction_length)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)

    # --- 3. Initialize and Train the Model ---
    # For reproducibility
    pl.seed_everything(42)

    # Define the trainer. For a real app, you might increase epochs.
    # On Streamlit Cloud's free tier, keep epochs low to avoid timeouts.
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="cpu",
        gradient_clip_val=0.1,
        logger=False # Disable logging to avoid creating log files on Streamlit
    )

    # Define the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]), # Key for uncertainty
        optimizer="AdamW"
    )

    # Train the model
    trainer.fit(tft, train_dataloaders=train_dataloader)

    # --- 4. Generate Predictions ---
    # Create a dataloader for the single series we want to predict
    pred_dataloader = training_dataset.to_dataloader(train=False, batch_size=1)
    
    # Make the prediction
    raw_predictions = tft.predict(pred_dataloader, mode="quantiles", return_x=True)

    # --- 5. Finalize and Return ---
    last_historical_date = historical_df['date'].max()
    final_forecast = _parse_tft_predictions(raw_predictions, last_historical_date, periods)
    
    # For simplicity, we'll use a stable, recent ATV to calculate sales
    recent_atv = historical_df.tail(14)['sales'].sum() / historical_df.tail(14)['customers'].sum()
    final_forecast['predicted_atv'] = recent_atv
    final_forecast['predicted_sales'] = final_forecast['predicted_customers'] * final_forecast['predicted_atv']

    return final_forecast, tft
