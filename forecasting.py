# forecasting.py
import pandas as pd
import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from datetime import timedelta
from data_processing import prepare_data_for_tft, create_tft_dataset

def _parse_tft_predictions(prediction_output, last_date, periods):
    """
    Helper function to format single-value TFT predictions into a clean DataFrame.
    This is much simpler than the quantile version.
    """
    # The model returns a single predicted value per timestep.
    # Shape is (batch_size, time_steps)
    preds = prediction_output.prediction[0].numpy()
    
    dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    
    forecast_df = pd.DataFrame({
        'ds': dates,
        'predicted_customers': preds,
    })

    forecast_df['predicted_customers'] = forecast_df['predicted_customers'].clip(lower=0).round().astype(int)
    
    return forecast_df

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a simplified and robust Temporal Fusion Transformer.
    """
    # --- 1. Data Preparation ---
    df_prepared = prepare_data_for_tft(historical_df, events_df)
    max_encoder_length = 60
    max_prediction_length = periods

    # --- 2. Create TFT Dataset and Dataloader ---
    training_dataset = create_tft_dataset(df_prepared, max_encoder_length, max_prediction_length)
    # Use batch_size=16 and num_workers=0 for stability on free cloud tiers
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
    val_dataloader = training_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)

    # --- 3. Initialize and Train the Model ---
    pl.seed_everything(42)

    # The logger=False argument is important for cloud deployments
    trainer = pl.Trainer(
        max_epochs=20, # Reduced epochs for faster, stable runs
        accelerator="cpu",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        logger=False 
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=16, # Simplified model size
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        # THIS IS THE KEY CHANGE: Simplified loss function
        loss=MSELoss(),
        optimizer="AdamW"
    )

    # --- 4. Train the model ---
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # --- 5. Generate Predictions ---
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    # Predict using the validation dataloader for simplicity
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    # --- 6. Finalize and Return ---
    last_historical_date = historical_df['date'].max()
    final_forecast = _parse_tft_predictions(raw_predictions, last_historical_date, periods)
    
    recent_atv = historical_df.tail(14)['sales'].sum() / historical_df.tail(14)['customers'].sum()
    final_forecast['predicted_atv'] = recent_atv
    final_forecast['predicted_sales'] = final_forecast['predicted_customers'] * final_forecast['predicted_atv']

    return final_forecast, best_tft
