# forecasting.py
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from datetime import timedelta
from data_processing import prepare_data_for_tft, create_tft_dataset
from pytorch_lightning.callbacks import ModelCheckpoint

def _parse_tft_predictions(prediction_output, last_date, periods):
    preds = prediction_output[0].numpy()
    dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    forecast_df = pd.DataFrame({'ds': dates, 'predicted_customers': preds})
    forecast_df['predicted_customers'] = forecast_df['predicted_customers'].clip(lower=0).round().astype(int)
    return forecast_df

def generate_forecast(historical_df, events_df, periods=15):
    df_prepared = prepare_data_for_tft(historical_df, events_df)
    max_encoder_length = 60
    max_prediction_length = periods

    training_dataset = create_tft_dataset(df_prepared, max_encoder_length, max_prediction_length)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
    val_dataloader = training_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)

    pl.seed_everything(42)

    # Define a callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model"
    )

    # Use the syntax for PyTorch Lightning v1.9.5
    trainer = pl.Trainer(
        max_epochs=20,
        enable_model_summary=False,
        gradient_clip_val=0.1,
        logger=False,
        callbacks=[checkpoint_callback]
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=RMSE(),
        optimizer="AdamW"
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    last_historical_date = historical_df['date'].max()
    final_forecast = _parse_tft_predictions(raw_predictions, last_historical_date, periods)
    
    recent_atv_data = historical_df.tail(30)
    if recent_atv_data['customers'].sum() > 0:
        recent_atv = recent_atv_data['sales'].sum() / recent_atv_data['customers'].sum()
    else:
        recent_atv = historical_df['atv'].mean()

    final_forecast['predicted_atv'] = recent_atv
    final_forecast['predicted_sales'] = final_forecast['predicted_customers'] * final_forecast['predicted_atv']

    return final_forecast, best_tft
