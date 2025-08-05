# forecasting.py

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, MultiLoss
from pytorch_forecasting.metrics import MAE
from data_processing import prepare_data_for_tft

def generate_forecast(historical_df, events_df, periods=15):
    """
    Orchestrates the entire deep learning forecast process using a stable stack.
    """
    dataset = prepare_data_for_tft(historical_df, events_df, periods_to_forecast=periods)
    
    dataloader = dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=40,
        gpus=0,
        weights_summary=None,
        gradient_clip_val=0.1,
        limit_train_batches=50,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=MultiLoss([MAE(), MAE()]),
        optimizer="AdamW"
    )

    trainer.fit(tft, train_dataloaders=dataloader)

    new_raw_predictions, x = tft.predict(dataset.to_dataloader(train=False), mode="raw", return_x=True)
    
    last_prediction_point = new_raw_predictions['prediction'][0]

    pred_customers = last_prediction_point[:, 0].numpy()
    pred_atv = last_prediction_point[:, 1].numpy()

    forecast_dates = [
        historical_df['date'].max() + pd.DateOffset(days=x) 
        for x in range(1, periods + 1)
    ]

    final_df = pd.DataFrame({
        'ds': forecast_dates,
        'forecast_customers': pred_customers,
        'forecast_atv': pred_atv,
    })
    
    final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
    
    final_df['forecast_sales'] = final_df['forecast_sales'].clip(lower=0)
    final_df['forecast_customers'] = final_df['forecast_customers'].clip(lower=0).round()
    final_df['forecast_atv'] = final_df['forecast_atv'].clip(lower=0)
    
    return final_df, tft, dataset
