# forecasting.py
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
import torch
import warnings

warnings.filterwarnings("ignore")

def load_tft_model(model_path="best_model.ckpt"):
    """Loads a pre-trained TFT model from a checkpoint file."""
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        return model
    except FileNotFoundError:
        return None

def generate_forecast(model, historical_df, events_df, periods=15):
    """
    Generates a forecast using the pre-trained TFT model.
    """
    from data_processing import create_features_for_tft # Local import to avoid circular dependency
    
    if model is None:
        return pd.DataFrame(), None

    # Prepare data for prediction
    data_for_features = create_features_for_tft(historical_df, events_df)
    
    # Create the prediction dataset
    # The model will use the last `max_encoder_length` days from the data to predict the future
    encoder_data = data_for_features[lambda x: x.time_idx > data_for_features["time_idx"].max() - model.hparams.max_encoder_length]
    
    # Create the decoder data (future dates with known features)
    last_date = data_for_features["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    
    # This is a placeholder for future known features
    decoder_data = pd.DataFrame({
        "date": future_dates,
        "time_idx": (future_dates - data_for_features["date"].min()).days,
        "group": "store_1"
    })
    
    # Add future known features (e.g., payday, events)
    # This part must be robust to ensure all needed columns are present
    future_features = create_features_for_tft(decoder_data, events_df)
    new_prediction_data = pd.concat([encoder_data, future_features], ignore_index=True)
    
    # Make predictions
    raw_predictions = model.predict(
        new_prediction_data,
        mode="prediction", # "prediction" gives point forecast, "quantiles" gives ranges
        return_x=True
    )
    
    # Extract predictions for the future dates
    pred_df = pd.DataFrame()
    pred_df['ds'] = future_dates
    # The output is a tuple (predictions, x_values). We take the predictions for each target.
    # We take the median prediction (index 3 of the 7 quantiles)
    pred_df['forecast_customers'] = raw_predictions.output[0].numpy()[:, :, 3].flatten()
    pred_df['forecast_atv'] = raw_predictions.output[1].numpy()[:, :, 3].flatten()
    
    # Post-process the forecast
    pred_df['forecast_sales'] = pred_df['forecast_customers'] * pred_df['forecast_atv']
    pred_df['forecast_customers'] = pred_df['forecast_customers'].clip(lower=0).round()
    pred_df['forecast_atv'] = pred_df['forecast_atv'].clip(lower=0)
    pred_df['forecast_sales'] = pred_df['forecast_sales'].clip(lower=0)

    return pred_df.head(periods), raw_predictions

def get_interpretation_plot(model, raw_predictions):
    """
    Generates the interpretation plot from the model's prediction output.
    """
    if model is None or raw_predictions is None:
        return None
    
    interpretation = model.interpret_output(raw_predictions.output, reduction="sum")
    fig = model.plot_interpretation(interpretation)
    return fig
