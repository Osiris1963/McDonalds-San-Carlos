# forecasting.py (Re-engineered with GluonTS and DeepAR)
import pandas as pd
import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NormalOutput
from pytorch_lightning import Trainer

# Import our enhanced feature engineering function
from data_processing import create_features

def prepare_gluon_dataset(df, target_cols, feature_cols, prediction_length):
    """
    Converts a pandas DataFrame into the ListDataset format required by GluonTS.
    
    Args:
        df (pd.DataFrame): The DataFrame with all historical data and features.
        target_cols (list): List of column names to be predicted (e.g., ['customers', 'atv']).
        feature_cols (list): List of feature column names to be used by the model.
        prediction_length (int): The number of future time steps to predict.

    Returns:
        A tuple of (ListDataset for training, future features DataFrame).
    """
    # Features that change over time are 'dynamic' features in GluonTS
    feat_dynamic_real = df[feature_cols].values.T
    
    # The main time series targets we want to predict
    target_values = df[target_cols].values.T
    
    # We create a GluonTS dataset object. We need to provide the target values,
    # the start date, and the time-varying features.
    train_ds = ListDataset(
        [{
            FieldName.TARGET: target,
            FieldName.START: df['date'].min(),
            FieldName.FEAT_DYNAMIC_REAL: feat_dynamic_real
        }],
        freq="D"  # Daily frequency
    )
    
    # We also need the features for the future period we want to predict.
    # We'll create a placeholder for future dates and generate features for them.
    future_date_range = pd.date_range(
        start=df['date'].max() + pd.Timedelta(days=1),
        periods=prediction_length
    )
    future_df_template = pd.DataFrame({'date': future_date_range})
    
    # To create features for the future, we need to append this to the historical data
    # so that rolling features can be calculated correctly.
    combined_df = pd.concat([df, future_df_template], ignore_index=True)
    
    # We pass a dummy events_df because future events are already in the main df
    # up to the historical point. The feature creator will handle the rest.
    combined_df_featured = create_features(combined_df, pd.DataFrame())
    
    future_features_df = combined_df_featured.tail(prediction_length)
    
    return train_ds, future_features_df[feature_cols]


def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a multivariate, point forecast using a DeepAR model from GluonTS.
    This is a unified model that learns from the entire dataset at once.
    
    Args:
        historical_df (pd.DataFrame): The complete historical sales data.
        events_df (pd.DataFrame): Data on future activities and events.
        periods (int): The number of days to forecast into the future.

    Returns:
        pd.DataFrame: A DataFrame containing the future forecast, or an empty DataFrame on error.
    """
    try:
        # --- 1. Feature Engineering ---
        # Create a rich feature set from the historical data
        df_featured = create_features(historical_df, events_df)
        
        target_cols = ['customers', 'atv']
        
        # Define the feature set for the model. Exclude non-numeric/ID columns.
        feature_cols = [
            col for col in df_featured.columns if col not in 
            ['date', 'doc_id', 'sales', 'customers', 'atv', 'add_on_sales', 'day_type', 'day_type_notes']
        ]

        # --- 2. Data Preparation for GluonTS ---
        gluon_ds, future_features = prepare_gluon_dataset(
            df_featured,
            target_cols,
            feature_cols,
            prediction_length=periods
        )

        # --- 3. Model Definition (DeepAR) ---
        # DeepAR is an autoregressive RNN model, well-suited for this task.
        # We configure it for speed and compatibility with Streamlit's environment.
        # Using NormalOutput for a point forecast (it will predict the mean).
        estimator = DeepAREstimator(
            freq="D",
            prediction_length=periods,
            num_layers=2,
            num_cells=40,
            distr_output=NormalOutput(), # Predicts mean and std, we will use the mean.
            trainer_kwargs={
                "max_epochs": 50,
                "accelerator": "cpu",  # Ensures it runs on Streamlit hosting without a GPU
                "enable_progress_bar": False, # Cleaner logs for production
                "logger": False # Disable verbose logging
            }
        )

        # --- 4. Model Training ---
        # This trains the model on the entire historical dataset.
        # The trained object is called a 'Predictor'.
        print("Starting model training...")
        predictor = estimator.train(training_data=gluon_ds)
        print("Model training complete.")

        # --- 5. Prediction ---
        # Use the trained predictor to forecast the future.
        # We must provide the known future features we prepared earlier.
        forecast_it = predictor.predict(
            dataset=gluon_ds,
            future_features=future_features.values.T
        )
        
        # The result is an iterator, get the first (and only) forecast object
        forecast = next(iter(forecast_it))

        # --- 6. Process Forecast Output ---
        # We get the mean of the predicted distribution for our point forecast.
        mean_preds = forecast.mean

        future_dates = pd.date_range(
            start=historical_df['date'].max() + pd.Timedelta(days=1),
            periods=periods
        )
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'forecast_customers': mean_preds[:, 0], # First target column
            'forecast_atv': mean_preds[:, 1],       # Second target column
        })

        # --- 7. Final Calculations & Cleanup ---
        forecast_df['forecast_sales'] = forecast_df['forecast_customers'] * forecast_df['forecast_atv']
        
        # Ensure forecasts are non-negative and customers are integers
        forecast_df['forecast_sales'] = forecast_df['forecast_sales'].clip(lower=0)
        forecast_df['forecast_customers'] = forecast_df['forecast_customers'].clip(lower=0).round().astype(int)
        forecast_df['forecast_atv'] = forecast_df['forecast_atv'].clip(lower=0)

        return forecast_df.round(2)

    except Exception as e:
        print(f"An error occurred during forecast generation: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

