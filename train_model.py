# train_model.py
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore")

# --- Import from our project modules ---
from data_processing import load_from_firestore, create_features_for_tft

# --- Main Training Execution ---
if __name__ == '__main__':
    print("--- Starting Model Training ---")

    # --- 1. Initialize Firestore (using service account key for offline scripts) ---
    # IMPORTANT: Create a `serviceAccountKey.json` file for this to work.
    # Do NOT commit this file to public git repositories.
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firestore initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize Firestore. Ensure 'serviceAccountKey.json' is present.")
        print(f"Error details: {e}")
        exit()

    # --- 2. Load and Prepare Data ---
    print("Loading data from Firestore...")
    historical_df = load_from_firestore(db, 'historical_data')
    events_df = load_from_firestore(db, 'future_activities')
    
    if len(historical_df) < 60:
        print(f"FATAL: Insufficient data for training. Found {len(historical_df)} records, need at least 60.")
        exit()
        
    print(f"Loaded {len(historical_df)} historical records.")
    data = create_features_for_tft(historical_df, events_df)
    print("Feature creation complete.")

    # --- 3. Define Model Parameters and Create TimeSeriesDataSet ---
    # We will forecast for the next 15 days
    max_prediction_length = 15
    # The model will look at the last 60 days of data to make a prediction
    max_encoder_length = 60

    training_cutoff = data["time_idx"].max() - max_prediction_length

    # Define the dataset
    training_dataset = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=["customers", "atv"],
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_categoricals=["month", "dayofweek", "dayofmonth", "weekofyear", "is_event", "is_payday_period"],
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=["is_not_normal_day"],
        time_varying_unknown_reals=["customers", "atv"],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation set and dataloaders
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data, predict=True, stop_randomization=True)
    batch_size = 16
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    print("TimeSeriesDataSet and dataloaders created.")

    # --- 4. Configure and Train the Temporal Fusion Transformer ---
    # Define callbacks for the trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=".",
        filename="best_model",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu", # Change to "gpu" if you have a compatible GPU
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=False, # Disable logging for simplicity
    )

    # Define the model with pre-tuned hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles by default
        loss=MultiLoss([QuantileLoss(), QuantileLoss()]), # One loss for each target
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"Training with {tft.size()} parameters.")

    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("--- Model Training Complete ---")
    print("The best model has been saved to 'best_model.ckpt'")
