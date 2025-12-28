import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features

def generate_customer_forecast(historical_df, events_df, periods=15):
    """Hybrid Tweedie LGBM with Recursive Bias Correction."""
    df_feat = create_advanced_features(historical_df, events_df)
    
    # Feature whitelist: excludes non-predictive strings
    EXCLUDE = ['date', 'sales', 'customers', 'atv', 'doc_id', 'day_type']
    FEATURES = [c for c in df_feat.columns if c not in EXCLUDE]
    
    # --- Tweedie Engine: Specifically designed for 'Spiky' Retail Sales ---
    model = lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=1.1, # Critical setting to capture high-revenue days
        learning_rate=0.03,
        n_estimators=1200,
        num_leaves=127,
        importance_type='gain',
        verbosity=-1,
        bagging_fraction=0.8,
        bagging_freq=5
    )
    
    # Adaptive Weighting: Gives 10x more importance to recent data
    weights = np.linspace(0.1, 1.0, len(df_feat))
    model.fit(df_feat[FEATURES], df_feat['customers'], sample_weight=weights)

    # --- Self-Learning Feedback Loop (Backtesting) ---
    # The model 'tests' itself on the last 7 days and calculates a correction factor
    last_7 = df_feat.tail(7)
    recent_preds = model.predict(last_7[FEATURES])
    bias_factor = (last_7['customers'] / (recent_preds + 1e-5)).mean()
    bias_factor = np.clip(bias_factor, 0.85, 1.15) # Safety rails for the AI

    # Recursive Forecasting: One day at a time to maintain context
    future_rows = []
    working_df = historical_df.copy()
    
    for _ in range(periods):
        next_date = working_df['date'].max() + timedelta(days=1)
        temp_df = pd.concat([working_df, pd.DataFrame([{'date': next_date}])], ignore_index=True)
        feat_row = create_advanced_features(temp_df, events_df).iloc[-1:]
        
        raw_pred = model.predict(feat_row[FEATURES])[0]
        # Human-like adjustment: Apply the bias we found in the past week
        corrected_pred = max(0, raw_pred * bias_correction)
        
        new_entry = {'date': next_date, 'customers': corrected_pred}
        future_rows.append(new_entry)
        working_df = pd.concat([working_df, pd.DataFrame([new_entry])], ignore_index=True)

    forecast_df = pd.DataFrame(future_rows)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    return forecast_df, model

def generate_atv_forecast(historical_df, events_df, periods=15):
    """Uses Prophet to capture stable long-term price trends."""
    hist = historical_df.copy()
    hist['atv'] = (hist['sales'] / hist['customers'].replace(0, np.nan)).fillna(hist['sales'].mean()/100)
    
    df_p = hist[['date', 'atv']].rename(columns={'date': 'ds', 'atv': 'y'})
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    atv_res = forecast[['ds', 'yhat']].tail(periods).rename(columns={'yhat': 'forecast_atv'})
    return atv_res, m
