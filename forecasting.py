# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from data_processing import create_advanced_features, prepare_data_for_prophet
from optimizer import get_self_learning_bias  # Import the new brain

def generate_customer_forecast(historical_df, events_df, forecast_log_df=None, periods=15):
    # Ask the optimizer for the bias correction factor
    bias_multiplier = get_self_learning_bias(historical_df, forecast_log_df)
    
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)
    
    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]

    # Train with sample weights (recent data = 1.0 weight, old data = 0.2)
    model_cust = lgb.LGBMRegressor(objective='poisson', n_estimators=1000, learning_rate=0.05, verbose=-1)
    weights = np.linspace(0.2, 1.0, len(train_df))
    model_cust.fit(train_df[FEATURES], train_df['customers'], sample_weight=weights)

    future_predictions_list = []
    history_for_recursion = historical_df.copy()
    last_date = history_for_recursion['date'].max()
    last_atv = history_for_recursion['atv'].iloc[-1] if not history_for_recursion.empty else 250

    for i in range(periods):
        current_date = last_date + timedelta(days=i + 1)
        temp_df = pd.concat([history_for_recursion, pd.DataFrame([{'date': current_date}])], ignore_index=True)
        feat = create_advanced_features(temp_df, events_df)
        X = feat[FEATURES].iloc[-1:]

        # Predict and apply the 'optimizer' bias
        pred = max(0, model_cust.predict(X)[0] * bias_multiplier)

        new_row = {'date': current_date, 'customers': pred, 'atv': last_atv, 'sales': pred * last_atv}
        future_predictions_list.append(new_row)
        history_for_recursion = pd.concat([history_for_recursion, pd.DataFrame([new_row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_predictions_list)
    forecast_df.rename(columns={'date': 'ds', 'customers': 'forecast_customers'}, inplace=True)
    forecast_df['forecast_customers'] = forecast_df['forecast_customers'].round().astype(int)
    
    return forecast_df[['ds', 'forecast_customers']], model_cust, bias_multiplier

def generate_atv_forecast(historical_df, events_df, periods=15):
    # Keep your existing Prophet logic here
    prophet_df, regs = prepare_data_for_prophet(historical_df, events_df)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
    for r in regs: model.add_regressor(r)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    # ... (fill regressors as per your previous file)
    forecast = model.predict(future).tail(periods)
    return forecast[['ds', 'yhat']].rename(columns={'yhat': 'forecast_atv'}), model
