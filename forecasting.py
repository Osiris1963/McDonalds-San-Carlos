# forecasting.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

# ---------- Helper Baselines ----------

def _same_dow_baseline(history_df, date, var):
    """
    Return robust same-weekday baseline using the prior 8 occurrences.
    """
    if history_df.empty:
        return np.nan
    target_dow = date.weekday()
    hist = history_df[history_df['date'].dt.weekday == target_dow].copy()
    hist = hist[hist['date'] < date].sort_values('date', ascending=False).head(8)
    if hist.empty or var not in hist.columns:
        return np.nan
    return np.nanmedian(hist[var].values)

def _fallback_rolling(history_df, date, var, window=28):
    """
    Fallback rolling mean from the last 'window' real observations before 'date'.
    """
    hist = history_df[history_df['date'] < date].copy().tail(window)
    if hist.empty or var not in hist.columns:
        return np.nan
    return np.nanmean(hist[var].values)

def _damped_blend(model_val, baseline_val, weight):
    """
    Blend model and baseline. If baseline is NaN, use model. If model is NaN, use baseline.
    """
    if np.isnan(model_val) and np.isnan(baseline_val):
        return np.nan
    if np.isnan(baseline_val):
        return model_val
    if np.isnan(model_val):
        return baseline_val
    return weight * model_val + (1.0 - weight) * baseline_val

def _clamp_to_reasonable_range(value, baseline, max_dev=0.35):
    """
    Limit predicted value to within +/- max_dev of baseline to prevent runaway trend-chasing.
    """
    if np.isnan(baseline):
        return max(0.0, value)
    lower = (1.0 - max_dev) * baseline
    upper = (1.0 + max_dev) * baseline
    return float(min(max(value, lower), upper))


def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a forecast using a unified LightGBM model with *stabilized* logic:
      - No ultra-short lags (no lag_1 or lag_2)
      - Rich same-weekday baselines (8-week median)
      - Dynamic blending (model vs baseline) with event-aware weighting
      - Clamp to reasonable range around the baseline
    """
    # --- 1) Feature Engineering & Train Split ---
    df_featured = create_advanced_features(historical_df, events_df)

    # Drop initial rows that can't compute weekly-based features
    # (Use a conservative cutoff to ensure quality)
    needed_col = 'sales_rolling_mean_28'
    if needed_col in df_featured.columns:
        train_df = df_featured.dropna(subset=[needed_col]).reset_index(drop=True)
    else:
        train_df = df_featured.copy().reset_index(drop=True)

    FEATURES = [c for c in train_df.columns if c not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes'
    ]]
    TARGET_CUST = 'customers'
    TARGET_ATV = 'atv'

    # --- 2) Models with stronger regularization (smoother, less reactive) ---
    lgbm_params = dict(
        objective='regression_l1',
        metric='rmse',
        n_estimators=1600,
        learning_rate=0.03,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        reg_alpha=0.6,
        reg_lambda=0.9,
        num_leaves=23,
        min_child_samples=50,
        verbose=-1,
        n_jobs=-1,
        seed=42,
        boosting_type='gbdt',
    )

    model_cust = lgb.LGBMRegressor(**lgbm_params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST])

    model_atv = lgb.LGBMRegressor(**lgbm_params)
    model_atv.fit(train_df[FEATURES], train_df[TARGET_ATV])

    # --- 3) Recursive Forecasting with stabilized logic ---
    future_predictions = []
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_date = last_date + timedelta(days=i + 1)

        # Build features for the current_date by temporarily appending a placeholder row
        temp_df = pd.concat([history_df, pd.DataFrame([{'date': current_date}])], ignore_index=True)
        featured = create_advanced_features(temp_df, events_df)
        X_pred = featured[FEATURES].iloc[-1:]

        # Raw model predictions
        model_c = float(model_cust.predict(X_pred)[0])
        model_a = float(model_atv.predict(X_pred)[0])

        # Robust same-weekday baselines
        base_c = _same_dow_baseline(history_df, current_date, 'customers')
        base_a = _same_dow_baseline(history_df, current_date, 'atv')

        # Fallback if missing
        if np.isnan(base_c):
            base_c = _fallback_rolling(history_df, current_date, 'customers', window=28)
        if np.isnan(base_a):
            base_a = _fallback_rolling(history_df, current_date, 'atv', window=28)

        # Dynamic blend weight: trust model more on events, otherwise rely more on baseline
        is_event = 0
        if events_df is not None and not events_df.empty:
            _evt = events_df.copy()
            _evt['date'] = pd.to_datetime(_evt['date']).dt.normalize()
            is_event = int((_evt['date'] == pd.to_datetime(current_date).normalize()).any())
        blend_w = 0.75 if is_event else 0.55

        # Blend + clamp to avoid runaway changes vs baseline
        blended_c = _damped_blend(model_c, base_c, weight=blend_w)
        blended_a = _damped_blend(model_a, base_a, weight=blend_w)

        blended_c = _clamp_to_reasonable_range(blended_c, base_c, max_dev=0.35)
        blended_a = _clamp_to_reasonable_range(blended_a, base_a, max_dev=0.35)

        # Non-negative and practical
        blended_c = max(0.0, blended_c)
        blended_a = max(0.0, blended_a)

        # Compose final row
        new_row = {
            'date': current_date,
            'customers': blended_c,
            'atv': blended_a,
            'sales': blended_c * blended_a,
            'add_on_sales': 0.0
        }
        future_predictions.append(new_row)

        # Append the BLENDED result back to history (stabilizes subsequent steps)
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    if not future_predictions:
        return pd.DataFrame(), None

    # --- 4) Finalize ---
    final_forecast = pd.DataFrame(future_predictions)
    final_forecast.rename(columns={
        'date': 'ds',
        'customers': 'forecast_customers',
        'atv': 'forecast_atv',
        'sales': 'forecast_sales'
    }, inplace=True)

    final_forecast['forecast_sales'] = final_forecast['forecast_sales'].clip(lower=0)
    final_forecast['forecast_customers'] = (
        final_forecast['forecast_customers'].clip(lower=0).round().astype(int)
    )
    final_forecast['forecast_atv'] = final_forecast['forecast_atv'].clip(lower=0)

    # Return forecast and the (customer) model for feature importance display in the UI
    return final_forecast[['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']], model_cust
