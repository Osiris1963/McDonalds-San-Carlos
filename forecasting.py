# forecasting.py — two-model LightGBM with recursive multi-step generation

import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def generate_forecast(historical_df: pd.DataFrame, events_df: pd.DataFrame | None, periods: int = 15):
    """
    Train two LightGBM models (Customers, ATV) and recursively forecast next `periods` days.
    Returns:
        final_forecast_df, customer_model
    """
    # 1) Build features from full history
    df_feat = create_advanced_features(historical_df, events_df)
    if df_feat.empty:
        return pd.DataFrame(), None

    # Targets
    TARGET_CUST = "customers"
    TARGET_ATV = "atv"

    # Feature columns (exclude targets & raw date)
    DROP = {"date", "customers", "sales", "atv"}
    FEATURES = [c for c in df_feat.columns if c not in DROP]

    train_df = df_feat.dropna(subset=[TARGET_CUST, TARGET_ATV]).copy()
    if len(train_df) < 30:
        return pd.DataFrame(), None

    # LightGBM params
    params = dict(
        n_estimators=700,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=-1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        boosting_type="gbdt",
    )

    # Train models
    model_cust = lgb.LGBMRegressor(**params)
    model_cust.fit(train_df[FEATURES], train_df[TARGET_CUST])

    model_atv = lgb.LGBMRegressor(**params)
    model_atv.fit(train_df[FEATURES], train_df[TARGET_ATV])

    # 2) Recursive forecast
    future = []
    history_df = historical_df.copy()
    last_date = pd.to_datetime(history_df["date"]).max()

    for i in range(periods):
        pred_date = last_date + timedelta(days=i + 1)
        placeholder = pd.DataFrame([{"date": pred_date}])
        tmp = pd.concat([history_df, placeholder], ignore_index=True)

        feat_pred = create_advanced_features(tmp, events_df)
        # most recent row corresponds to pred_date
        row = feat_pred.loc[feat_pred["date"] == pd.to_datetime(pred_date)].copy()
        row_features = row[FEATURES].fillna(0.0)

        pred_customers = float(model_cust.predict(row_features)[0])
        pred_customers = max(0.0, pred_customers)

        pred_atv = float(model_atv.predict(row_features)[0])
        pred_atv = max(0.0, pred_atv)

        pred_sales = pred_customers * pred_atv

        future.append({
            "date": pred_date,
            "customers": pred_customers,
            "atv": pred_atv,
            "sales": pred_sales,
        })

        # append synthetic row to history so next step can build features
        history_df = pd.concat([history_df, pd.DataFrame([{
            "date": pred_date,
            "customers": pred_customers,
            "atv": pred_atv,
            "sales": pred_sales
        }])], ignore_index=True)

    if not future:
        return pd.DataFrame(), None

    final = pd.DataFrame(future).rename(columns={
        "date": "ds",
        "customers": "forecast_customers",
        "atv": "forecast_atv",
        "sales": "forecast_sales",
    }).copy()

    final["forecast_customers"] = final["forecast_customers"].clip(lower=0).round().astype(int)
    final["forecast_atv"] = final["forecast_atv"].clip(lower=0)
    final["forecast_sales"] = final["forecast_sales"].clip(lower=0)
    return final[["ds", "forecast_customers", "forecast_atv", "forecast_sales"]], model_cust
