# forecasting.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from data_processing import create_advanced_features

def _winsorize(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)

def _seasonal_baseline(history_df: pd.DataFrame, window_weeks: int = 8):
    """
    Stable weekday baseline using the last N weeks.
    Returns two dicts: {dow: avg_customers}, {dow: avg_atv}
    """
    if history_df is None or history_df.empty:
        return {i: 0.0 for i in range(7)}, {i: 0.0 for i in range(7)}

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["dow"] = df["date"].dt.dayofweek
    cutoff = df["date"].max() - pd.Timedelta(weeks=window_weeks)
    df = df[df["date"] > cutoff]

    if df.empty:
        return {i: 0.0 for i in range(7)}, {i: 0.0 for i in range(7)}

    cust_by_dow = df.groupby("dow")["customers"].mean().to_dict() if "customers" in df else {i: 0.0 for i in range(7)}
    # Derive ATV baseline from base sales per customer if sales available
    if "sales" in df and "customers" in df:
        with np.errstate(divide="ignore", invalid="ignore"):
            atv_series = df["sales"] / df["customers"].replace({0: np.nan})
        atv_by_dow = df.assign(_atv=atv_series).groupby("dow")["_atv"].mean().fillna(0.0).to_dict()
    else:
        atv_by_dow = {i: 0.0 for i in range(7)}

    # Fill any missing DOWs with globals
    global_c = float(df["customers"].mean() if "customers" in df else 0.0)
    global_a = float((df["sales"].sum() / max(df["customers"].sum(), 1)) if "sales" in df and "customers" in df else 0.0)
    for i in range(7):
        cust_by_dow.setdefault(i, global_c)
        atv_by_dow.setdefault(i, global_a)
    return cust_by_dow, atv_by_dow

def generate_forecast(historical_df: pd.DataFrame,
                      events_df: pd.DataFrame | None,
                      periods: int = 15):
    """
    Trains two LightGBM models (customers, atv) with time-based validation + early stopping,
    then performs recursive multi-step forecasting. Predictions are blended with a
    seasonal weekday baseline to reduce drift.
    Returns: (forecast_df, customer_model)
      forecast_df columns: ['ds', 'forecast_customers', 'forecast_atv', 'forecast_sales']
    """
    if historical_df is None or historical_df.empty:
        return pd.DataFrame(columns=["ds", "forecast_customers", "forecast_atv", "forecast_sales"]), None

    # 1) Build features
    df_feat = create_advanced_features(historical_df, events_df)
    # Keep rows where rolling windows are valid to avoid noisy early periods
    if "sales_rm14" in df_feat.columns:
        train_df = df_feat.dropna(subset=["sales_rm14"]).reset_index(drop=True)
    else:
        train_df = df_feat.reset_index(drop=True)

    if train_df.empty:
        return pd.DataFrame(columns=["ds", "forecast_customers", "forecast_atv", "forecast_sales"]), None

    # Targets
    TARGET_C = "customers"
    TARGET_A = "atv"

    # Exclude raw & meta from features
    EXCLUDE = {"date", "sales", "customers", "atv", "doc_id", "day_type", "day_type_notes"}
    FEATURES = [c for c in train_df.columns if c not in EXCLUDE]

    # Winsorize to de-sensitize outliers
    y_c = _winsorize(train_df[TARGET_C].astype(float))
    y_a = _winsorize(train_df[TARGET_A].astype(float))

    # 2) Time-based validation split (last 28 days if possible), with early stopping
    last_day = train_df["date"].max()
    val_start = last_day - pd.Timedelta(days=27)
    is_val = train_df["date"] >= val_start
    if is_val.sum() < 14:  # fallback if very short history
        is_val = train_df.index >= max(len(train_df) - 20, 0)

    X_tr, X_va = train_df.loc[~is_val, FEATURES], train_df.loc[is_val, FEATURES]
    y_tr_c, y_va_c = y_c.loc[~is_val], y_c.loc[is_val]
    y_tr_a, y_va_a = y_a.loc[~is_val], y_a.loc[is_val]

    params = dict(
        objective="regression_l1",
        metric="rmse",
        n_estimators=5000,
        learning_rate=0.035,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.15,
        reg_lambda=0.15,
        num_leaves=31,
        verbose=-1,
        n_jobs=-1,
        seed=42,
        boosting_type="gbdt",
    )
    model_c = lgb.LGBMRegressor(**params)
    model_a = lgb.LGBMRegressor(**params)

    model_c.fit(
        X_tr, y_tr_c,
        eval_set=[(X_va, y_va_c)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )
    model_a.fit(
        X_tr, y_tr_a,
        eval_set=[(X_va, y_va_a)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    # 3) Seasonal baseline
    hist = historical_df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    cust_base_by_dow, atv_base_by_dow = _seasonal_baseline(hist, window_weeks=8)

    # 4) Recursive multi-step forecast
    preds = []
    last_date = hist["date"].max()

    for h in range(1, periods + 1):
        target_date = last_date + timedelta(days=h)

        # Append placeholder row to recompute features with lags/rollings
        tmp = pd.concat([hist, pd.DataFrame([{"date": target_date}])], ignore_index=True)
        feat_pred = create_advanced_features(tmp, events_df)

        Xp = feat_pred[FEATURES].iloc[-1:]
        cust_hat = float(model_c.predict(Xp)[0])
        atv_hat = float(model_a.predict(Xp)[0])

        # Blend with weekday baseline (more seasonal farther out)
        dow = int(target_date.weekday())
        base_c = float(cust_base_by_dow.get(dow, cust_hat))
        base_a = float(atv_base_by_dow.get(dow, atv_hat))
        w = 0.75 if h <= 7 else 0.65
        cust_hat = max(0.0, w * cust_hat + (1 - w) * base_c)
        atv_hat = max(0.0, w * atv_hat + (1 - w) * base_a)

        row = {
            "date": target_date,
            "customers": cust_hat,
            "atv": atv_hat,
            "sales": cust_hat * atv_hat,
            "add_on_sales": 0.0,
        }
        preds.append(row)

        # Feed back into history so next horizon has fresh lags
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    if not preds:
        return pd.DataFrame(columns=["ds", "forecast_customers", "forecast_atv", "forecast_sales"]), model_c

    out = pd.DataFrame(preds).rename(columns={
        "date": "ds",
        "customers": "forecast_customers",
        "atv": "forecast_atv",
        "sales": "forecast_sales",
    })
    # Business-safe clipping/rounding
    out["forecast_sales"] = out["forecast_sales"].clip(lower=0)
    out["forecast_customers"] = out["forecast_customers"].clip(lower=0).round().astype(int)
    out["forecast_atv"] = out["forecast_atv"].clip(lower=0)

    return out[["ds", "forecast_customers", "forecast_atv", "forecast_sales"]], model_c
