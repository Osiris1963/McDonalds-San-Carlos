import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

# ---------------------------
# Utility
# ---------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denom != 0
    sm = np.mean((diff[mask] / denom[mask])) * 100.0 if mask.any() else 0.0
    return sm

def mase(y_true, y_pred, y_train):
    # scale by naive seasonal difference (period=7); fallback to 1
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    d = 7
    if len(y_train) <= d:
        scale = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else 1.0
    else:
        scale = np.mean(np.abs(y_train[d:] - y_train[:-d]))
    scale = max(scale, 1e-8)
    return np.mean(np.abs(y_true - y_pred)) / scale

# ---------------------------
# Customers: ETS baseline + recent-trend correction + event uplift + caps
# ---------------------------
def _ets_baseline_customers(hist: pd.DataFrame, H: int) -> pd.Series:
    y = hist["customers"].astype(float)
    # damped trend, weekly seasonality
    model = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, use_brute=True)
    fc = fit.forecast(H)
    fc.index = pd.date_range(hist.index.max() + pd.Timedelta(days=1), periods=H, freq="D")
    return fc.clip(lower=0)

def _recent_trend_multiplier(hist: pd.DataFrame, decay_lambda: float = 0.9) -> float:
    # Exponentially-weighted recent growth vs. same weekday lag7
    recent = hist.tail(28).copy()
    if len(recent) < 8:
        return 1.0
    valid = recent["customers"] / (recent["customers"].shift(7).replace(0, np.nan))
    valid = valid.dropna()
    if len(valid) == 0:
        return 1.0
    weights = np.array([decay_lambda ** i for i in range(len(valid)-1, -1, -1)], dtype=float)
    weights /= weights.sum()
    mult = float(np.sum(valid.values * weights))
    # bound the multiplier to avoid runaway
    return float(np.clip(mult, 0.85, 1.25))

def _weekday_growth_caps(hist: pd.DataFrame, fc_idx: pd.DatetimeIndex, base: pd.Series) -> pd.Series:
    # cap future customers relative to last *same weekday* value (± realistic bound)
    out = base.copy()
    last_by_dow = hist.groupby(hist.index.dayofweek)["customers"].last()
    caps = {}
    # caps per weekday: allow up to +40% and down to -40% vs last same weekday (tunable)
    up = 1.40
    down = 0.60
    for d in range(7):
        caps[d] = (last_by_dow.get(d, np.nan) * down, last_by_dow.get(d, np.nan) * up)
    for d in range(len(out)):
        dow = fc_idx[d].dayofweek
        low, high = caps.get(dow, (np.nan, np.nan))
        if not np.isnan(low):
            out.iloc[d] = np.clip(out.iloc[d], low, high)
    return out.clip(lower=0)

def forecast_customers_with_trend_correction(
    hist: pd.DataFrame,
    future_cal: pd.DataFrame,
    H: int,
    decay_lambda: float = 0.9,
    apply_weekday_caps: bool = True,
    event_uplift_pct: Optional[pd.Series] = None,
    return_bands: bool = True,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    base = _ets_baseline_customers(hist, H)
    mult = _recent_trend_multiplier(hist, decay_lambda=decay_lambda)
    fc = (base * mult).clip(lower=0)

    # apply events
    if event_uplift_pct is not None and len(event_uplift_pct) == H:
        fc = fc * (1.0 + (event_uplift_pct.values / 100.0))

    if apply_weekday_caps:
        fc = _weekday_growth_caps(hist, fc.index, fc)

    # Uncertainty via residual bootstrap from training residuals
    bands = None
    if return_bands:
        # crude residuals from last 56 days against lag-7 naive
        hist_res = hist.copy()
        if len(hist_res) > 14:
            resid = hist_res["customers"] - hist_res["customers"].shift(7)
            resid = resid.dropna()
            if len(resid) >= 20:
                draws = np.random.choice(resid.values, size=(1000, H), replace=True)
                sims = np.clip(fc.values + draws, 0, None)
                p10 = np.percentile(sims, 10, axis=0)
                p50 = fc.values
                p90 = np.percentile(sims, 90, axis=0)
                bands = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=fc.index)
    return fc, bands

# ---------------------------
# ATV: Direct multi-horizon LightGBM + guardrails + events
# ---------------------------
def _build_atv_feature_matrix(hist: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = hist.copy()
    # Features purposely exclude current-day ATV leakage
    feats = [
        "dow", "is_weekend", "is_holiday", "is_payday", "is_payday_minus1", "is_payday_plus1",
        "month", "week",
        "atv_lag7", "atv_lag14", "atv_lag28",
        "atv_roll7_med", "atv_roll14_med", "atv_roll28_med",
        "customers_lag7", "customers_lag14", "customers_lag28",
    ]
    X = df[feats].copy()
    y = df["atv"].astype(float).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return X, y

def _train_direct_horizon_models(hist: pd.DataFrame, H: int):
    # We train H independent models, each predicting ATV_t+h using the last available features.
    # For each horizon h, we align targets by shifting -h (future) and keep rows with full lags.
    X_full, y_full = _build_atv_feature_matrix(hist)
    models = []
    features = X_full.columns.tolist()

    for h in range(1, H+1):
        y_h = y_full.shift(-h)
        df_h = pd.concat([X_full, y_h.rename("y")], axis=1).dropna()
        if len(df_h) < 100:  # need enough samples
            # fallback simple model if very short history
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = lgb.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
            )
        model.fit(df_h[features], df_h["y"])
        models.append(model)
    return models, features

def _apply_guardrails_atv(hist: pd.DataFrame, preds: pd.Series, mad_mult: float) -> pd.Series:
    # Guardrails: median ± k * MAD (robust)
    atv_hist = hist["atv"].astype(float)
    med = atv_hist.median()
    mad = np.median(np.abs(atv_hist - med)) + 1e-6
    low = med - mad_mult * mad
    high = med + mad_mult * mad
    return preds.clip(lower=max(5.0, low), upper=high)  # never below ₱5

def forecast_atv_direct(
    hist: pd.DataFrame,
    future_cal: pd.DataFrame,
    H: int,
    guardrail_mad_mult: float = 3.0,
    event_uplift_pct: Optional[pd.Series] = None,
    return_bands: bool = True,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    models, features = _train_direct_horizon_models(hist, H)

    # Build "last row" features for each horizon from future_cal by copying its calendar features
    # and merging with last known rolling/lag values from history.
    last = hist.iloc[-1:].copy()
    # construct the feature rows for each horizon using future calendar and last known lags/rollings
    Xf = future_cal.copy()

    # Use last known lag/rolling as static features (reasonable for short horizon)
    for col in ["atv_lag7","atv_lag14","atv_lag28","atv_roll7_med","atv_roll14_med","atv_roll28_med",
                "customers_lag7","customers_lag14","customers_lag28"]:
        Xf[col] = float(last[col].iloc[0]) if col in last.columns else np.nan
    Xf = Xf[features].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

    preds = []
    for i, m in enumerate(models):
        # each model expects a single row for horizon i
        row = Xf.iloc[[i]].copy() if i < len(Xf) else Xf.iloc[[-1]].copy()
        preds.append(float(m.predict(row)[0]))
    preds = pd.Series(preds, index=future_cal.index, name="atv")

    # Apply events
    if event_uplift_pct is not None and len(event_uplift_pct) == H:
        preds = preds * (1.0 + (event_uplift_pct.values / 100.0))

    # Guardrails
    preds = _apply_guardrails_atv(hist, preds, mad_mult=guardrail_mad_mult)

    bands = None
    if return_bands:
        # simple residual-based bands from in-sample errors (last model as proxy)
        X_hist, y_hist = _build_atv_feature_matrix(hist)
        # naive residuals using GradientBoosting as proxy (fast)
        proxy = GradientBoostingRegressor(random_state=42).fit(X_hist, y_hist)
        resid = y_hist - proxy.predict(X_hist)
        if len(resid) >= 30:
            draws = np.random.choice(resid.values, size=(1000, H), replace=True)
            sims = preds.values + draws
            p10 = np.percentile(sims, 10, axis=0)
            p50 = preds.values
            p90 = np.percentile(sims, 90, axis=0)
            bands = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=preds.index)
    return preds, bands

# ---------------------------
# Combine to Sales + Bands
# ---------------------------
def combine_sales_and_bands(
    dates: pd.DatetimeIndex,
    customers: pd.Series,
    customers_bands: Optional[pd.DataFrame],
    atv: pd.Series,
    atv_bands: Optional[pd.DataFrame],
    return_bands: bool = True,
) -> pd.DataFrame:
    out = pd.DataFrame(index=dates)
    if return_bands and (customers_bands is not None) and (atv_bands is not None):
        out["customers_p10"] = np.maximum(0, customers_bands["p10"])
        out["customers_p50"] = np.maximum(0, customers_bands["p50"])
        out["customers_p90"] = np.maximum(0, customers_bands["p90"])

        out["atv_p10"] = np.maximum(0, atv_bands["p10"])
        out["atv_p50"] = np.maximum(0, atv_bands["p50"])
        out["atv_p90"] = np.maximum(0, atv_bands["p90"])

        out["sales_p10"] = out["customers_p10"] * out["atv_p10"]
        out["sales_p50"] = out["customers_p50"] * out["atv_p50"]
        out["sales_p90"] = out["customers_p90"] * out["atv_p90"]
    else:
        out["customers_p50"] = customers
        out["atv_p50"] = atv
        out["sales_p50"] = customers * atv
    return out

# ---------------------------
# Backtesting (rolling origin)
# ---------------------------
def backtest_metrics(hist: pd.DataFrame, horizon: int = 15, folds: int = 6) -> pd.DataFrame:
    # We'll simulate the pipeline with decreasing cutoffs
    # For speed, do minimal bands here; core metrics use p50
    cutoffs = []
    smapes = {"customers": [], "atv": [], "sales": []}
    mases = {"customers": [], "atv": [], "sales": []}

    total_len = len(hist)
    min_train = 200 if total_len > 400 else int(total_len * 0.5)
    step = max(1, (total_len - min_train - horizon) // max(1, folds))

    for i in range(folds):
        train_end = min_train + i * step
        if train_end + horizon >= total_len:
            break
        train = hist.iloc[:train_end].copy()
        test = hist.iloc[train_end:train_end+horizon].copy()
        future_idx = test.index

        # Build future calendar (only calendar features are needed)
        future_cal = test[["dow","is_weekend","is_holiday","is_payday","is_payday_minus1","is_payday_plus1","month","week"]].copy()

        cust_fc, _ = forecast_customers_with_trend_correction(
            hist=train, future_cal=future_cal, H=horizon, return_bands=False
        )
        atv_fc, _ = forecast_atv_direct(
            hist=train, future_cal=future_cal, H=horizon, return_bands=False
        )

        sales_fc = cust_fc * atv_fc

        # metrics
        smapes["customers"].append(smape(test["customers"], cust_fc))
        smapes["atv"].append(smape(test["atv"], atv_fc))
        smapes["sales"].append(smape(test["sales"], sales_fc))

        mases["customers"].append(mase(test["customers"], cust_fc, train["customers"]))
        mases["atv"].append(mase(test["atv"], atv_fc, train["atv"]))
        mases["sales"].append(mase(test["sales"], sales_fc, train["sales"]))

        cutoffs.append(train.index.max())

    data = {
        "cutoff": cutoffs,
        "sMAPE_customers": smapes["customers"],
        "sMAPE_atv": smapes["atv"],
        "sMAPE_sales": smapes["sales"],
        "MASE_customers": mases["customers"],
        "MASE_atv": mases["atv"],
        "MASE_sales": mases["sales"],
    }
    return pd.DataFrame(data)
