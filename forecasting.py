# forecasting.py — resilient to short history, LightGBM optional

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import GradientBoostingRegressor

# LightGBM is great but optional (falls back if unavailable)
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# ---------- Metrics ----------
def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denom != 0
    return np.mean(diff[mask] / denom[mask]) * 100.0 if mask.any() else 0.0

def mase(y_true, y_pred, y_train):
    y_true, y_pred, y_train = map(lambda a: np.asarray(a, float), (y_true, y_pred, y_train))
    d = 7
    scale = np.mean(np.abs(y_train[d:] - y_train[:-d])) if len(y_train) > d else (np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else 1.0)
    scale = max(scale, 1e-8)
    return np.mean(np.abs(y_true - y_pred)) / scale

# ---------- Customers ----------
def _ets_baseline_customers(hist: pd.DataFrame, H: int) -> pd.Series:
    """
    Robust ETS baseline:
    - <10 pts: repeat last-week mean (or last value)
    - <2*seasonal_period: trend-only ETS
    - else: additive trend + weekly seasonality
    """
    y = hist["customers"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    future_index = pd.date_range(hist.index.max() + pd.Timedelta(days=1), periods=H, freq="D")

    # Very short history → naive mean/last
    if len(y) < 10:
        base = (y.tail(7).mean() if len(y) >= 7 else y.iloc[-1])
        return pd.Series(base, index=future_index)

    seasonal_periods = 7
    try:
        if len(y) < 2 * seasonal_periods:
            # Not enough to estimate seasonality → trend-only
            model = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            )
        else:
            # Full model with weekly seasonality
            model = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        fit = model.fit(optimized=True, use_brute=True)
        fc = fit.forecast(H)
        fc.index = future_index
        return fc.clip(lower=0)
    except Exception:
        # Last-resort fallback: weekly repeat
        last_week = y.tail(7).values
        if len(last_week) == 0:
            last_week = np.array([0.0])
        reps = int(np.ceil(H / len(last_week)))
        vals = np.tile(last_week, reps)[:H]
        return pd.Series(vals, index=future_index).clip(lower=0)

def _recent_trend_multiplier(hist: pd.DataFrame, decay_lambda: float = 0.9) -> float:
    recent = hist.tail(28)
    if len(recent) < 8: return 1.0
    valid = (recent["customers"] / recent["customers"].shift(7).replace(0, np.nan)).dropna()
    if not len(valid): return 1.0
    w = np.array([decay_lambda ** i for i in range(len(valid)-1, -1, -1)], float); w /= w.sum()
    mult = float(np.sum(valid.values * w))
    return float(np.clip(mult, 0.85, 1.25))

def _weekday_growth_caps(hist: pd.DataFrame, fc_idx: pd.DatetimeIndex, base: pd.Series) -> pd.Series:
    out = base.copy()
    last_by_dow = hist.groupby(hist.index.dayofweek)["customers"].last()
    up, down = 1.40, 0.60
    for i in range(len(out)):
        last = last_by_dow.get(fc_idx[i].dayofweek, np.nan)
        if not np.isnan(last): out.iloc[i] = np.clip(out.iloc[i], last*down, last*up)
    return out.clip(lower=0)

def forecast_customers_with_trend_correction(
    hist: pd.DataFrame, future_cal: pd.DataFrame, H: int,
    decay_lambda: float = 0.9, apply_weekday_caps: bool = True,
    event_uplift_pct: Optional[pd.Series] = None, return_bands: bool = True,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    base = _ets_baseline_customers(hist, H)
    mult = _recent_trend_multiplier(hist, decay_lambda)
    fc = (base * mult).clip(lower=0)
    if event_uplift_pct is not None and len(event_uplift_pct) == H:
        fc = fc * (1.0 + (event_uplift_pct.values / 100.0))
    if apply_weekday_caps:
        fc = _weekday_growth_caps(hist, fc.index, fc)

    bands = None
    if return_bands:
        resid = (hist["customers"] - hist["customers"].shift(7)).dropna()
        if len(resid) >= 20:
            draws = np.random.choice(resid.values, size=(1000, H), replace=True)
            sims = np.clip(fc.values + draws, 0, None)
            bands = pd.DataFrame({"p10": np.percentile(sims,10,axis=0), "p50": fc.values, "p90": np.percentile(sims,90,axis=0)}, index=fc.index)
    return fc, bands

# ---------- ATV ----------
def _build_atv_feature_matrix(hist: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feats = [
        "dow","is_weekend","is_holiday","is_payday","is_payday_minus1","is_payday_plus1",
        "month","week",
        "atv_lag7","atv_lag14","atv_lag28",
        "atv_roll7_med","atv_roll14_med","atv_roll28_med",
        "customers_lag7","customers_lag14","customers_lag28",
    ]
    X = hist[feats].copy()
    y = hist["atv"].astype(float).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return X, y

def _train_direct_horizon_models(hist: pd.DataFrame, H: int):
    X_full, y_full = _build_atv_feature_matrix(hist)
    models, features = [], X_full.columns.tolist()
    for h in range(1, H+1):
        y_h = y_full.shift(-h)
        df_h = pd.concat([X_full, y_h.rename("y")], axis=1).dropna()
        if _HAS_LGBM and len(df_h) >= 100:
            model = lgb.LGBMRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=-1,
                subsample=0.85, colsample_bytree=0.85,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
        else:
            model = GradientBoostingRegressor(random_state=42)
        model.fit(df_h[features], df_h["y"]); models.append(model)
    return models, features

def _apply_guardrails_atv(hist: pd.DataFrame, preds: pd.Series, mad_mult: float) -> pd.Series:
    atv_hist = hist["atv"].astype(float)
    med = atv_hist.median(); mad = np.median(np.abs(atv_hist - med)) + 1e-6
    return preds.clip(lower=max(5.0, med - mad_mult*mad), upper=(med + mad_mult*mad))

def forecast_atv_direct(
    hist: pd.DataFrame, future_cal: pd.DataFrame, H: int,
    guardrail_mad_mult: float = 3.0, event_uplift_pct: Optional[pd.Series] = None,
    return_bands: bool = True,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    models, features = _train_direct_horizon_models(hist, H)
    last = hist.iloc[-1:].copy()
    Xf = future_cal.copy()
    for col in ["atv_lag7","atv_lag14","atv_lag28","atv_roll7_med","atv_roll14_med","atv_roll28_med","customers_lag7","customers_lag14","customers_lag28"]:
        Xf[col] = float(last[col].iloc[0]) if col in last.columns else np.nan
    Xf = Xf[features].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

    preds = []
    for i, m in enumerate(models):
        row = Xf.iloc[[i]] if i < len(Xf) else Xf.iloc[[-1]]
        preds.append(float(m.predict(row)[0]))
    preds = pd.Series(preds, index=future_cal.index, name="atv")

    if event_uplift_pct is not None and len(event_uplift_pct) == H:
        preds = preds * (1.0 + (event_uplift_pct.values / 100.0))

    preds = _apply_guardrails_atv(hist, preds, mad_mult=guardrail_mad_mult)

    bands = None
    if return_bands:
        X_hist, y_hist = _build_atv_feature_matrix(hist)
        proxy = GradientBoostingRegressor(random_state=42).fit(X_hist, y_hist)
        resid = y_hist - proxy.predict(X_hist)
        if len(resid) >= 30:
            draws = np.random.choice(resid.values, size=(1000, H), replace=True)
            sims = preds.values + draws
            bands = pd.DataFrame({"p10": np.percentile(sims,10,axis=0), "p50": preds.values, "p90": np.percentile(sims,90,axis=0)}, index=preds.index)
    return preds, bands

def combine_sales_and_bands(dates, customers, customers_bands, atv, atv_bands, return_bands=True):
    out = pd.DataFrame(index=dates)
    if return_bands and customers_bands is not None and atv_bands is not None:
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
        out["customers_p50"] = customers; out["atv_p50"] = atv; out["sales_p50"] = customers * atv
    return out

def backtest_metrics(hist: pd.DataFrame, horizon: int = 15, folds: int = 6) -> pd.DataFrame:
    cutoffs, sm_c, sm_a, sm_s, ms_c, ms_a, ms_s = [], [], [], [], [], [], []
    total_len = len(hist); min_train = 200 if total_len > 400 else int(total_len * 0.5)
    step = max(1, (total_len - min_train - horizon) // max(1, folds))
    for i in range(folds):
        train_end = min_train + i*step
        if train_end + horizon >= total_len: break
        train, test = hist.iloc[:train_end], hist.iloc[train_end:train_end+horizon]
        future_cal = test[["dow","is_weekend","is_holiday","is_payday","is_payday_minus1","is_payday_plus1","month","week"]].copy()
        cust_fc,_ = forecast_customers_with_trend_correction(train, future_cal, horizon, return_bands=False)
        atv_fc,_  = forecast_atv_direct(train, future_cal, horizon, return_bands=False)
        sales_fc = cust_fc * atv_fc
        sm_c.append(smape(test["customers"], cust_fc)); sm_a.append(smape(test["atv"], atv_fc)); sm_s.append(smape(test["sales"], sales_fc))
        ms_c.append(mase(test["customers"], cust_fc, train["customers"])); ms_a.append(mase(test["atv"], atv_fc, train["atv"])); ms_s.append(mase(test["sales"], sales_fc, train["sales"]))
        cutoffs.append(train.index.max())
    return pd.DataFrame({"cutoff": cutoffs, "sMAPE_customers": sm_c, "sMAPE_atv": sm_a, "sMAPE_sales": sm_s, "MASE_customers": ms_c, "MASE_atv": ms_a, "MASE_sales": ms_s})
