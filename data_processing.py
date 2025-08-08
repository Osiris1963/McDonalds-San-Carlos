import numpy as np
import pandas as pd
from datetime import datetime, date
import holidays
from dateutil.relativedelta import relativedelta

PH_HOLIDAYS = holidays.country_holidays("PH")

REQ_BASE_COLS = ["date"]
OPTIONAL_COLS = ["sales", "customers", "atv", "weather", "event_flag", "notes"]

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Some dates could not be parsed. Please clean your 'date' column.")
    out = out.sort_values("date").drop_duplicates("date")
    out = out.set_index("date")
    return out

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek  # 0=Mon
    out["dom"] = idx.day
    out["week"] = idx.isocalendar().week.astype(int)
    out["month"] = idx.month
    out["year"] = idx.year
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Paydays (15th and end-of-month); plus ±1 day effect
    out["is_payday"] = ((out["dom"] == 15) | (idx == (idx + pd.offsets.MonthEnd(0)))).astype(int)
    out["is_payday_minus1"] = ((idx + pd.Timedelta(days=1)).day == 15).astype(int) | ((idx + pd.Timedelta(days=1)) == ((idx + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1))).astype(int)
    out["is_payday_plus1"]  = ((idx - pd.Timedelta(days=1)).day == 15).astype(int) | ((idx - pd.Timedelta(days=1)) == ((idx + pd.offsets.MonthEnd(0)) - pd.Timedelta(days=1))).astype(int)

    # PH holidays
    out["is_holiday"] = [int(d in PH_HOLIDAYS) for d in idx.date]
    return out

def _compute_atv_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_sales = "sales" in out.columns
    has_customers = "customers" in out.columns
    has_atv = "atv" in out.columns
    if has_sales and has_customers:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["atv"] = out["sales"] / out["customers"]
    elif has_atv and has_customers and not has_sales:
        out["sales"] = out["atv"] * out["customers"]
    elif has_atv and has_sales and not has_customers:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["customers"] = np.where(out["atv"] > 0, out["sales"] / out["atv"], np.nan)

    # sanitize
    if "customers" in out.columns:
        out["customers"] = out["customers"].replace([np.inf, -np.inf], np.nan)
    if "atv" in out.columns:
        out["atv"] = out["atv"].replace([np.inf, -np.inf], np.nan)
    if "sales" in out.columns:
        out["sales"] = out["sales"].replace([np.inf, -np.inf], np.nan)
    return out

def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ensure expected numeric cols exist if provided
    for c in ["sales", "customers", "atv"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = _compute_atv_if_missing(out)

    # Basic validity checks
    if "customers" in out.columns and out["customers"].notna().sum() == 0:
        raise ValueError("No valid 'customers' found/derivable.")
    if "atv" in out.columns and out["atv"].notna().sum() == 0:
        raise ValueError("No valid 'atv' found/derivable.")
    if "sales" in out.columns and out["sales"].notna().sum() == 0:
        # If sales entirely missing, compute now that we (likely) have both customers and atv
        out["sales"] = out["customers"] * out["atv"]

    out = _add_calendar_features(out)

    # Lags/rolling (for ATV modeling mostly)
    for lag in [7, 14, 28]:
        out[f"customers_lag{lag}"] = out["customers"].shift(lag)
        out[f"atv_lag{lag}"] = out["atv"].shift(lag)
    out["atv_roll7_med"] = out["atv"].rolling(7, min_periods=3).median()
    out["atv_roll14_med"] = out["atv"].rolling(14, min_periods=5).median()
    out["atv_roll28_med"] = out["atv"].rolling(28, min_periods=7).median()

    out = out.dropna().copy()  # safe for modeling
    return out

def add_future_calendar(hist: pd.DataFrame, periods: int) -> pd.DataFrame:
    last_date = hist.index.max()
    future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    future = pd.DataFrame(index=future_idx)
    future = _add_calendar_features(future)
    return future

def build_event_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    ev = events_df.copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    if ev["date"].isna().any():
        raise ValueError("Event dates contain invalid entries.")
    ev = ev.set_index("date").sort_index()
    # Enforce columns
    for c in ["uplift_customers_pct", "uplift_atv_pct"]:
        if c not in ev.columns:
            ev[c] = 0.0
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0.0)
    if "notes" not in ev.columns:
        ev["notes"] = ""
    return ev

def pretty_money(x: float) -> str:
    try:
        return f"₱{x:,.2f}"
    except Exception:
        return str(x)
