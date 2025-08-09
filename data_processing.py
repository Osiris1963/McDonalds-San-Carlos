# data_processing.py
from typing import Optional
import pandas as pd
import numpy as np

# ---------------------------
# Firestore -> DataFrame
# ---------------------------
def load_from_firestore(db_client, collection_name: str) -> pd.DataFrame:
    """
    Loads and preprocesses data from a Firestore collection, ensuring no duplicate dates.
    Expected doc fields (typical): date (str/ts), sales (float), customers (int),
    add_on_sales (float, optional), day_type (str, optional), day_type_notes (str, optional).
    """
    if db_client is None:
        return pd.DataFrame()

    docs = db_client.collection(collection_name).stream()
    records = []
    for doc in docs:
        d = doc.to_dict() or {}
        d["doc_id"] = doc.id
        records.append(d)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Normalize date
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Basic cols
    for col in ["sales", "customers", "add_on_sales"]:
        if col not in df.columns:
            df[col] = 0.0 if col != "customers" else 0

    # De-duplicate by date (keep latest)
    df = df.sort_values(["date"]).drop_duplicates(subset=["date"], keep="last")

    # Type safety
    df["customers"] = pd.to_numeric(df["customers"], errors="coerce").fillna(0).astype(int)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    df["add_on_sales"] = pd.to_numeric(df["add_on_sales"], errors="coerce").fillna(0.0)

    # Guard for negatives
    df["customers"] = df["customers"].clip(lower=0)
    df["sales"] = df["sales"].clip(lower=0)
    df["add_on_sales"] = df["add_on_sales"].clip(lower=0)

    return df.sort_values("date").reset_index(drop=True)

# ---------------------------
# Feature Engineering
# ---------------------------
def _cyclical_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    angle = 2 * np.pi * series / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(angle),
        f"{prefix}_cos": np.cos(angle),
    }, index=series.index)

def _safe_div(n, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d == 0, np.nan, n / d)
    return pd.Series(out)

def _rolling_shifted(s: pd.Series, window: int, shift_by: int = 1, fn: str = "mean"):
    if fn == "mean":
        return s.shift(shift_by).rolling(window).mean()
    if fn == "std":
        return s.shift(shift_by).rolling(window).std()
    if fn == "sum":
        return s.shift(shift_by).rolling(window).sum()
    return s.shift(shift_by).rolling(window).mean()

def _add_weekend_payday_flags(df: pd.DataFrame):
    dow = df["date"].dt.dayofweek
    df["is_weekend"] = (dow >= 5).astype(int)
    # Simple payday heuristic: 15th & 30th/31st, plus nearest Friday if weekend
    d = df["date"].dt.day
    payday = (d.isin([15, 30, 31])).astype(int)
    friday = (dow == 4)
    payday |= ((d.isin([14, 16, 29])) & friday).astype(int)
    df["is_paydayish"] = payday
    return df

def create_advanced_features(df: pd.DataFrame, events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a robust feature set for customers + ATV models.
    Assumes df has columns: date, sales, customers, add_on_sales (optional), day_type/notes (optional).
    Returns a DataFrame with features + targets (customers, atv).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Base and ATV (base = sales minus add-ons)
    if "add_on_sales" not in df.columns:
        df["add_on_sales"] = 0.0
    base_sales = (df["sales"] - df["add_on_sales"]).clip(lower=0)
    customers_safe = df["customers"].replace(0, np.nan)
    df["atv"] = _safe_div(base_sales, customers_safe)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Cyclical encodings
    dow_cyc = _cyclical_encode(df["dayofweek"], 7, "dow")
    moy_cyc = _cyclical_encode(df["month"], 12, "moy")
    df = pd.concat([df, dow_cyc, moy_cyc], axis=1)

    # Lags (shifted to avoid leakage)
    for col in ["customers", "sales"]:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_2"] = df[col].shift(2)
        df[f"{col}_lag_7"] = df[col].shift(7)
        df[f"{col}_lag_14"] = df[col].shift(14)

    # Rolling stats (shifted by 1)
    def _roll(col): return df[col]
    df["customers_rm7"] = _rolling_shifted(_roll("customers"), 7, 1, "mean")
    df["customers_rm14"] = _rolling_shifted(_roll("customers"), 14, 1, "mean")
    df["customers_rm28"] = _rolling_shifted(_roll("customers"), 28, 1, "mean")
    df["sales_rm7"] = _rolling_shifted(_roll("sales"), 7, 1, "mean")
    df["sales_rm14"] = _rolling_shifted(_roll("sales"), 14, 1, "mean")
    df["sales_rm28"] = _rolling_shifted(_roll("sales"), 28, 1, "mean")
    df["sales_rstd14"] = _rolling_shifted(_roll("sales"), 14, 1, "std")
    df["customers_rstd14"] = _rolling_shifted(_roll("customers"), 14, 1, "std")

    # Flags
    df = _add_weekend_payday_flags(df)

    # Events integration (binary)
    if events_df is not None and not events_df.empty and "date" in events_df.columns:
        e = events_df.copy()
        e["date"] = pd.to_datetime(e["date"]).dt.normalize()
        e = e.drop_duplicates(subset=["date"], keep="first")
        df = df.merge(e[["date"]], on="date", how="left", indicator=True)
        df["is_event"] = (df["_merge"] == "both").astype(int)
        df.drop(columns=["_merge"], inplace=True)
    else:
        df["is_event"] = 0

    # Day-type
    if "day_type" in df.columns:
        df["is_not_normal_day"] = (df["day_type"].astype(str) == "Not Normal Day").astype(int)
    else:
        df["is_not_normal_day"] = 0

    # Fill NaNs
    df = df.fillna(0)

    return df
