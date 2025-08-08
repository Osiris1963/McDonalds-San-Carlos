# data_processing.py — Firestore I/O + feature engineering

import pandas as pd
import numpy as np

def load_from_firestore(db_client, collection_name: str) -> pd.DataFrame:
    """Load a collection from Firestore into a clean DataFrame with a proper date index."""
    if db_client is None:
        return pd.DataFrame()

    docs = list(db_client.collection(collection_name).stream())
    rows = []
    for d in docs:
        rec = d.to_dict() or {}
        # prefer explicit 'date' field; else use doc id
        rec_date = rec.get("date", d.id)
        # Firestore Timestamp → datetime
        if hasattr(rec_date, "to_datetime"):
            rec_date = rec_date.to_datetime()
        # datetime/date → string
        if hasattr(rec_date, "isoformat"):
            rec_date = rec_date.strftime("%Y-%m-%d")
        rec["date"] = rec_date
        rec["doc_id"] = d.id
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def create_advanced_features(df: pd.DataFrame, events_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Feature builder used for both training and forecasting.
    - computes atv if sales+customers present
    - calendar, payday, weekend, cyclical features
    - weather dummies (if present)
    - lags & rolling stats for customers/atv
    - event flag (from events_df)
    - day_type guard (is_not_normal_day if provided)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # numeric casts
    for col in ("customers", "sales", "atv", "add_on_sales"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # compute atv if possible — remove add_on_sales from base
    if "atv" not in out.columns or out["atv"].isna().all():
        if "sales" in out.columns and "customers" in out.columns:
            base_sales = out["sales"] - out.get("add_on_sales", 0)
            cust = out["customers"].replace(0, np.nan)
            out["atv"] = base_sales / cust

    # calendar
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date")
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["year"] = out["date"].dt.year
    out["time_idx"] = (out["date"] - out["date"].min()).dt.days

    # payday / weekend interactions
    out["is_payday_period"] = out["day"].isin([14,15,16,29,30,31,1,2]).astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["payday_weekend_interaction"] = out["is_payday_period"] * out["is_weekend"]

    # weather dummies if present
    if "weather" in out.columns:
        out["weather"] = out["weather"].fillna("Unknown").astype(str)
        wd = pd.get_dummies(out["weather"], prefix="weather", dtype=int)
        out = pd.concat([out, wd], axis=1)

    # lags & rolling
    for lag in (7, 14, 28):
        if "customers" in out.columns:
            out[f"customers_lag{lag}"] = out["customers"].shift(lag)
        if "atv" in out.columns:
            out[f"atv_lag{lag}"] = out["atv"].shift(lag)

    for w in (7, 14):
        if "customers" in out.columns:
            out[f"customers_roll_mean_{w}"] = out["customers"].shift(1).rolling(w, min_periods=1).mean()
            out[f"customers_roll_std_{w}"]  = out["customers"].shift(1).rolling(w, min_periods=1).std()
        if "atv" in out.columns:
            out[f"atv_roll_mean_{w}"] = out["atv"].shift(1).rolling(w, min_periods=1).mean()
            out[f"atv_roll_std_{w}"]  = out["atv"].shift(1).rolling(w, min_periods=1).std()

    # cyclical encodings
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["weekofyear_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52)
    out["weekofyear_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52)

    # events flag
    if events_df is not None and not events_df.empty:
        ev = events_df.copy()
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.normalize()
        ev = ev.dropna(subset=["date"]).drop_duplicates(subset=["date"])
        out = out.merge(ev[["date"]], on="date", how="left", indicator=True)
        out["is_event"] = (out["_merge"] == "both").astype(int)
        out = out.drop(columns=["_merge"])
    else:
        out["is_event"] = 0

    # day_type
    if "day_type" in out.columns:
        out["is_not_normal_day"] = (out["day_type"].astype(str) == "Not Normal Day").astype(int)
    else:
        out["is_not_normal_day"] = 0

    return out.fillna(0)
