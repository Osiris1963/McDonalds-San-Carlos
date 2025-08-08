import pandas as pd
import numpy as np
import holidays

PH_HOLIDAYS = holidays.country_holidays("PH")

def load_history_from_firestore(db, collection: str, raw: bool = False) -> pd.DataFrame:
    docs = list(db.collection(collection).stream())
    rows = []
    for d in docs:
        rec = d.to_dict() or {}
        rec_date = rec.get("date", d.id)
        if hasattr(rec_date, "to_datetime"): rec_date = rec_date.to_datetime()
        if hasattr(rec_date, "isoformat"):   rec_date = rec_date.strftime("%Y-%m-%d")
        rec["date"] = rec_date; rec["doc_id"] = d.id; rows.append(rec)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if raw or df.empty: return df
    df.columns = [c.lower() for c in df.columns]
    return df

def load_events_from_firestore(db, collection: str, horizon: int, start_date) -> pd.DataFrame:
    docs = list(db.collection(collection).stream())
    rows = []
    for d in docs:
        rec = d.to_dict() or {}
        rec_date = rec.get("date", d.id)
        if hasattr(rec_date, "to_datetime"): rec_date = rec_date.to_datetime()
        if hasattr(rec_date, "isoformat"):   rec_date = rec_date.strftime("%Y-%m-%d")
        rows.append({
            "date": rec_date,
            "uplift_customers_pct": float(rec.get("uplift_customers_pct", 0) or 0),
            "uplift_atv_pct": float(rec.get("uplift_atv_pct", 0) or 0),
            "notes": rec.get("notes", ""),
        })
    ev = pd.DataFrame(rows)
    if ev.empty:
        idx = pd.date_range(start=pd.to_datetime(start_date), periods=horizon, freq="D")
        return pd.DataFrame({"date": idx, "uplift_customers_pct": 0.0, "uplift_atv_pct": 0.0, "notes": ""})
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).sort_values("date")
    return ev

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    out = out.set_index("date")
    return out

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); idx = out.index
    out["dow"] = idx.dayofweek
    out["dom"] = idx.day
    out["week"] = idx.isocalendar().week.astype(int)
    out["month"] = idx.month
    out["year"] = idx.year
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = [int(d in PH_HOLIDAYS) for d in idx.date]
    out["is_payday"] = ((out["dom"] == 15) | (idx == (idx + pd.offsets.MonthEnd(0)))).astype(int)
    out["is_payday_minus1"] = ((idx + pd.Timedelta(days=1)).day == 15).astype(int) | ((idx + pd.Timedelta(days=1)) == ((idx + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1))).astype(int)
    out["is_payday_plus1"]  = ((idx - pd.Timedelta(days=1)).day == 15).astype(int) | ((idx - pd.Timedelta(days=1)) == ((idx + pd.offsets.MonthEnd(0)) - pd.Timedelta(days=1))).astype(int)
    return out

def _compute_atv_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sales" in out.columns and "customers" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["atv"] = out["sales"] / out["customers"].replace(0, np.nan)
    elif "atv" in out.columns and "customers" in out.columns and "sales" not in out.columns:
        out["sales"] = out["atv"] * out["customers"]
    elif "sales" in out.columns and "atv" in out.columns and "customers" not in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["customers"] = out["sales"] / out["atv"].replace(0, np.nan)
    for c in ("customers","sales","atv"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
    return out

def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _compute_atv_if_missing(out)
    out = _add_calendar_features(out)
    for lag in [7,14,28]:
        if "customers" in out.columns: out[f"customers_lag{lag}"] = out["customers"].shift(lag)
        if "atv" in out.columns: out[f"atv_lag{lag}"] = out["atv"].shift(lag)
    out["atv_roll7_med"]  = out["atv"].rolling(7,  min_periods=3).median()
    out["atv_roll14_med"] = out["atv"].rolling(14, min_periods=5).median()
    out["atv_roll28_med"] = out["atv"].rolling(28, min_periods=7).median()
    return out.dropna().copy()

def add_future_calendar(hist: pd.DataFrame, periods: int) -> pd.DataFrame:
    last = hist.index.max()
    idx = pd.date_range(start=last + pd.Timedelta(days=1), periods=periods, freq="D")
    fut = pd.DataFrame(index=idx)
    fut = _add_calendar_features(fut)
    return fut

def build_event_frame_from_df(ev_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    # Robust to duplicate dates: aggregate then align.
    ev = pd.DataFrame(index=index)
    ev["uplift_customers_pct"] = 0.0
    ev["uplift_atv_pct"] = 0.0
    if ev_df is None or ev_df.empty:
        return ev
    tmp = ev_df.copy()
    tmp["date"] = pd.to_datetime(tmp.get("date"), errors="coerce")
    for c in ("uplift_customers_pct", "uplift_atv_pct"):
        if c not in tmp.columns: tmp[c] = 0.0
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
    tmp = tmp.dropna(subset=["date"])
    tmp = tmp.groupby("date", as_index=True)[["uplift_customers_pct","uplift_atv_pct"]].sum().sort_index()
    aligned = tmp.reindex(index, fill_value=0.0)
    ev["uplift_customers_pct"] = aligned["uplift_customers_pct"].values
    ev["uplift_atv_pct"] = aligned["uplift_atv_pct"].values
    return ev

def pretty_money(x: float) -> str:
    try: return f"₱{x:,.2f}"
    except Exception: return str(x)
