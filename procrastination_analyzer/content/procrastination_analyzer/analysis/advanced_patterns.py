
import pandas as pd
import numpy as np

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'ts' datetime column or a 'timestamp' column.")
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid timestamps after parsing.")
    return df

def gaps_hours(df: pd.DataFrame) -> pd.Series:
    df = ensure_ts(df)
    return df["ts"].diff().dt.total_seconds().fillna(0) / 3600

def time_of_day_bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late_night"

def burstiness_5in2h(df: pd.DataFrame) -> int:
    df = ensure_ts(df)
    if len(df) < 5:
        return 0
    count = 0
    for i in range(len(df) - 4):
        window = df.iloc[i:i+5]
        span_h = (window["ts"].iloc[-1] - window["ts"].iloc[0]).total_seconds() / 3600
        if span_h <= 2:
            count += 1
    return int(count)

def long_gaps_count(df: pd.DataFrame, threshold_h: float = 24.0) -> int:
    df = ensure_ts(df)
    g = gaps_hours(df)
    return int((g > threshold_h).sum())

def late_night_ratio(df: pd.DataFrame) -> float:
    df = ensure_ts(df)
    return float((df["ts"].dt.hour >= 23).mean())

def weekend_ratio(df: pd.DataFrame) -> float:
    df = ensure_ts(df)
    return float((df["ts"].dt.weekday >= 5).mean())

def hour_variance(df: pd.DataFrame) -> float:
    df = ensure_ts(df)
    v = df["ts"].dt.hour.var()
    return float(0.0 if np.isnan(v) else v)

def detect_perfectionism(df: pd.DataFrame) -> float:
    df = ensure_ts(df)
    n = len(df)

    lg = long_gaps_count(df, 24)
    burst = burstiness_5in2h(df)
    late = late_night_ratio(df)

    volume_norm = max(1.0, n / 25.0)
    raw = (0.55 * (lg / volume_norm)) + (0.35 * (burst / volume_norm)) + (0.10 * late * 5)
    score = float(np.clip(raw, 0.0, 1.0))
    return round(score, 2)

def detect_pattern(df: pd.DataFrame) -> str:
    df = ensure_ts(df)
    hours = df["ts"].dt.hour

    day = int(((hours >= 9) & (hours < 18)).sum())
    evening = int(((hours >= 18) & (hours < 23)).sum())
    late = int((hours >= 23).sum())
    total = max(1, len(df))

    lg = long_gaps_count(df, 24)
    burst = burstiness_5in2h(df)

    late_pct = late / total
    evening_pct = evening / total
    day_pct = day / total

    if lg >= 2 and late_pct >= 0.35:
        return "Avoidance-driven procrastination"
    if evening_pct >= 0.45 and late_pct < 0.25:
        return "Fatigue-driven procrastination"
    if burst >= 2 and (late_pct + evening_pct) >= 0.60:
        return "Deadline-chasing (bursty) procrastination"
    if day_pct >= 0.55 and lg == 0:
        return "Consistent / low-procrastination pattern"
    return "Mixed / situational procrastination"

def predict_risk(df: pd.DataFrame) -> float:
    df = ensure_ts(df)
    last_ts = df["ts"].max()
    last_week = df[df["ts"] >= last_ts - pd.Timedelta(days=7)]
    if last_week.empty:
        return 0.5

    inactivity_h = float((last_ts - last_week["ts"].max()).total_seconds() / 3600)
    hv = hour_variance(last_week)
    perf = detect_perfectionism(df)
    wknd = weekend_ratio(last_week)

    raw = 0.18 + 0.02 * inactivity_h + 0.015 * hv + 0.45 * perf + 0.10 * wknd
    risk = float(np.clip(raw, 0.0, 1.0))
    return round(risk, 2)

def build_explainability(df: pd.DataFrame) -> dict:
    df = ensure_ts(df).copy()
    df["hour"] = df["ts"].dt.hour
    df["bucket"] = df["hour"].apply(time_of_day_bucket)

    total = int(len(df))
    peak_hour = int(df["hour"].value_counts().idxmax()) if total else None

    g = gaps_hours(df)
    return {
        "total_events": total,
        "peak_hour": peak_hour,
        "bucket_counts": df["bucket"].value_counts().to_dict(),
        "late_night_ratio": round(late_night_ratio(df), 2),
        "weekend_ratio": round(weekend_ratio(df), 2),
        "hour_variance": round(hour_variance(df), 2),
        "long_gaps_24h": int((g > 24).sum()),
        "long_gaps_48h": int((g > 48).sum()),
        "max_gap_hours": round(float(g.max()) if len(g) else 0.0, 2),
        "burst_count_5in2h": int(burstiness_5in2h(df)),
    }
