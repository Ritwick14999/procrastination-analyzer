"""Scale-invariant behavioural feature extraction.

Design rules, all of which the original implementation violated somewhere:

1. **Every feature is scale-invariant.** A feature must describe a *rate* or a
   *share*, never a raw count. Otherwise a user who logs six months of data
   scores worse than an identically-behaved user who logs one month, which was
   the single worst defect in the original scoring code.
2. **Sessions are segmented once, then reused.** Bursts are counted over
   disjoint sessions rather than sliding windows, so one long sitting counts as
   one session instead of ``n - 4`` overlapping "bursts".
3. **Weekends are discounted from gaps.** A Friday-evening to Monday-morning
   silence is normal, not avoidance, so gap length is measured in *active*
   hours with weekend hours removed.
4. **No feature reads a value it cannot see.** Notably, "hours since last
   activity" is measured against the analysis reference time, not against the
   last event, which is trivially itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalyzerConfig
from .schema import EventFrame

__all__ = ["BehaviouralFeatures", "extract_features", "segment_sessions", "FEATURE_NAMES"]


@dataclass(frozen=True)
class BehaviouralFeatures:
    """Scale-invariant descriptors of a single user's activity record.

    All ``*_share`` fields are in [0, 1]. All ``*_rate`` fields are per-active-day
    or per-gap and are unbounded above but non-negative.
    """

    # --- Volume / coverage ---------------------------------------------------
    n_events: int
    span_days: float
    active_days: int
    #: Fraction of calendar days in the span that saw any activity. A proxy for
    #: consistency that does not care how long the record is.
    coverage: float
    #: Mean events on days that had activity.
    events_per_active_day: float

    # --- Gap structure -------------------------------------------------------
    #: Share of inter-event gaps exceeding the long-gap threshold.
    long_gap_rate: float
    #: Share of inter-event gaps exceeding the extended threshold.
    extended_gap_rate: float
    #: Longest weekend-adjusted gap, in hours.
    max_gap_hours: float
    #: Median weekend-adjusted gap, in hours. Robust to a single outlier.
    median_gap_hours: float

    # --- Session / burst structure -------------------------------------------
    n_sessions: int
    #: Share of *events* that occurred inside a burst session. Using events
    #: rather than session count keeps this comparable across record lengths.
    burst_event_share: float
    #: Mean session length in hours.
    mean_session_hours: float
    #: Mean events per session.
    mean_session_events: float

    # --- Circadian structure -------------------------------------------------
    late_night_share: float
    evening_share: float
    working_hours_share: float
    weekend_share: float
    #: Circular standard deviation of activity hour, normalised to [0, 1].
    #: 0 = every event at the same clock time, 1 = uniformly spread.
    rhythm_irregularity: float
    peak_hour: int

    # --- Trend ---------------------------------------------------------------
    #: Ratio of activity in the final third of the record to the first third.
    #: >1 means ramping up (deadline chasing), <1 means fading out.
    recency_trend: float
    #: Hours between the analysis reference time and the last event.
    hours_since_last_event: float

    # --- Meta ----------------------------------------------------------------
    #: False when the record is too short for the numbers to mean much.
    is_confident: bool

    def to_dict(self) -> dict[str, float]:
        """Flat dict, suitable for JSON, reports, or a model feature vector."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Ordered numeric vector for the ML models. Order is ``FEATURE_NAMES``."""
        return np.array([float(getattr(self, name)) for name in FEATURE_NAMES], dtype=float)


#: The subset of features fed to the ML models, in a fixed order. Excludes raw
#: volume terms (``n_events``, ``span_days``, ``active_days``) precisely because
#: those are the scale-dependent ones we do not want the model leaning on.
FEATURE_NAMES = (
    "coverage",
    "events_per_active_day",
    "long_gap_rate",
    "extended_gap_rate",
    "max_gap_hours",
    "median_gap_hours",
    "burst_event_share",
    "mean_session_hours",
    "mean_session_events",
    "late_night_share",
    "evening_share",
    "working_hours_share",
    "weekend_share",
    "rhythm_irregularity",
    "recency_trend",
    # A duration, not a count, so it stays scale-invariant. The heuristic weights
    # this heavily, so withholding it from the models would rig the comparison.
    "hours_since_last_event",
)


def _weekend_adjusted_gaps(ts: pd.Series, config: AnalyzerConfig) -> np.ndarray:
    """Inter-event gaps in hours, optionally discounting weekend hours.

    A gap that runs from Friday 18:00 to Monday 09:00 spans 63 clock hours but
    only ~15 hours of ordinary working time. Counting the raw 63 would flag
    every Monday-to-Friday professional as a procrastinator, which is exactly
    what the original ``long_gaps_count`` did.
    """
    if len(ts) < 2:
        return np.zeros(0, dtype=float)

    starts = ts.iloc[:-1].to_numpy()
    ends = ts.iloc[1:].to_numpy()
    raw_hours = (ends - starts) / np.timedelta64(1, "h")

    if not config.exclude_weekend_from_gaps:
        return raw_hours.astype(float)

    # Count weekend hours inside each interval by walking day boundaries. Gaps
    # are typically short, so we only do the expensive path for long ones.
    adjusted = raw_hours.astype(float).copy()
    long_enough = np.flatnonzero(adjusted > 12.0)

    for i in long_enough:
        start = pd.Timestamp(starts[i])
        end = pd.Timestamp(ends[i])
        weekend_hours = 0.0
        # Iterate day-by-day over the interval, clipping to the gap boundaries.
        day = start.normalize()
        while day <= end:
            if day.weekday() >= 5:
                overlap_start = max(start, day)
                overlap_end = min(end, day + pd.Timedelta(days=1))
                if overlap_end > overlap_start:
                    weekend_hours += (overlap_end - overlap_start).total_seconds() / 3600.0
            day += pd.Timedelta(days=1)
        adjusted[i] = max(0.0, adjusted[i] - weekend_hours)

    return adjusted


def segment_sessions(events: EventFrame, config: AnalyzerConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Group consecutive events into disjoint working sessions.

    Events separated by more than ``config.session_timeout_h`` start a new
    session. Returns one row per session with its start, end, span and size.

    This replaces the original sliding-window burst counter, which reported 16
    "bursts" for a single 20-event sitting because it counted every overlapping
    5-event window.
    """
    ts = events.ts
    if len(ts) == 0:
        return pd.DataFrame(columns=["start", "end", "n_events", "span_hours", "is_burst"])

    gap_h = ts.diff().dt.total_seconds().to_numpy() / 3600.0
    gap_h[0] = np.inf  # first event always opens a session
    session_id = np.cumsum(gap_h > config.session_timeout_h) - 1

    grouped = pd.DataFrame({"ts": ts.to_numpy(), "session": session_id}).groupby("session")["ts"]
    sessions = pd.DataFrame(
        {
            "start": grouped.min(),
            "end": grouped.max(),
            "n_events": grouped.size(),
        }
    ).reset_index(drop=True)

    sessions["span_hours"] = (sessions["end"] - sessions["start"]).dt.total_seconds() / 3600.0
    sessions["is_burst"] = (sessions["n_events"] >= config.burst_min_events) & (
        sessions["span_hours"] <= config.burst_max_span_h
    )
    return sessions


def _circular_irregularity(hours: pd.Series) -> float:
    """Circular dispersion of clock hours, normalised to [0, 1].

    Ordinary variance is wrong for clock time: 23:00 and 01:00 are two hours
    apart, not twenty-two. The original ``hour_variance`` used plain variance
    and so treated a rock-steady midnight worker as maximally erratic. This maps
    each hour onto the unit circle and measures the resultant vector length.
    """
    if len(hours) == 0:
        return 0.0
    angles = 2.0 * np.pi * hours.to_numpy(dtype=float) / 24.0
    resultant = np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


def _recency_trend(ts: pd.Series) -> float:
    """Ratio of activity in the last third of the record to the first third."""
    n = len(ts)
    if n < 6:
        return 1.0
    span = ts.iloc[-1] - ts.iloc[0]
    if span.total_seconds() <= 0:
        return 1.0
    third = span / 3
    first_cut = ts.iloc[0] + third
    last_cut = ts.iloc[-1] - third
    first_third = int((ts <= first_cut).sum())
    last_third = int((ts >= last_cut).sum())
    if first_third == 0:
        return float(last_third) if last_third else 1.0
    return float(last_third / first_third)


def extract_features(
    events: EventFrame,
    config: AnalyzerConfig = DEFAULT_CONFIG,
    *,
    reference_time: pd.Timestamp | None = None,
) -> BehaviouralFeatures:
    """Compute the full scale-invariant feature set in a single pass.

    Args:
        events: A validated event frame.
        config: Thresholds to apply.
        reference_time: "Now" for recency features. Defaults to the last event,
            which makes ``hours_since_last_event`` zero — correct when analysing
            a closed historical record, but callers scoring a *live* user should
            pass the real current time.
    """
    ts = events.ts
    n = len(ts)
    hours = ts.dt.hour
    weekday = ts.dt.weekday

    ref = reference_time if reference_time is not None else ts.iloc[-1]
    hours_since_last = max(0.0, (ref - ts.iloc[-1]).total_seconds() / 3600.0)

    gaps = _weekend_adjusted_gaps(ts, config)
    n_gaps = max(1, len(gaps))

    sessions = segment_sessions(events, config)
    burst_events = int(sessions.loc[sessions["is_burst"], "n_events"].sum())

    span_days = events.span_days
    active_days = events.active_days
    # +1 because a record spanning 0 days still covers one calendar day.
    calendar_days = max(1.0, span_days + 1.0)

    return BehaviouralFeatures(
        n_events=n,
        span_days=round(span_days, 3),
        active_days=active_days,
        coverage=round(float(np.clip(active_days / calendar_days, 0.0, 1.0)), 4),
        events_per_active_day=round(float(n / max(1, active_days)), 3),
        long_gap_rate=round(float((gaps > config.long_gap_h).sum() / n_gaps), 4),
        extended_gap_rate=round(float((gaps > config.extended_gap_h).sum() / n_gaps), 4),
        max_gap_hours=round(float(gaps.max()) if len(gaps) else 0.0, 2),
        median_gap_hours=round(float(np.median(gaps)) if len(gaps) else 0.0, 2),
        n_sessions=int(len(sessions)),
        burst_event_share=round(float(burst_events / n), 4),
        mean_session_hours=round(float(sessions["span_hours"].mean()), 3),
        mean_session_events=round(float(sessions["n_events"].mean()), 3),
        # Late night wraps around midnight: 00:30 is late night, not "morning".
        # Testing only ``hour >= 23`` (as the original did) silently classifies
        # every after-midnight worker as having zero late-night activity.
        late_night_share=round(
            float(((hours >= config.late_night_hour) | (hours < config.morning_start)).mean()),
            4,
        ),
        evening_share=round(
            float(((hours >= config.evening_start) & (hours < config.night_start)).mean()), 4
        ),
        working_hours_share=round(float(((hours >= 9) & (hours < 18)).mean()), 4),
        weekend_share=round(float((weekday >= 5).mean()), 4),
        rhythm_irregularity=round(_circular_irregularity(hours), 4),
        peak_hour=int(hours.value_counts().idxmax()),
        recency_trend=round(_recency_trend(ts), 3),
        hours_since_last_event=round(hours_since_last, 2),
        is_confident=n >= config.min_events_for_confidence,
    )
