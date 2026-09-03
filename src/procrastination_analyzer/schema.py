"""Canonical event-frame construction and validation.

The original code called ``ensure_ts`` at the top of every function, so a single
call to ``detect_perfectionism`` re-parsed, re-copied and re-sorted the frame
five times over. Here the normalisation happens exactly once, at the boundary,
and everything downstream takes an already-validated :class:`EventFrame`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

__all__ = ["EventFrame", "TIMESTAMP_COLUMN", "build_event_frame", "InsufficientDataError"]

TIMESTAMP_COLUMN = "ts"

#: Column names accepted as the source of timestamps, in priority order.
_CANDIDATE_COLUMNS: Sequence[str] = ("ts", "timestamp", "time", "date", "datetime", "created_at")


class InsufficientDataError(ValueError):
    """Raised when there are too few usable timestamps to analyse."""


@dataclass(frozen=True)
class EventFrame:
    """A validated, sorted, timezone-naive series of activity timestamps.

    Construct via :func:`build_event_frame` rather than directly, so that the
    invariants below are guaranteed:

    * ``frame`` has a ``ts`` column of dtype datetime64[ns]
    * it is sorted ascending, with no NaT values
    * the index is a clean RangeIndex
    """

    frame: pd.DataFrame
    #: Rows dropped because the timestamp could not be parsed.
    dropped_unparseable: int = 0
    #: Rows dropped as exact duplicate timestamps.
    dropped_duplicates: int = 0

    @property
    def ts(self) -> pd.Series:
        """The timestamp column."""
        return self.frame[TIMESTAMP_COLUMN]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def span_days(self) -> float:
        """Calendar days between the first and last event."""
        if len(self.frame) < 2:
            return 0.0
        delta = self.ts.iloc[-1] - self.ts.iloc[0]
        return float(delta.total_seconds() / 86400.0)

    @property
    def active_days(self) -> int:
        """Number of distinct calendar dates with at least one event."""
        return int(self.ts.dt.normalize().nunique())

    def last_n_days(self, days: float) -> EventFrame:
        """Return a new frame restricted to the final ``days`` of the record."""
        if self.frame.empty:
            return self
        cutoff = self.ts.max() - pd.Timedelta(days=days)
        subset = self.frame[self.ts >= cutoff].reset_index(drop=True)
        return EventFrame(subset)


def _coerce_to_series(
    data: pd.DataFrame | pd.Series | Iterable[object],
    column: str | None,
) -> pd.Series:
    """Pull the timestamp column out of whatever the caller handed us."""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if column is not None:
            if column not in data.columns:
                raise KeyError(f"Column {column!r} not found. Available: {list(data.columns)}")
            return data[column]
        for candidate in _CANDIDATE_COLUMNS:
            if candidate in data.columns:
                return data[candidate]
        # Single-column frames are unambiguous, so accept them whatever the name.
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise KeyError(
            "Could not find a timestamp column. Expected one of "
            f"{list(_CANDIDATE_COLUMNS)}, got {list(data.columns)}. "
            "Pass column= to disambiguate."
        )
    return pd.Series(list(data))


def build_event_frame(
    data: pd.DataFrame | pd.Series | Iterable[object],
    column: str | None = None,
    *,
    min_events: int = 3,
    drop_duplicates: bool = True,
    tz_convert: str | None = None,
) -> EventFrame:
    """Normalise arbitrary timestamp input into a validated :class:`EventFrame`.

    Args:
        data: A DataFrame, Series, or any iterable of timestamp-like values.
        column: Explicit column name to read from a DataFrame. Auto-detected
            when omitted.
        min_events: Minimum usable events; below this an
            :class:`InsufficientDataError` is raised.
        drop_duplicates: Collapse exact duplicate timestamps. Duplicates are
            common in exported logs (e.g. a squashed commit written twice) and
            would otherwise fabricate zero-length gaps.
        tz_convert: If given, convert timezone-aware input to this zone before
            dropping the offset. Naive input is left alone.

    Returns:
        A validated frame plus counts of what was discarded, so the UI and the
        report can be honest about data quality.

    Raises:
        InsufficientDataError: if fewer than ``min_events`` timestamps survive.
    """
    raw = _coerce_to_series(data, column)
    n_input = len(raw)

    try:
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)
    except ValueError:
        # Input mixes UTC offsets (common when logs are merged across machines);
        # pandas refuses to guess, so normalise everything through UTC.
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)

    # Older pandas returns an object-dtype series here instead of raising.
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        if tz_convert:
            parsed = parsed.dt.tz_convert(tz_convert)
        parsed = parsed.dt.tz_localize(None)

    parsed = parsed.dropna()
    dropped_unparseable = n_input - len(parsed)

    dropped_duplicates = 0
    if drop_duplicates:
        before = len(parsed)
        parsed = parsed.drop_duplicates()
        dropped_duplicates = before - len(parsed)

    parsed = parsed.sort_values().reset_index(drop=True)

    if len(parsed) < min_events:
        raise InsufficientDataError(
            f"Need at least {min_events} valid timestamps to analyse, got {len(parsed)} "
            f"(from {n_input} input rows; {dropped_unparseable} unparseable, "
            f"{dropped_duplicates} duplicates)."
        )

    frame = pd.DataFrame({TIMESTAMP_COLUMN: parsed})
    return EventFrame(
        frame=frame,
        dropped_unparseable=dropped_unparseable,
        dropped_duplicates=dropped_duplicates,
    )
