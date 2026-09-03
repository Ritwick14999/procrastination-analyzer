"""Tests for event-frame construction and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from procrastination_analyzer.schema import (
    InsufficientDataError,
    build_event_frame,
)


def test_accepts_plain_iterable():
    events = build_event_frame(["2025-01-01 09:00", "2025-01-02 10:00", "2025-01-03 11:00"])
    assert len(events) == 3
    assert str(events.ts.dtype).startswith("datetime64")


def test_sorts_and_resets_index():
    events = build_event_frame(["2025-01-03 11:00", "2025-01-01 09:00", "2025-01-02 10:00"])
    assert events.ts.is_monotonic_increasing
    assert list(events.frame.index) == [0, 1, 2]


def test_drops_unparseable_and_reports_count():
    events = build_event_frame(
        ["2025-01-01 09:00", "not a date", "2025-01-02 10:00", "2025-01-03 11:00"]
    )
    assert len(events) == 3
    assert events.dropped_unparseable == 1


def test_collapses_duplicate_timestamps():
    events = build_event_frame(
        ["2025-01-01 09:00", "2025-01-01 09:00", "2025-01-02 10:00", "2025-01-03 09:00"]
    )
    assert len(events) == 3
    assert events.dropped_duplicates == 1


def test_raises_when_too_few_events():
    with pytest.raises(InsufficientDataError, match="at least 3"):
        build_event_frame(["2025-01-01 09:00", "2025-01-02 09:00"])


def test_autodetects_common_column_names():
    for name in ("ts", "timestamp", "created_at", "datetime"):
        frame = pd.DataFrame({name: ["2025-01-01", "2025-01-02", "2025-01-03"]})
        assert len(build_event_frame(frame)) == 3


def test_single_column_frame_accepted_whatever_the_name():
    frame = pd.DataFrame({"weird_name": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    assert len(build_event_frame(frame)) == 3


def test_ambiguous_multicolumn_frame_raises():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(KeyError, match="Could not find a timestamp column"):
        build_event_frame(frame)


def test_explicit_missing_column_raises():
    frame = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    with pytest.raises(KeyError, match="nope"):
        build_event_frame(frame, column="nope")


def test_mixed_utc_offsets_do_not_crash():
    """Merged logs from different machines carry mixed offsets."""
    events = build_event_frame(
        [
            "2025-01-01T10:00:00+05:30",
            "2025-01-02T11:00:00-08:00",
            "2025-01-03T09:00:00Z",
        ]
    )
    assert len(events) == 3
    assert events.ts.dt.tz is None


def test_tz_convert_shifts_wall_clock():
    events = build_event_frame(
        ["2025-01-01T10:00:00+00:00"] * 1
        + ["2025-01-02T10:00:00+00:00", "2025-01-03T10:00:00+00:00"],
        tz_convert="Asia/Kolkata",
    )
    assert events.ts.iloc[0].hour == 15  # 10:00 UTC -> 15:30 IST


def test_span_and_active_days():
    events = build_event_frame(["2025-01-01 09:00", "2025-01-01 18:00", "2025-01-05 09:00"])
    assert events.active_days == 2
    assert events.span_days == pytest.approx(4.0, abs=0.01)


def test_last_n_days_filters():
    events = build_event_frame(
        ["2025-01-01 09:00", "2025-01-10 09:00", "2025-01-11 09:00", "2025-01-12 09:00"]
    )
    assert len(events.last_n_days(3)) == 3
