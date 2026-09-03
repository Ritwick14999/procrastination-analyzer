"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procrastination_analyzer.schema import EventFrame, build_event_frame


def _make_events(timestamps) -> EventFrame:
    """Build an event frame from anything iterable, bypassing the minimum."""
    return build_event_frame(list(timestamps), min_events=1)


@pytest.fixture
def make_events():
    """Factory fixture for building event frames inside a test."""
    return _make_events


@pytest.fixture
def steady_weekday_worker() -> EventFrame:
    """Four events every weekday morning for eight weeks. No procrastination."""
    ts = [
        pd.Timestamp("2025-01-06 09:00") + pd.Timedelta(weeks=w, days=d, hours=h)
        for w in range(8)
        for d in range(5)
        for h in range(4)
    ]
    return _make_events(ts)


@pytest.fixture
def late_night_crammer() -> EventFrame:
    """Two dense late-night sessions a week, with multi-day silences."""
    ts = [
        pd.Timestamp("2025-01-07 23:00") + pd.Timedelta(weeks=w, days=d, minutes=11 * i)
        for w in range(8)
        for d in (0, 3)
        for i in range(8)
    ]
    return _make_events(ts)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
