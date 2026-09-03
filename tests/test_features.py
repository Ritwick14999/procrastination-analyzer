"""Feature extraction tests, including regressions for the original defects."""

from __future__ import annotations

import pandas as pd
import pytest

from procrastination_analyzer.config import DEFAULT_CONFIG
from procrastination_analyzer.features import (
    FEATURE_NAMES,
    extract_features,
    segment_sessions,
)
from procrastination_analyzer.schema import build_event_frame


def _repeat_weekly(make_events, weeks: int):
    """An identical two-sessions-per-week rhythm repeated for `weeks`."""
    ts = [
        pd.Timestamp("2025-01-06 23:00") + pd.Timedelta(weeks=w, days=d, minutes=20 * i)
        for w in range(weeks)
        for d in (0, 3)
        for i in range(3)
    ]
    return make_events(ts)


class TestScaleInvariance:
    """Regression: scores must not drift as more of the same behaviour is added.

    The original implementation normalised counts by ``max(1, n / 25)``, so an
    identical weekly rhythm scored differently at 12 events and 192 events.
    """

    def test_gap_rate_converges_as_record_lengthens(self, make_events):
        rates = [
            extract_features(_repeat_weekly(make_events, w)).long_gap_rate for w in (8, 16, 32)
        ]
        # Successive differences must shrink: the statistic converges.
        assert abs(rates[2] - rates[1]) < abs(rates[1] - rates[0])
        assert abs(rates[2] - rates[1]) < 0.02

    def test_share_features_are_exactly_stable(self, make_events):
        short = extract_features(_repeat_weekly(make_events, 4))
        long = extract_features(_repeat_weekly(make_events, 32))
        assert short.late_night_share == long.late_night_share
        assert short.rhythm_irregularity == pytest.approx(long.rhythm_irregularity, abs=1e-9)
        assert short.burst_event_share == long.burst_event_share


class TestSessionSegmentation:
    """Regression: bursts were counted over overlapping sliding windows."""

    def test_one_sitting_is_one_session(self, make_events):
        events = make_events(
            pd.Timestamp("2025-02-03 09:00") + pd.Timedelta(minutes=10 * i) for i in range(20)
        )
        sessions = segment_sessions(events)
        assert len(sessions) == 1
        assert int(sessions["n_events"].iloc[0]) == 20

    def test_separated_sittings_split(self, make_events):
        events = make_events(
            [
                pd.Timestamp("2025-02-03 09:00"),
                pd.Timestamp("2025-02-03 09:30"),
                pd.Timestamp("2025-02-03 18:00"),
                pd.Timestamp("2025-02-04 09:00"),
            ]
        )
        assert len(segment_sessions(events)) == 3

    def test_dense_short_session_flagged_as_burst(self, make_events):
        events = make_events(
            pd.Timestamp("2025-02-03 22:00") + pd.Timedelta(minutes=11 * i) for i in range(8)
        )
        sessions = segment_sessions(events)
        assert bool(sessions["is_burst"].iloc[0]) is True

    def test_slow_trickle_is_not_a_burst(self, make_events):
        """Many events spread over a long stretch is not a cram session."""
        config = DEFAULT_CONFIG.with_overrides(session_timeout_h=8.0)
        events = make_events(
            pd.Timestamp("2025-02-03 08:00") + pd.Timedelta(hours=1.5 * i) for i in range(8)
        )
        sessions = segment_sessions(events, config)
        assert not sessions["is_burst"].any()


class TestWeekendAwareGaps:
    """Regression: a Mon-Fri worker was flagged for not working weekends."""

    def test_weekend_silence_is_not_a_long_gap(self, steady_weekday_worker):
        features = extract_features(steady_weekday_worker)
        assert features.long_gap_rate == 0.0

    def test_weekend_discount_can_be_disabled(self, steady_weekday_worker):
        config = DEFAULT_CONFIG.with_overrides(exclude_weekend_from_gaps=False)
        features = extract_features(steady_weekday_worker, config)
        assert features.long_gap_rate > 0.0

    def test_genuine_weekday_silence_still_counts(self, make_events):
        """A Monday-to-Thursday gap has no weekend to discount."""
        events = make_events(
            ["2025-01-06 09:00", "2025-01-06 10:00", "2025-01-09 09:00", "2025-01-09 10:00"]
        )
        assert extract_features(events).long_gap_rate > 0.0


class TestCircadianFeatures:
    """Regression: late night did not wrap past midnight; variance was linear."""

    def test_after_midnight_counts_as_late_night(self, make_events):
        events = make_events(["2025-01-06 00:30", "2025-01-07 01:15", "2025-01-08 02:00"])
        assert extract_features(events).late_night_share == 1.0

    def test_regular_midnight_worker_is_not_irregular(self, make_events):
        """23:00 and 01:00 are two hours apart, not twenty-two."""
        ts = [
            pd.Timestamp("2025-01-06 23:30") + pd.Timedelta(days=d, minutes=20 * i)
            for d in range(14)
            for i in range(3)
        ]
        features = extract_features(make_events(ts))
        assert features.rhythm_irregularity < 0.15

    def test_scattered_hours_are_irregular(self, rng, make_events):
        ts = [
            pd.Timestamp("2025-01-06") + pd.Timedelta(days=d, hours=float(rng.uniform(0, 24)))
            for d in range(40)
        ]
        assert extract_features(make_events(ts)).rhythm_irregularity > 0.4


class TestRecencyAndReference:
    """Regression: 'hours since last activity' was structurally always zero."""

    def test_defaults_to_zero_for_closed_records(self, steady_weekday_worker):
        assert extract_features(steady_weekday_worker).hours_since_last_event == 0.0

    def test_reflects_supplied_reference_time(self, steady_weekday_worker):
        last = steady_weekday_worker.ts.max()
        features = extract_features(
            steady_weekday_worker, reference_time=last + pd.Timedelta(hours=36)
        )
        assert features.hours_since_last_event == pytest.approx(36.0, abs=0.01)

    def test_ramping_up_gives_trend_above_one(self, make_events):
        ts = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=d) for d in range(0, 30, 6)] + [
            pd.Timestamp("2025-01-25") + pd.Timedelta(hours=h) for h in range(0, 96, 4)
        ]
        assert extract_features(make_events(ts)).recency_trend > 1.5


class TestFeatureContract:
    def test_vector_matches_declared_names(self, steady_weekday_worker):
        features = extract_features(steady_weekday_worker)
        assert features.to_vector().shape == (len(FEATURE_NAMES),)

    def test_shares_are_bounded(self, late_night_crammer):
        f = extract_features(late_night_crammer)
        for name in (
            "coverage",
            "late_night_share",
            "evening_share",
            "working_hours_share",
            "weekend_share",
            "burst_event_share",
            "rhythm_irregularity",
            "long_gap_rate",
        ):
            assert 0.0 <= getattr(f, name) <= 1.0, name

    def test_confidence_flag_tracks_event_count(self):
        small = build_event_frame(["2025-01-01", "2025-01-02", "2025-01-03"])
        assert extract_features(small).is_confident is False

    def test_minimal_three_event_record_does_not_crash(self):
        features = extract_features(
            build_event_frame(["2025-01-01 09:00", "2025-01-02 10:00", "2025-01-03 11:00"])
        )
        assert features.n_events == 3
        assert features.peak_hour in range(24)
