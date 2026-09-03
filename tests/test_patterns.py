"""Pattern classification and heuristic scoring tests."""

from __future__ import annotations

import pandas as pd
import pytest

from procrastination_analyzer.config import DEFAULT_CONFIG, AnalyzerConfig
from procrastination_analyzer.features import extract_features
from procrastination_analyzer.patterns import (
    Pattern,
    avoidance_score,
    classify_pattern,
    heuristic_risk,
    risk_band,
)


class TestConfigValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            AnalyzerConfig(avoidance_weights={"long_gap_rate": 0.5})

    def test_gap_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match="long_gap_h < extended_gap_h"):
            AnalyzerConfig(long_gap_h=48.0, extended_gap_h=24.0)

    def test_risk_bands_must_be_ordered(self):
        with pytest.raises(ValueError, match="risk_medium < risk_high"):
            AnalyzerConfig(risk_medium=0.8, risk_high=0.4)

    def test_with_overrides_revalidates(self):
        with pytest.raises(ValueError):
            DEFAULT_CONFIG.with_overrides(long_gap_h=100.0)

    def test_with_overrides_returns_a_new_config(self):
        updated = DEFAULT_CONFIG.with_overrides(long_gap_h=12.0)
        assert updated.long_gap_h == 12.0
        assert DEFAULT_CONFIG.long_gap_h == 24.0


class TestClassification:
    def test_steady_weekday_worker_is_consistent(self, steady_weekday_worker):
        result = classify_pattern(extract_features(steady_weekday_worker))
        assert result.pattern is Pattern.CONSISTENT

    def test_crammer_is_not_labelled_consistent(self, late_night_crammer):
        result = classify_pattern(extract_features(late_night_crammer))
        assert result.pattern is not Pattern.CONSISTENT

    def test_result_always_carries_evidence(self, late_night_crammer):
        result = classify_pattern(extract_features(late_night_crammer))
        assert result.evidence
        assert all(isinstance(e, str) and e for e in result.evidence)

    def test_all_patterns_are_scored(self, steady_weekday_worker):
        result = classify_pattern(extract_features(steady_weekday_worker))
        assert set(result.scores) == {p.value for p in Pattern}

    def test_confidence_is_reduced_for_thin_records(self, make_events):
        thin = make_events(["2025-01-01 09:00", "2025-01-04 23:00", "2025-01-09 22:00"])
        assert classify_pattern(extract_features(thin)).confidence < 0.6

    def test_coverage_alone_cannot_win_against_long_gaps(self, make_events):
        """A record touching many days but riddled with silences is not consistent."""
        ts = [pd.Timestamp("2025-01-01 14:00") + pd.Timedelta(days=2 * d) for d in range(20)]
        result = classify_pattern(extract_features(make_events(ts)))
        assert result.pattern is not Pattern.CONSISTENT

    def test_single_afternoon_burst_is_not_consistent(self, make_events):
        """A record confined to one sitting trivially covers every day it spans."""
        ts = [pd.Timestamp("2025-01-06 15:00") + pd.Timedelta(minutes=7 * i) for i in range(9)]
        result = classify_pattern(extract_features(make_events(ts)))
        assert result.pattern is not Pattern.CONSISTENT

    def test_consistency_evidence_requires_an_observable_span(self, make_events):
        """Neither coverage nor gap-freedom may vouch for a sub-day record."""
        short = [pd.Timestamp("2025-01-06 15:00") + pd.Timedelta(minutes=7 * i) for i in range(9)]
        long = [
            pd.Timestamp("2025-01-06 09:00") + pd.Timedelta(days=d, hours=h)
            for d in range(20)
            for h in range(3)
        ]
        short_features = extract_features(make_events(short))
        assert short_features.span_days < DEFAULT_CONFIG.min_span_days_for_rhythm

        short_score = classify_pattern(short_features).scores[Pattern.CONSISTENT.value]
        long_score = classify_pattern(extract_features(make_events(long))).scores[
            Pattern.CONSISTENT.value
        ]
        assert short_score < long_score

    @pytest.mark.parametrize("pattern", list(Pattern))
    def test_every_pattern_has_prose_and_a_category(self, pattern):
        assert len(pattern.description()) > 30
        assert pattern.retrieval_category()


class TestAvoidanceScore:
    def test_bounded_between_zero_and_one(self, steady_weekday_worker, late_night_crammer):
        for events in (steady_weekday_worker, late_night_crammer):
            assert 0.0 <= avoidance_score(extract_features(events)) <= 1.0

    def test_does_not_saturate_at_one(self, late_night_crammer):
        """The original score pinned to exactly 1.0 for almost any real record."""
        assert avoidance_score(extract_features(late_night_crammer)) < 1.0

    def test_crammer_scores_above_steady_worker(self, steady_weekday_worker, late_night_crammer):
        assert avoidance_score(extract_features(late_night_crammer)) > avoidance_score(
            extract_features(steady_weekday_worker)
        )

    def test_is_stable_as_the_record_lengthens(self, make_events):
        def build(weeks):
            return make_events(
                pd.Timestamp("2025-01-06 23:00") + pd.Timedelta(weeks=w, days=d, minutes=20 * i)
                for w in range(weeks)
                for d in (0, 3)
                for i in range(3)
            )

        short = avoidance_score(extract_features(build(4)))
        long = avoidance_score(extract_features(build(32)))
        assert abs(short - long) < 0.05


class TestHeuristicRisk:
    def test_bounded(self, steady_weekday_worker):
        assert 0.0 <= heuristic_risk(extract_features(steady_weekday_worker)) <= 1.0

    def test_inactivity_actually_moves_the_estimate(self, steady_weekday_worker):
        """Regression: the original inactivity term was structurally always zero."""
        last = steady_weekday_worker.ts.max()
        fresh = heuristic_risk(extract_features(steady_weekday_worker, reference_time=last))
        stale = heuristic_risk(
            extract_features(steady_weekday_worker, reference_time=last + pd.Timedelta(days=5))
        )
        assert stale > fresh + 0.05

    def test_steady_worker_is_lower_risk_than_crammer(
        self, steady_weekday_worker, late_night_crammer
    ):
        assert heuristic_risk(extract_features(steady_weekday_worker)) < heuristic_risk(
            extract_features(late_night_crammer)
        )

    @pytest.mark.parametrize(("risk", "expected"), [(0.05, "Low"), (0.5, "Medium"), (0.9, "High")])
    def test_bands(self, risk, expected):
        assert risk_band(risk) == expected
