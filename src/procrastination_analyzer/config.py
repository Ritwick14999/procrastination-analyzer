"""Tunable parameters for feature extraction and scoring.

Every threshold used anywhere in the analysis pipeline lives here. Nothing in
``features.py`` or ``patterns.py`` is allowed to hard-code a number: that keeps
the heuristics auditable, lets the evaluation harness sweep parameters, and
makes it obvious which choices are judgement calls rather than derived facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["AnalyzerConfig", "DEFAULT_CONFIG"]


@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for the analysis pipeline.

    Attributes are grouped by the stage that consumes them. All hour values are
    local wall-clock hours in the 0-23 range; all durations are in hours.
    """

    # --- Time-of-day segmentation -------------------------------------------
    morning_start: int = 5
    afternoon_start: int = 12
    evening_start: int = 17
    night_start: int = 22
    #: Activity at or after this hour counts as "late night" for scoring.
    late_night_hour: int = 23

    # --- Session / burst detection ------------------------------------------
    #: Two consecutive events separated by more than this many hours are treated
    #: as belonging to different working sessions.
    session_timeout_h: float = 2.0
    #: A session must contain at least this many events to count as a "burst".
    burst_min_events: int = 5
    #: ...and must be no longer than this to count as a burst (a slow all-day
    #: trickle is not a cram session).
    burst_max_span_h: float = 3.0

    # --- Gap analysis --------------------------------------------------------
    #: Inter-event gaps longer than this are "long gaps". Measured in *active*
    #: hours, so weekends do not inflate them (see ``exclude_weekend_from_gaps``).
    long_gap_h: float = 24.0
    #: A second, more severe gap threshold used as a separate signal.
    extended_gap_h: float = 48.0
    #: When true, hours falling on a weekend are discounted before comparing a
    #: gap against ``long_gap_h``. A Mon-Fri worker should not be penalised for
    #: not committing on Sunday.
    exclude_weekend_from_gaps: bool = True

    # --- Scoring -------------------------------------------------------------
    #: Weights for the avoidance score. Must sum to 1.0; validated on init.
    avoidance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "long_gap_rate": 0.35,
            "burst_event_share": 0.25,
            "late_night_share": 0.20,
            "rhythm_irregularity": 0.20,
        }
    )
    #: Logistic squashing steepness for turning raw signals into 0-1 scores.
    #: Higher = sharper transition around the midpoint.
    score_steepness: float = 4.0

    # --- Risk thresholds -----------------------------------------------------
    risk_high: float = 0.66
    risk_medium: float = 0.40

    # --- Data requirements ---------------------------------------------------
    #: Below this many events the output is reported as low-confidence.
    min_events_for_confidence: int = 20
    #: Below this many events analysis refuses to run at all.
    min_events_required: int = 3
    #: Coverage (share of days with activity) is meaningless over a span shorter
    #: than this, since a record confined to one afternoon trivially "covers"
    #: every day it spans. Rules that rely on coverage are gated on it.
    min_span_days_for_rhythm: float = 3.0

    def __post_init__(self) -> None:
        total = sum(self.avoidance_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"avoidance_weights must sum to 1.0, got {total:.4f} ({self.avoidance_weights})"
            )
        if not 0 <= self.late_night_hour <= 23:
            raise ValueError("late_night_hour must be in [0, 23]")
        if self.burst_min_events < 2:
            raise ValueError("burst_min_events must be at least 2")
        if self.long_gap_h <= 0 or self.extended_gap_h <= self.long_gap_h:
            raise ValueError("require 0 < long_gap_h < extended_gap_h")
        if not 0 < self.risk_medium < self.risk_high < 1:
            raise ValueError("require 0 < risk_medium < risk_high < 1")

    def with_overrides(self, **kwargs: Any) -> AnalyzerConfig:
        """Return a copy with the given fields replaced (validated on build)."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for embedding in reports, so results stay reproducible."""
        return {
            "session_timeout_h": self.session_timeout_h,
            "burst_min_events": self.burst_min_events,
            "burst_max_span_h": self.burst_max_span_h,
            "long_gap_h": self.long_gap_h,
            "extended_gap_h": self.extended_gap_h,
            "exclude_weekend_from_gaps": self.exclude_weekend_from_gaps,
            "late_night_hour": self.late_night_hour,
            "avoidance_weights": dict(self.avoidance_weights),
            "score_steepness": self.score_steepness,
            "min_span_days_for_rhythm": self.min_span_days_for_rhythm,
        }


DEFAULT_CONFIG = AnalyzerConfig()
