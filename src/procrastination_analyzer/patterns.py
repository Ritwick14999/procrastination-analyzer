"""Rule-based pattern classification and heuristic scoring.

This is the interpretable baseline. It is deliberately kept simple and fully
transparent so that :mod:`procrastination_analyzer.evaluate` can measure how
much (if anything) the learned model in :mod:`procrastination_analyzer.risk`
adds on top of it.

Every rule reports the evidence that fired it, so the UI never has to say
"trust me" — see :class:`PatternResult.evidence`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import DEFAULT_CONFIG, AnalyzerConfig
from .features import BehaviouralFeatures

__all__ = [
    "Pattern",
    "PatternResult",
    "classify_pattern",
    "avoidance_score",
    "heuristic_risk",
    "risk_band",
]


class Pattern(str, Enum):
    """The behavioural archetypes the rule engine can report."""

    AVOIDANCE = "Avoidance-driven"
    DEADLINE_DRIVEN = "Deadline-driven (bursty)"
    FATIGUE = "Fatigue-driven (evening-loaded)"
    NOCTURNAL = "Nocturnal but consistent"
    CONSISTENT = "Consistent / low-procrastination"
    MIXED = "Mixed / situational"

    def description(self) -> str:
        """A one-line, non-clinical explanation shown next to the label."""
        return {
            Pattern.AVOIDANCE: (
                "Long silences followed by late returns. Usually means the task feels "
                "high-stakes or under-specified rather than genuinely hard."
            ),
            Pattern.DEADLINE_DRIVEN: (
                "Work clusters into dense bursts with quiet stretches between. Effective "
                "under pressure, but fragile when several deadlines collide."
            ),
            Pattern.FATIGUE: (
                "Activity is concentrated in the evening. Workable, but starting is "
                "hardest exactly when energy is lowest."
            ),
            Pattern.NOCTURNAL: (
                "A consistently late schedule rather than a disrupted one. Regular timing "
                "matters more than which hours you pick."
            ),
            Pattern.CONSISTENT: (
                "Steady, well-distributed activity with few long gaps. The goal here is "
                "protecting the rhythm, not changing it."
            ),
            Pattern.MIXED: (
                "No single dominant signal. Typically responds to clearer next steps and "
                "a predictable work slot."
            ),
        }[self]

    def retrieval_category(self) -> str:
        """The snippet category most relevant to this pattern."""
        return {
            Pattern.AVOIDANCE: "avoidance",
            Pattern.DEADLINE_DRIVEN: "time_management",
            Pattern.FATIGUE: "fatigue",
            Pattern.NOCTURNAL: "habits",
            Pattern.CONSISTENT: "habits",
            Pattern.MIXED: "planning",
        }[self]


@dataclass(frozen=True)
class PatternResult:
    """A classification plus the evidence behind it."""

    pattern: Pattern
    #: 0-1 confidence, discounted when the record is short.
    confidence: float
    #: Human-readable statements of what drove the decision.
    evidence: list[str] = field(default_factory=list)
    #: Score for every candidate pattern, for transparency and debugging.
    scores: dict[str, float] = field(default_factory=dict)


def _logistic(x: float, midpoint: float, steepness: float) -> float:
    """Squash a raw signal into (0, 1) around ``midpoint``."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))


def classify_pattern(
    features: BehaviouralFeatures, config: AnalyzerConfig = DEFAULT_CONFIG
) -> PatternResult:
    """Assign a behavioural archetype using additive, inspectable evidence.

    Rather than the original cascade of ``if`` statements — where rule order
    silently decided the answer and a Mon-Fri worker fell through to "Mixed" —
    each pattern accumulates a score from independent signals and the highest
    wins. Ties and near-ties surface as lower confidence.
    """
    f = features
    scores: dict[Pattern, float] = dict.fromkeys(Pattern, 0.0)
    evidence: dict[Pattern, list[str]] = {p: [] for p in Pattern}

    def add(pattern: Pattern, weight: float, reason: str) -> None:
        if weight > 0:
            scores[pattern] += weight
            evidence[pattern].append(reason)

    # --- Avoidance: long silences plus a late-night return --------------------
    if f.long_gap_rate > 0.15:
        add(
            Pattern.AVOIDANCE,
            2.0 * f.long_gap_rate,
            f"{f.long_gap_rate:.0%} of gaps exceed {config.long_gap_h:.0f}h of active time",
        )
    if f.extended_gap_rate > 0.05:
        add(
            Pattern.AVOIDANCE,
            1.5 * f.extended_gap_rate,
            f"{f.extended_gap_rate:.0%} of gaps exceed {config.extended_gap_h:.0f}h",
        )
    # Sparse attendance is the strongest single avoidance signal, and it is what
    # separates avoidance from a merely late schedule, so it carries real weight.
    if f.coverage < 0.5:
        add(
            Pattern.AVOIDANCE,
            2.5 * (0.5 - f.coverage),
            f"Active on only {f.coverage:.0%} of days in the record",
        )
    if f.rhythm_irregularity > 0.5:
        add(
            Pattern.AVOIDANCE,
            1.2 * (f.rhythm_irregularity - 0.5),
            f"Work times are scattered rather than habitual "
            f"(irregularity {f.rhythm_irregularity:.2f})",
        )

    # --- Deadline-driven: dense bursts, ramping up ---------------------------
    if f.burst_event_share > 0.2:
        add(
            Pattern.DEADLINE_DRIVEN,
            2.0 * f.burst_event_share,
            f"{f.burst_event_share:.0%} of events happen inside dense burst sessions",
        )
    if f.recency_trend > 1.5:
        add(
            Pattern.DEADLINE_DRIVEN,
            min(1.0, 0.5 * (f.recency_trend - 1.0)),
            f"Activity is ramping up ({f.recency_trend:.1f}x more in the final third)",
        )
    # Session size is largely redundant with burst share, so it only contributes
    # a small tie-breaking amount; weighting it heavily made every persona with
    # medium-sized sessions look deadline-driven.
    if f.mean_session_events > 6:
        add(
            Pattern.DEADLINE_DRIVEN,
            min(0.4, 0.10 * (f.mean_session_events - 6)),
            f"Sessions average {f.mean_session_events:.1f} events",
        )

    # --- Fatigue: evening-loaded but not nocturnal ---------------------------
    if f.evening_share > 0.35 and f.late_night_share < 0.25:
        add(
            Pattern.FATIGUE,
            2.0 * f.evening_share,
            f"{f.evening_share:.0%} of activity falls in the evening block",
        )

    # --- Nocturnal: late *and* regular --------------------------------------
    # The label is "nocturnal but consistent", so consistency is a gate, not a
    # bonus. Without it every scattered late-night avoider scores as nocturnal,
    # since both profiles share the same clock hours and differ only in rhythm.
    if f.late_night_share > 0.35:
        consistency = (1.0 - f.rhythm_irregularity) * max(0.0, 1.0 - 2.0 * f.long_gap_rate)
        add(
            Pattern.NOCTURNAL,
            (1.0 * f.late_night_share + 1.5 * consistency) * min(1.0, f.coverage / 0.5),
            f"{f.late_night_share:.0%} of activity is late night, on a "
            f"{'regular' if f.rhythm_irregularity < 0.4 else 'variable'} schedule",
        )

    # --- Consistent: broad coverage, few gaps, ordinary hours ---------------
    # Coverage alone is not consistency: a record can touch many days and still
    # be full of long silences. Discount the coverage bonus by the gap rate so
    # "Consistent" cannot win on attendance while the gaps say otherwise.
    # Coverage is only evidence of consistency once the record spans enough days
    # to have observed day-to-day behaviour at all. A burst of commits inside a
    # single afternoon trivially covers 100% of the days it spans, which would
    # otherwise read as exemplary consistency.
    if f.coverage > 0.5 and f.span_days >= config.min_span_days_for_rhythm:
        gap_penalty = max(0.0, 1.0 - 2.0 * f.long_gap_rate)
        add(
            Pattern.CONSISTENT,
            1.5 * f.coverage * gap_penalty,
            f"Active on {f.coverage:.0%} of days across {f.span_days:.0f} days, "
            f"with few interrupting gaps",
        )
    # "No long gaps" is equally trivial over a sub-day span: there was never room
    # for one. Gate it on the same observable-span requirement as coverage.
    if f.long_gap_rate < 0.1 and f.span_days >= config.min_span_days_for_rhythm:
        add(
            Pattern.CONSISTENT,
            1.0 * (1.0 - f.long_gap_rate * 10),
            "Very few long inactivity gaps",
        )
    if f.working_hours_share > 0.5:
        add(
            Pattern.CONSISTENT,
            1.0 * f.working_hours_share,
            f"{f.working_hours_share:.0%} of activity is in standard working hours",
        )

    # --- Mixed: the floor, so something always wins -------------------------
    scores[Pattern.MIXED] = 0.9
    evidence[Pattern.MIXED].append("No single dominant behavioural signal")

    ranked: list[tuple[Pattern, float]] = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winner, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence: how clearly the winner separates, discounted for thin data.
    separation = (top_score - runner_up) / top_score if top_score > 0 else 0.0
    confidence = float(np.clip(0.45 + 0.55 * separation, 0.0, 1.0))
    if not f.is_confident:
        confidence *= 0.6

    return PatternResult(
        pattern=winner,
        confidence=round(confidence, 3),
        evidence=evidence[winner][:5],
        scores={p.value: round(s, 3) for p, s in ranked},
    )


def avoidance_score(
    features: BehaviouralFeatures, config: AnalyzerConfig = DEFAULT_CONFIG
) -> float:
    """A 0-1 avoidance/perfectionism indicator.

    Unlike the original — which summed raw counts, clipped at 1.0, and therefore
    saturated at exactly 1.0 for almost any real record — each component is
    squashed into (0, 1) first, then combined by the configured weights. The
    result genuinely varies across its range and does not depend on how long the
    record happens to be.
    """
    f = features
    components = {
        "long_gap_rate": _logistic(f.long_gap_rate, 0.25, config.score_steepness * 3),
        "burst_event_share": _logistic(f.burst_event_share, 0.35, config.score_steepness * 2),
        "late_night_share": _logistic(f.late_night_share, 0.30, config.score_steepness * 3),
        "rhythm_irregularity": _logistic(f.rhythm_irregularity, 0.50, config.score_steepness * 2),
    }
    score = sum(config.avoidance_weights[k] * v for k, v in components.items())
    return round(float(np.clip(score, 0.0, 1.0)), 3)


def heuristic_risk(features: BehaviouralFeatures, config: AnalyzerConfig = DEFAULT_CONFIG) -> float:
    """Rule-based probability that the next day sees no meaningful activity.

    Fixes the dead term in the original ``predict_risk``: it computed "hours
    since last activity" as ``last_ts - last_week_max``, which is identically
    zero because ``last_ts`` *is* the maximum of the last week. The feature
    contributed nothing to any prediction. Here the value comes from
    ``hours_since_last_event``, measured against a caller-supplied reference
    time, so it is only non-zero when it genuinely can be.
    """
    f = features

    inactivity = _logistic(f.hours_since_last_event, 24.0, 0.08)
    gap_pressure = _logistic(f.long_gap_rate, 0.25, config.score_steepness * 3)
    irregularity = _logistic(f.rhythm_irregularity, 0.55, config.score_steepness * 2)
    fading = _logistic(1.0 / max(f.recency_trend, 0.05), 1.6, 1.5)
    inconsistency = 1.0 - f.coverage

    raw = (
        0.30 * gap_pressure
        + 0.22 * inactivity
        + 0.18 * inconsistency
        + 0.15 * irregularity
        + 0.15 * fading
    )
    return round(float(np.clip(raw, 0.0, 1.0)), 3)


def risk_band(risk: float, config: AnalyzerConfig = DEFAULT_CONFIG) -> str:
    """Map a risk probability onto a coarse, human-facing band."""
    if risk >= config.risk_high:
        return "High"
    if risk >= config.risk_medium:
        return "Medium"
    return "Low"
