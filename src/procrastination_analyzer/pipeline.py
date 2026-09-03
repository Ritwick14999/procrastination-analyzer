"""End-to-end analysis pipeline.

A single entry point that the CLI, the Streamlit app and the tests all share,
so there is exactly one definition of "analyse this log" and the UI cannot
drift away from what the CLI does.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CONFIG, AnalyzerConfig
from .features import BehaviouralFeatures, extract_features
from .patterns import (
    PatternResult,
    avoidance_score,
    classify_pattern,
    heuristic_risk,
    risk_band,
)
from .retrieval import retrieve
from .risk import RiskModel, top_risk_drivers
from .schema import EventFrame, build_event_frame

__all__ = ["AnalysisResult", "analyze", "analyze_file"]


@dataclass
class AnalysisResult:
    """Everything the pipeline produces for one activity log."""

    features: BehaviouralFeatures
    pattern: PatternResult
    avoidance: float
    risk: float
    risk_band: str
    #: Which predictor produced ``risk``: "heuristic" or "model:<kind>".
    risk_source: str
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    #: Per-feature contributions when a linear model produced the risk.
    risk_drivers: list[Any] = field(default_factory=list)
    #: Data-quality notes worth surfacing to the user.
    warnings: list[str] = field(default_factory=list)
    generated_at: str = ""
    config_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary, used by ``--format json`` and the report."""
        return {
            "generated_at": self.generated_at,
            "summary": {
                "pattern": self.pattern.pattern.value,
                "pattern_description": self.pattern.pattern.description(),
                "pattern_confidence": self.pattern.confidence,
                "avoidance_score": self.avoidance,
                "risk": self.risk,
                "risk_band": self.risk_band,
                "risk_source": self.risk_source,
            },
            "evidence": self.pattern.evidence,
            "pattern_scores": self.pattern.scores,
            "features": self.features.to_dict(),
            "risk_drivers": [{"feature": n, "contribution": v} for n, v in self.risk_drivers],
            "suggestions": self.suggestions,
            "warnings": self.warnings,
            "config": self.config_used,
        }


def _collect_warnings(events: EventFrame, features: BehaviouralFeatures) -> list[str]:
    """Surface data-quality caveats rather than silently scoring bad input."""
    warnings: list[str] = []
    if not features.is_confident:
        warnings.append(
            f"Only {features.n_events} events analysed. Scores below "
            f"{DEFAULT_CONFIG.min_events_for_confidence} events are indicative at best."
        )
    if features.span_days < DEFAULT_CONFIG.min_span_days_for_rhythm:
        warnings.append(
            f"The record spans under {DEFAULT_CONFIG.min_span_days_for_rhythm:.0f} days, "
            "so day-to-day consistency cannot be assessed at all."
        )
    elif features.span_days < 14:
        warnings.append(
            f"The record spans {features.span_days:.1f} days. Weekly rhythms cannot be "
            "estimated reliably from under two weeks of data."
        )
    if events.dropped_unparseable:
        warnings.append(f"{events.dropped_unparseable} row(s) had unparseable timestamps.")
    if events.dropped_duplicates:
        warnings.append(f"{events.dropped_duplicates} duplicate timestamp(s) were collapsed.")
    return warnings


def analyze(
    data: pd.DataFrame | pd.Series | Iterable[object] | EventFrame,
    *,
    column: str | None = None,
    config: AnalyzerConfig = DEFAULT_CONFIG,
    model: RiskModel | None = None,
    reference_time: pd.Timestamp | None = None,
    top_k_suggestions: int = 4,
    snippets_path: str | None = None,
    category_override: str | None = None,
) -> AnalysisResult:
    """Run the full analysis over a timestamp log.

    Args:
        data: Raw timestamps in any form accepted by
            :func:`~procrastination_analyzer.schema.build_event_frame`, or an
            already-built :class:`EventFrame`.
        column: Timestamp column name, when passing a multi-column DataFrame.
        config: Threshold configuration.
        model: A trained :class:`RiskModel`. When omitted the transparent
            heuristic is used, so the pipeline never requires a model artifact.
        reference_time: "Now", for recency features. Pass the real current time
            when scoring a live user; defaults to the last observed event.
        top_k_suggestions: How many snippets to retrieve.
        snippets_path: Override the packaged corpus.
        category_override: Force a snippet category instead of deriving it from
            the detected pattern.

    Returns:
        A fully populated :class:`AnalysisResult`.
    """
    events = (
        data
        if isinstance(data, EventFrame)
        else build_event_frame(data, column, min_events=config.min_events_required)
    )

    features = extract_features(events, config, reference_time=reference_time)
    pattern = classify_pattern(features, config)
    avoidance = avoidance_score(features, config)

    if model is not None:
        risk = round(model.predict_one(features), 3)
        risk_source = f"model:{model.kind}"
        drivers = top_risk_drivers(model, features)
    else:
        risk = heuristic_risk(features, config)
        risk_source = "heuristic"
        drivers = []

    category = category_override or pattern.pattern.retrieval_category()
    query = f"{pattern.pattern.value} {pattern.pattern.description()} {' '.join(pattern.evidence)}"
    suggestions = retrieve(
        query, k=top_k_suggestions, category=category, snippets_path=snippets_path
    )

    return AnalysisResult(
        features=features,
        pattern=pattern,
        avoidance=avoidance,
        risk=risk,
        risk_band=risk_band(risk, config),
        risk_source=risk_source,
        suggestions=suggestions,
        risk_drivers=drivers,
        warnings=_collect_warnings(events, features),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config_used=config.to_dict(),
    )


def analyze_file(
    path: str | Path,
    *,
    column: str | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Analyse a CSV file of timestamps.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    frame = pd.read_csv(path)
    return analyze(frame, column=column, **kwargs)
