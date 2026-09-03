"""Procrastination Pattern Analyzer.

Behavioural analytics over activity timestamps: scale-invariant feature
extraction, an interpretable rule-based classifier, calibrated risk models
validated against a labelled simulator, and TF-IDF retrieval of suggestions.

Typical use::

    from procrastination_analyzer import analyze
    result = analyze(["2025-01-06 09:00", "2025-01-06 09:30", ...])
    print(result.pattern.pattern.value, result.risk)
"""

from __future__ import annotations

__version__ = "0.2.0"

from .config import DEFAULT_CONFIG, AnalyzerConfig
from .features import BehaviouralFeatures, extract_features
from .patterns import Pattern, avoidance_score, classify_pattern, heuristic_risk
from .pipeline import AnalysisResult, analyze, analyze_file
from .risk import RiskModel
from .schema import EventFrame, InsufficientDataError, build_event_frame

__all__ = [
    "__version__",
    "AnalysisResult",
    "AnalyzerConfig",
    "BehaviouralFeatures",
    "DEFAULT_CONFIG",
    "EventFrame",
    "InsufficientDataError",
    "Pattern",
    "RiskModel",
    "analyze",
    "analyze_file",
    "avoidance_score",
    "build_event_frame",
    "classify_pattern",
    "extract_features",
    "heuristic_risk",
]
