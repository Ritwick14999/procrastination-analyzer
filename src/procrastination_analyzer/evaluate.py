"""Evaluation harness: does any of this actually work?

Provides cross-validated comparison of the learned risk models against three
reference points, because a model is only interesting relative to what it beats:

1. **The base rate** — always predict the training-set positive rate. Any model
   that cannot beat this is worthless.
2. **The hand-tuned heuristic** — the existing rule-based score. A learned model
   that ties with it adds complexity for nothing.
3. **The other learned model** — linear vs non-linear, to show whether the extra
   capacity buys anything.

Metrics reported, and why each one is here:

* ``roc_auc`` — ranking quality, insensitive to threshold and to calibration.
* ``average_precision`` — more informative than AUC when classes are imbalanced.
* ``brier`` — squared error of the probabilities themselves; catches the
  miscalibration that AUC is blind to. Lower is better.
* ``calibration_error`` — mean absolute gap between predicted and observed
  frequency across bins (ECE).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import DEFAULT_CONFIG, AnalyzerConfig
from .features import BehaviouralFeatures, extract_features
from .patterns import Pattern, classify_pattern, heuristic_risk
from .risk import ModelKind, RiskModel, features_to_matrix
from .schema import EventFrame
from .simulate import simulate_cohort

__all__ = [
    "MetricSet",
    "SimulatedDataset",
    "EvaluationReport",
    "compute_metrics",
    "expected_calibration_error",
    "evaluate_risk_models",
    "PatternEvaluation",
    "evaluate_pattern_rules",
    "build_dataset",
]

#: Maps simulator ground-truth personas onto the rule engine's pattern labels.
PERSONA_TO_PATTERN: dict[str, Pattern] = {
    "avoidant": Pattern.AVOIDANCE,
    "deadline_driven": Pattern.DEADLINE_DRIVEN,
    "fatigued": Pattern.FATIGUE,
    "nocturnal": Pattern.NOCTURNAL,
    "consistent": Pattern.CONSISTENT,
}


@dataclass(frozen=True)
class MetricSet:
    """Scores for one predictor on one fold (or averaged across folds)."""

    roc_auc: float
    average_precision: float
    brier: float
    calibration_error: float
    n: int

    def to_dict(self) -> dict[str, float]:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "brier": round(self.brier, 4),
            "calibration_error": round(self.calibration_error, 4),
            "n": self.n,
        }


@dataclass
class EvaluationReport:
    """Cross-validated comparison across every predictor under test."""

    #: predictor name -> mean metrics across folds
    means: dict[str, MetricSet] = field(default_factory=dict)
    #: predictor name -> standard deviation of roc_auc across folds
    roc_auc_std: dict[str, float] = field(default_factory=dict)
    n_folds: int = 0
    n_samples: int = 0
    base_rate: float = 0.0
    #: Feature importances from the model trained on the full dataset.
    feature_importance: dict[str, float] = field(default_factory=dict)
    logistic_coefficients: dict[str, float] = field(default_factory=dict)

    def best(self, metric: str = "roc_auc") -> str:
        """Name of the top predictor by the given metric (lower-is-better aware)."""
        lower_is_better = metric in {"brier", "calibration_error"}
        return min(
            self.means,
            key=lambda k: getattr(self.means[k], metric) * (1 if lower_is_better else -1),
        )

    def to_markdown(self) -> str:
        """Render as a comparison table for the README or a report."""
        lines = [
            f"Cross-validated over {self.n_samples} simulated users, "
            f"{self.n_folds} stratified folds. Base rate = {self.base_rate:.3f}.",
            "",
            "| Predictor | ROC-AUC | Avg. precision | Brier ↓ | Calib. error ↓ |",
            "|---|---|---|---|---|",
        ]
        for name, m in sorted(self.means.items(), key=lambda kv: kv[1].roc_auc, reverse=True):
            std = self.roc_auc_std.get(name, 0.0)
            lines.append(
                f"| {name} | {m.roc_auc:.3f} ± {std:.3f} | {m.average_precision:.3f} "
                f"| {m.brier:.3f} | {m.calibration_error:.3f} |"
            )
        return "\n".join(lines)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and observed frequency.

    Bins predictions into ``n_bins`` equal-width buckets and takes the
    sample-weighted mean of ``|mean_predicted - observed_rate|``. Empty bins are
    skipped. A perfectly calibrated model scores 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``right=True`` with a clip keeps probability 0.0 inside the first bin.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        total += count * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total / len(y_true))


def compute_metrics(y_true: ArrayLike, y_prob: ArrayLike) -> MetricSet:
    """Compute the full metric set for one set of predictions."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # A degenerate fold (single class) makes ranking metrics undefined; report
    # 0.5 (chance) rather than crashing or silently dropping the fold.
    if len(np.unique(y_true_arr)) < 2:
        auc = 0.5
        ap = float(y_true_arr.mean())
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        ap = float(average_precision_score(y_true_arr, y_prob_arr))

    return MetricSet(
        roc_auc=auc,
        average_precision=ap,
        brier=float(brier_score_loss(y_true_arr, y_prob_arr)),
        calibration_error=expected_calibration_error(y_true_arr, y_prob_arr),
        n=int(len(y_true_arr)),
    )


@dataclass
class SimulatedDataset:
    """A simulated cohort with features, labels and the generating probability."""

    features: list[BehaviouralFeatures]
    labels: np.ndarray
    personas: list[str]
    #: The true P(next-day inactive) each label was drawn from. Only knowable
    #: because the data is simulated; used to estimate the Bayes ceiling.
    oracle_probs: np.ndarray

    def __len__(self) -> int:
        return len(self.features)


def build_dataset(
    n_per_persona: int = 60,
    *,
    days: int = 60,
    seed: int = 20240,
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> SimulatedDataset:
    """Simulate a cohort and extract features, labels and personas.

    Returns:
        A :class:`SimulatedDataset`. Labels are the binary next-day-inactive
        outcome; ``oracle_probs`` are the probabilities those outcomes were
        sampled from.
    """
    cohort = simulate_cohort(n_per_persona=n_per_persona, days=days, seed=seed)
    features: list[BehaviouralFeatures] = []
    labels: list[int] = []
    personas: list[str] = []
    oracle: list[float] = []

    for events, meta in cohort:
        feats = extract_features(
            events,
            config,
            reference_time=meta["reference_time"],  # type: ignore[arg-type]
        )
        features.append(feats)
        labels.append(int(bool(meta["next_day_inactive"])))
        personas.append(str(meta["persona"]))
        oracle.append(float(meta["p_next_day_inactive"]))  # type: ignore[arg-type]

    return SimulatedDataset(
        features=features,
        labels=np.asarray(labels, dtype=int),
        personas=personas,
        oracle_probs=np.asarray(oracle, dtype=float),
    )


def evaluate_risk_models(
    features: Sequence[BehaviouralFeatures] | None = None,
    labels: np.ndarray | None = None,
    *,
    n_folds: int = 5,
    seed: int = 0,
    kinds: Sequence[ModelKind] = ("logistic", "gradient_boosting"),
    config: AnalyzerConfig = DEFAULT_CONFIG,
    oracle_probs: np.ndarray | None = None,
) -> EvaluationReport:
    """Cross-validate every risk predictor and compare them head to head.

    Args:
        features: Pre-extracted features. A default cohort is simulated if None.
        labels: Binary next-day-inactive labels, aligned with ``features``.
        n_folds: Stratified CV folds.
        seed: Reproducibility seed.
        kinds: Which learned models to include.
        config: Analyzer config, used by the heuristic baseline.
        oracle_probs: True generating probabilities, when known. Adds a Bayes
            ceiling row showing the best score any predictor could achieve given
            the irreducible noise in the labels.

    Returns:
        An :class:`EvaluationReport` with per-predictor mean metrics.
    """
    if features is None or labels is None:
        dataset = build_dataset(seed=seed or 20240, config=config)
        features, labels = dataset.features, dataset.labels
        if oracle_probs is None:
            oracle_probs = dataset.oracle_probs

    feature_list = list(features)
    y = np.asarray(labels, dtype=int)
    X = features_to_matrix(feature_list)

    # Baselines are deterministic functions of the features, so they need no
    # training and are simply scored on each fold's test rows.
    baselines: dict[str, Callable[[BehaviouralFeatures], float]] = {
        "heuristic (rule-based)": lambda f: heuristic_risk(f, config),
    }

    per_fold: dict[str, list[MetricSet]] = {name: [] for name in baselines}
    for kind in kinds:
        per_fold[f"model: {kind}"] = []
    per_fold["base rate (no skill)"] = []
    if oracle_probs is not None:
        per_fold["oracle (Bayes ceiling)"] = []

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        train_feats = [feature_list[i] for i in train_idx]
        test_feats = [feature_list[i] for i in test_idx]

        # Reference point: predict the training base rate for everyone.
        base = float(y_train.mean())
        per_fold["base rate (no skill)"].append(compute_metrics(y_test, np.full(len(y_test), base)))

        for name, fn in baselines.items():
            per_fold[name].append(compute_metrics(y_test, [fn(f) for f in test_feats]))

        if oracle_probs is not None:
            per_fold["oracle (Bayes ceiling)"].append(
                compute_metrics(y_test, np.asarray(oracle_probs)[test_idx])
            )

        for kind in kinds:
            model = RiskModel.fit(train_feats, y_train.astype(bool), kind=kind, seed=seed)
            per_fold[f"model: {kind}"].append(
                compute_metrics(y_test, model.predict_proba(test_feats))
            )

    means: dict[str, MetricSet] = {}
    stds: dict[str, float] = {}
    for name, folds in per_fold.items():
        means[name] = MetricSet(
            roc_auc=float(np.mean([m.roc_auc for m in folds])),
            average_precision=float(np.mean([m.average_precision for m in folds])),
            brier=float(np.mean([m.brier for m in folds])),
            calibration_error=float(np.mean([m.calibration_error for m in folds])),
            n=int(np.sum([m.n for m in folds])),
        )
        stds[name] = float(np.std([m.roc_auc for m in folds]))

    # Importances come from a model trained on everything, purely for reporting.
    full_model = RiskModel.fit(feature_list, y.astype(bool), kind="logistic", seed=seed)
    coefficients = full_model.coefficients() or {}
    importance = full_model.permutation_importance_scores(feature_list, y.astype(bool), seed=seed)

    return EvaluationReport(
        means=means,
        roc_auc_std=stds,
        n_folds=n_folds,
        n_samples=int(len(y)),
        base_rate=float(y.mean()),
        feature_importance=importance,
        logistic_coefficients=coefficients,
    )


@dataclass
class PatternEvaluation:
    """Scores for the rule-based pattern classifier against simulator labels."""

    accuracy: float
    n: int
    per_persona_recall: dict[str, float]
    mean_confidence: float
    confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form, used by the CLI's ``--format json``."""
        return {
            "accuracy": self.accuracy,
            "n": self.n,
            "per_persona_recall": self.per_persona_recall,
            "mean_confidence": self.mean_confidence,
            "confusion": self.confusion,
        }


def evaluate_pattern_rules(
    events_and_meta: Sequence[tuple[EventFrame, dict[str, object]]] | None = None,
    *,
    n_per_persona: int = 40,
    seed: int = 7,
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> PatternEvaluation:
    """Score the rule-based pattern classifier against simulator ground truth.

    Returns accuracy, per-persona recall, and the full confusion matrix. Because
    the rules were authored by hand against this same generative process, treat
    this as a *consistency check* — evidence that the rules encode what they
    claim to — and not as evidence of real-world validity.
    """
    if events_and_meta is None:
        events_and_meta = simulate_cohort(n_per_persona=n_per_persona, seed=seed)

    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    total = 0
    confidences: list[float] = []

    for events, meta in events_and_meta:
        persona = str(meta["persona"])
        feats = extract_features(events, config, reference_time=meta["reference_time"])  # type: ignore[arg-type]
        result = classify_pattern(feats, config)
        row = confusion.setdefault(persona, {})
        row[result.pattern.value] = row.get(result.pattern.value, 0) + 1
        expected = PERSONA_TO_PATTERN.get(persona)
        is_correct = expected is not None and result.pattern == expected
        correct += int(is_correct)
        total += 1
        confidences.append(result.confidence)

    recall = {
        persona: round(row.get(PERSONA_TO_PATTERN[persona].value, 0) / max(1, sum(row.values())), 4)
        for persona, row in confusion.items()
        if persona in PERSONA_TO_PATTERN
    }

    return PatternEvaluation(
        accuracy=round(correct / max(1, total), 4),
        n=total,
        per_persona_recall=recall,
        mean_confidence=round(float(np.mean(confidences)) if confidences else 0.0, 4),
        confusion=confusion,
    )
