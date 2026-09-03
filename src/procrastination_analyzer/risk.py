"""Learned next-day inactivity risk models.

The heuristic in :func:`procrastination_analyzer.patterns.heuristic_risk` is a
hand-weighted linear combination. This module provides learned alternatives and
— importantly — the machinery to check whether learning actually *helps*, since
a model that fails to beat a well-chosen heuristic is not worth deploying.

Two things matter more than raw discrimination here:

* **Calibration.** The output is shown to a user as "risk", so a predicted 0.7
  should mean roughly a 70% chance. ROC-AUC is invariant to monotone rescaling
  and so cannot detect miscalibration; Brier score and reliability curves can.
  Both models are therefore wrapped in :class:`~sklearn.calibration.CalibratedClassifierCV`.
* **Honest validation.** Grouped, stratified cross-validation only — no fitting
  and scoring on the same rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, BehaviouralFeatures

__all__ = [
    "LabelArray",
    "ModelKind",
    "RiskModel",
    "build_estimator",
    "features_to_matrix",
]

ModelKind = Literal["logistic", "gradient_boosting"]

#: Binary labels accepted by the training and scoring entry points. Numpy arrays
#: are the common case in the evaluation harness, plain sequences in user code.
LabelArray = Sequence[bool] | Sequence[int] | np.ndarray


def build_estimator(kind: ModelKind = "logistic", *, seed: int = 0) -> Pipeline:
    """Construct an untrained, calibrated pipeline.

    Args:
        kind: ``"logistic"`` for the interpretable linear baseline (coefficients
            are directly readable as log-odds contributions), or
            ``"gradient_boosting"`` for a non-linear model that can capture
            interactions such as "long gaps *only* matter when coverage is low".
        seed: Seed for reproducibility.

    Returns:
        A pipeline of scaler -> calibrated classifier.
    """
    if kind == "logistic":
        base: object = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
        )
    elif kind == "gradient_boosting":
        base = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.06, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model kind {kind!r}; expected 'logistic' or 'gradient_boosting'")

    # Isotonic calibration needs more data than we typically have per fold, so
    # sigmoid (Platt) scaling is the safer default here.
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("scale", StandardScaler()), ("clf", calibrated)])


def features_to_matrix(features: Sequence[BehaviouralFeatures]) -> np.ndarray:
    """Stack extracted features into an ``(n_samples, n_features)`` matrix."""
    if not features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.vstack([f.to_vector() for f in features])


@dataclass
class RiskModel:
    """A trained, calibrated next-day inactivity classifier."""

    pipeline: Pipeline
    kind: ModelKind
    feature_names: tuple[str, ...] = FEATURE_NAMES
    #: Training-set positive rate, used as the no-skill reference point.
    base_rate: float = 0.5
    n_training_samples: int = 0

    @classmethod
    def fit(
        cls,
        features: Sequence[BehaviouralFeatures],
        labels: LabelArray,
        *,
        kind: ModelKind = "logistic",
        seed: int = 0,
    ) -> RiskModel:
        """Train on extracted features and binary next-day-inactive labels."""
        X = features_to_matrix(features)
        y = np.asarray(labels, dtype=int)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/label length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels contain only one class; cannot fit a classifier.")

        pipeline = build_estimator(kind, seed=seed)
        pipeline.fit(X, y)
        return cls(
            pipeline=pipeline,
            kind=kind,
            base_rate=float(y.mean()),
            n_training_samples=int(len(y)),
        )

    def predict_proba(
        self, features: BehaviouralFeatures | Sequence[BehaviouralFeatures]
    ) -> np.ndarray:
        """Predicted probability of next-day inactivity, one value per input."""
        batch: Sequence[BehaviouralFeatures] = (
            [features] if isinstance(features, BehaviouralFeatures) else features
        )
        X = features_to_matrix(batch)
        return np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=float)

    def predict_one(self, features: BehaviouralFeatures) -> float:
        """Convenience wrapper returning a single float risk."""
        return float(self.predict_proba(features)[0])

    def coefficients(self) -> dict[str, float] | None:
        """Mean log-odds coefficients, for the logistic model only.

        Returns ``None`` for non-linear models, where
        :meth:`permutation_importance_scores` is the right tool instead.
        """
        if self.kind != "logistic":
            return None
        calibrated = self.pipeline.named_steps["clf"]
        coefs = [
            cc.estimator.coef_[0]
            for cc in calibrated.calibrated_classifiers_
            if hasattr(cc.estimator, "coef_")
        ]
        if not coefs:
            return None
        mean_coef = np.mean(coefs, axis=0)
        return {
            name: round(float(value), 4)
            for name, value in sorted(
                zip(self.feature_names, mean_coef, strict=True),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
        }

    def permutation_importance_scores(
        self,
        features: Sequence[BehaviouralFeatures],
        labels: LabelArray,
        *,
        n_repeats: int = 10,
        seed: int = 0,
    ) -> dict[str, float]:
        """Model-agnostic feature importance via permutation on held-out data.

        Measures the drop in ROC-AUC when each feature is shuffled. Unlike
        impurity-based importances this does not inflate the ranking of
        high-cardinality features, and it works for any estimator.
        """
        X = features_to_matrix(features)
        y = np.asarray(labels, dtype=int)
        result = permutation_importance(
            self.pipeline, X, y, n_repeats=n_repeats, random_state=seed, scoring="roc_auc"
        )
        return {
            name: round(float(value), 4)
            for name, value in sorted(
                zip(self.feature_names, result.importances_mean, strict=True),
                key=lambda kv: kv[1],
                reverse=True,
            )
        }

    def save(self, path: str | Path) -> Path:
        """Persist the trained model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> RiskModel:
        """Load a model previously written by :meth:`save`."""
        import joblib

        model = joblib.load(Path(path))
        if not isinstance(model, RiskModel):
            raise TypeError(f"{path} does not contain a RiskModel (got {type(model).__name__})")
        return model


def top_risk_drivers(
    model: RiskModel, features: BehaviouralFeatures, k: int = 4
) -> list[tuple[str, float]]:
    """Per-prediction contributions for the linear model.

    For logistic regression the contribution of each feature to the log-odds is
    ``coef * standardised_value``, which decomposes an individual prediction
    exactly. Returns an empty list for non-linear models.
    """
    coefs = model.coefficients()
    if coefs is None:
        return []
    scaler: StandardScaler = model.pipeline.named_steps["scale"]
    z = (features.to_vector() - scaler.mean_) / np.sqrt(scaler.var_ + 1e-12)
    contributions = {
        name: float(coefs[name] * zi) for name, zi in zip(model.feature_names, z, strict=True)
    }
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [(n, round(v, 4)) for n, v in ranked[:k]]
