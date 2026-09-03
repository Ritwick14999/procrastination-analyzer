"""Simulator, risk model, and evaluation harness tests."""

from __future__ import annotations

import numpy as np
import pytest

from procrastination_analyzer.evaluate import (
    build_dataset,
    compute_metrics,
    evaluate_pattern_rules,
    evaluate_risk_models,
    expected_calibration_error,
)
from procrastination_analyzer.features import extract_features
from procrastination_analyzer.risk import RiskModel, build_estimator
from procrastination_analyzer.simulate import PERSONAS, simulate_cohort, simulate_user


class TestSimulator:
    @pytest.mark.parametrize("persona", sorted(PERSONAS))
    def test_every_persona_generates_analysable_data(self, persona):
        events, meta = simulate_user(persona, days=60, rng=np.random.default_rng(3))
        assert len(events) >= 3
        assert meta["persona"] == persona
        assert 0.0 <= float(meta["p_next_day_inactive"]) <= 1.0

    def test_unknown_persona_raises(self):
        with pytest.raises(KeyError, match="Unknown persona"):
            simulate_user("wizard")

    def test_same_seed_gives_identical_output(self):
        a, _ = simulate_user("avoidant", days=45, rng=np.random.default_rng(99))
        b, _ = simulate_user("avoidant", days=45, rng=np.random.default_rng(99))
        assert a.ts.equals(b.ts)

    def test_different_seeds_differ(self):
        a, _ = simulate_user("avoidant", days=45, rng=np.random.default_rng(1))
        b, _ = simulate_user("avoidant", days=45, rng=np.random.default_rng(2))
        assert not a.ts.equals(b.ts)

    def test_cohort_is_balanced_and_reproducible(self):
        first = simulate_cohort(n_per_persona=4, seed=5)
        second = simulate_cohort(n_per_persona=4, seed=5)
        assert len(first) == 4 * len(PERSONAS)
        assert [len(e) for e, _ in first] == [len(e) for e, _ in second]

    def test_personas_are_behaviourally_distinct(self):
        """Consistent workers must show broader coverage than avoidant ones."""

        def mean_coverage(persona: str) -> float:
            rng = np.random.default_rng(21)
            return float(
                np.mean(
                    [
                        extract_features(simulate_user(persona, days=60, rng=rng)[0]).coverage
                        for _ in range(12)
                    ]
                )
            )

        assert mean_coverage("consistent") > mean_coverage("avoidant") + 0.2


class TestCalibrationMetric:
    def test_perfect_calibration_scores_zero(self):
        probs = np.full(200, 0.5)
        labels = np.array([0, 1] * 100)
        assert expected_calibration_error(labels, probs) < 1e-9

    def test_confidently_wrong_scores_high(self):
        probs = np.full(100, 0.95)
        labels = np.zeros(100, dtype=int)
        assert expected_calibration_error(labels, probs) == pytest.approx(0.95, abs=1e-6)

    def test_handles_empty_input(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_probability_zero_lands_in_first_bin(self):
        assert expected_calibration_error(np.zeros(10), np.zeros(10)) == 0.0


class TestMetrics:
    def test_single_class_fold_does_not_crash(self):
        metrics = compute_metrics([1, 1, 1], [0.4, 0.6, 0.8])
        assert metrics.roc_auc == 0.5

    def test_perfect_ranking_scores_one(self):
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert metrics.roc_auc == 1.0


class TestRiskModel:
    @staticmethod
    @pytest.fixture(scope="class")
    def dataset():
        return build_dataset(n_per_persona=25, seed=77)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown model kind"):
            build_estimator("magic")  # type: ignore[arg-type]

    def test_rejects_single_class_labels(self, dataset):
        with pytest.raises(ValueError, match="only one class"):
            RiskModel.fit(dataset.features, [True] * len(dataset.features))

    def test_rejects_length_mismatch(self, dataset):
        with pytest.raises(ValueError, match="length mismatch"):
            RiskModel.fit(dataset.features, dataset.labels[:-3].astype(bool))

    @pytest.mark.parametrize("kind", ["logistic", "gradient_boosting"])
    def test_predictions_are_valid_probabilities(self, dataset, kind):
        model = RiskModel.fit(dataset.features, dataset.labels.astype(bool), kind=kind)
        probs = model.predict_proba(dataset.features)
        assert probs.shape == (len(dataset.features),)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_predict_one_matches_batch(self, dataset):
        model = RiskModel.fit(dataset.features, dataset.labels.astype(bool))
        single = model.predict_one(dataset.features[0])
        batch = model.predict_proba(dataset.features[:1])[0]
        assert single == pytest.approx(batch)

    def test_logistic_exposes_coefficients(self, dataset):
        model = RiskModel.fit(dataset.features, dataset.labels.astype(bool), kind="logistic")
        coefs = model.coefficients()
        assert coefs is not None
        assert set(coefs) == set(model.feature_names)

    def test_tree_model_has_no_coefficients(self, dataset):
        model = RiskModel.fit(
            dataset.features, dataset.labels.astype(bool), kind="gradient_boosting"
        )
        assert model.coefficients() is None

    def test_roundtrips_through_disk(self, dataset, tmp_path):
        model = RiskModel.fit(dataset.features, dataset.labels.astype(bool))
        path = model.save(tmp_path / "nested" / "model.joblib")
        restored = RiskModel.load(path)
        assert restored.predict_one(dataset.features[0]) == pytest.approx(
            model.predict_one(dataset.features[0])
        )

    def test_loading_a_non_model_raises(self, tmp_path):
        import joblib

        path = tmp_path / "junk.joblib"
        joblib.dump({"not": "a model"}, path)
        with pytest.raises(TypeError, match="does not contain a RiskModel"):
            RiskModel.load(path)


class TestEvaluation:
    def test_models_beat_the_no_skill_baseline(self):
        dataset = build_dataset(n_per_persona=40, seed=31)
        report = evaluate_risk_models(
            dataset.features, dataset.labels, n_folds=3, oracle_probs=dataset.oracle_probs
        )
        assert report.means["base rate (no skill)"].roc_auc == pytest.approx(0.5)
        for name in ("heuristic (rule-based)", "model: gradient_boosting"):
            assert report.means[name].roc_auc > 0.6, name

    def test_oracle_bounds_every_learned_predictor(self):
        """No predictor should beat the Bayes ceiling by a meaningful margin."""
        dataset = build_dataset(n_per_persona=40, seed=31)
        report = evaluate_risk_models(
            dataset.features, dataset.labels, n_folds=3, oracle_probs=dataset.oracle_probs
        )
        ceiling = report.means["oracle (Bayes ceiling)"].roc_auc
        for name, metrics in report.means.items():
            if "oracle" not in name:
                assert metrics.roc_auc <= ceiling + 0.05, name

    def test_markdown_table_renders(self):
        dataset = build_dataset(n_per_persona=20, seed=8)
        report = evaluate_risk_models(dataset.features, dataset.labels, n_folds=3)
        table = report.to_markdown()
        assert "ROC-AUC" in table and "heuristic" in table

    def test_pattern_rules_recover_the_simulated_personas(self):
        result = evaluate_pattern_rules(n_per_persona=20, seed=404)
        assert result.accuracy > 0.75
        assert set(result.per_persona_recall) == set(PERSONAS)
