"""End-to-end pipeline, report and CLI tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from procrastination_analyzer import analyze
from procrastination_analyzer.cli import main
from procrastination_analyzer.config import DEFAULT_CONFIG
from procrastination_analyzer.pipeline import analyze_file
from procrastination_analyzer.report import render_markdown, write_report
from procrastination_analyzer.retrieval import default_snippets_path
from procrastination_analyzer.schema import InsufficientDataError
from procrastination_analyzer.simulate import simulate_user

SAMPLE = default_snippets_path().parent / "sample_commits.csv"


@pytest.fixture
def csv_file(tmp_path):
    """A written-out CSV of a simulated avoidant user."""
    import numpy as np

    events, _ = simulate_user("avoidant", days=70, rng=np.random.default_rng(4))
    path = tmp_path / "log.csv"
    pd.DataFrame({"timestamp": events.ts}).to_csv(path, index=False)
    return path


class TestPipeline:
    def test_analyzes_raw_timestamp_list(self):
        result = analyze(["2025-01-06 09:00", "2025-01-08 23:00", "2025-01-14 22:30"])
        assert result.pattern.pattern is not None
        assert 0.0 <= result.risk <= 1.0
        assert result.risk_source == "heuristic"

    def test_packaged_sample_data_analyzes(self):
        result = analyze_file(SAMPLE)
        assert result.features.n_events == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_file(tmp_path / "nope.csv")

    def test_too_few_events_raises(self):
        with pytest.raises(InsufficientDataError):
            analyze(["2025-01-06 09:00"])

    def test_warns_about_thin_records(self):
        result = analyze(["2025-01-06 09:00", "2025-01-06 10:00", "2025-01-07 09:00"])
        assert any("events analysed" in w for w in result.warnings)

    def test_warns_when_span_is_too_short_for_rhythm(self):
        result = analyze(["2025-01-06 15:00", "2025-01-06 15:20", "2025-01-06 15:40"])
        assert any("day-to-day consistency" in w for w in result.warnings)

    def test_reports_dropped_rows(self):
        result = analyze(["2025-01-06 09:00", "garbage", "2025-01-08 10:00", "2025-01-11 09:00"])
        assert any("unparseable" in w for w in result.warnings)

    def test_model_changes_the_risk_source(self, csv_file):
        from procrastination_analyzer.evaluate import build_dataset
        from procrastination_analyzer.risk import RiskModel

        dataset = build_dataset(n_per_persona=20, seed=6)
        model = RiskModel.fit(dataset.features, dataset.labels.astype(bool), kind="logistic")
        result = analyze_file(csv_file, model=model)
        assert result.risk_source == "model:logistic"
        assert result.risk_drivers  # linear model exposes per-feature contributions

    def test_category_override_is_respected(self):
        result = analyze(
            ["2025-01-06 09:00", "2025-01-08 23:00", "2025-01-14 22:30"],
            category_override="fatigue",
        )
        assert {s["category"] for s in result.suggestions} == {"fatigue"}

    def test_config_is_recorded_for_reproducibility(self):
        result = analyze(["2025-01-06 09:00", "2025-01-08 23:00", "2025-01-14 22:30"])
        assert result.config_used["long_gap_h"] == DEFAULT_CONFIG.long_gap_h

    def test_to_dict_is_json_serialisable(self, csv_file):
        payload = json.dumps(analyze_file(csv_file).to_dict())
        assert "summary" in json.loads(payload)


class TestReport:
    def test_markdown_contains_the_key_sections(self, csv_file):
        text = render_markdown(analyze_file(csv_file))
        for heading in ("# Procrastination", "## Summary", "## Suggestions", "## Configuration"):
            assert heading in text

    def test_markdown_carries_a_non_clinical_disclaimer(self, csv_file):
        assert "not a psychological" in render_markdown(analyze_file(csv_file))

    def test_writes_markdown_and_creates_parents(self, csv_file, tmp_path):
        path = write_report(analyze_file(csv_file), tmp_path / "a" / "b" / "r.md")
        assert path.exists() and path.read_text(encoding="utf-8").startswith("#")

    def test_writes_json(self, csv_file, tmp_path):
        path = write_report(analyze_file(csv_file), tmp_path / "r.json", fmt="json")
        assert "features" in json.loads(path.read_text(encoding="utf-8"))

    def test_rejects_unknown_format(self, csv_file, tmp_path):
        with pytest.raises(ValueError, match="Unknown report format"):
            write_report(analyze_file(csv_file), tmp_path / "r.pdf", fmt="pdf")


class TestCli:
    def test_analyze_text_output(self, csv_file, capsys):
        assert main(["analyze", str(csv_file)]) == 0
        assert "PROCRASTINATION PATTERN ANALYSIS" in capsys.readouterr().out

    def test_analyze_json_output(self, csv_file, capsys):
        assert main(["analyze", str(csv_file), "--format", "json"]) == 0
        assert "summary" in json.loads(capsys.readouterr().out)

    def test_analyze_writes_markdown_file(self, csv_file, tmp_path):
        out = tmp_path / "report.md"
        assert main(["analyze", str(csv_file), "--format", "markdown", "-o", str(out)]) == 0
        assert out.exists()

    def test_analyze_accepts_a_reference_time(self, csv_file, capsys):
        assert main(["analyze", str(csv_file), "--now", "2030-01-01"]) == 0
        assert "Next-day risk" in capsys.readouterr().out

    def test_missing_file_exits_nonzero(self, tmp_path, capsys):
        assert main(["analyze", str(tmp_path / "absent.csv")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_simulate_emits_csv(self, capsys):
        assert main(["simulate", "--persona", "consistent", "--days", "30"]) == 0
        assert "timestamp,persona,user_id" in capsys.readouterr().out

    def test_simulate_rejects_unknown_persona(self, capsys):
        assert main(["simulate", "--persona", "wizard"]) == 2
        assert "Unknown persona" in capsys.readouterr().err

    def test_simulate_cohort_to_file(self, tmp_path):
        out = tmp_path / "cohort.csv"
        assert main(["simulate", "-n", "2", "--days", "30", "-o", str(out)]) == 0
        assert len(pd.read_csv(out)) > 0

    def test_train_writes_a_model(self, tmp_path):
        out = tmp_path / "model.joblib"
        assert main(["train", "-n", "20", "-o", str(out)]) == 0
        assert out.exists()

    def test_evaluate_json_includes_predictors(self, tmp_path):
        out = tmp_path / "eval.json"
        assert (
            main(["evaluate", "-n", "20", "--folds", "3", "--format", "json", "-o", str(out)]) == 0
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "heuristic (rule-based)" in payload["risk"]["predictors"]

    def test_evaluate_with_patterns_flag(self, tmp_path):
        out = tmp_path / "eval.json"
        assert (
            main(
                [
                    "evaluate",
                    "-n",
                    "20",
                    "--folds",
                    "3",
                    "--patterns",
                    "--format",
                    "json",
                    "-o",
                    str(out),
                ]
            )
            == 0
        )
        assert "accuracy" in json.loads(out.read_text(encoding="utf-8"))["patterns"]
