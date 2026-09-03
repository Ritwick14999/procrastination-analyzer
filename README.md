# Procrastination Pattern Analyzer

Behavioural analytics over activity timestamps. Given a log of *when* work happened —
git commits, task completions, any event stream with a time column — it extracts
scale-invariant rhythm features, classifies a behavioural pattern with an
interpretable rule engine, estimates next-day inactivity risk with calibrated
models, and retrieves relevant suggestions.

The project's organising question is not "can I build a dashboard?" but **"how would
I know whether any of this works?"** Behavioural side projects usually cannot answer
that, because there are no labels. This one ships a generative simulator that
*defines* ground truth, so every claim below is a measured number with a stated
validity boundary rather than an assertion.

> **Scope.** This is exploratory analysis of timing data. It is not a psychological
> or clinical instrument, and the scores carry no diagnostic meaning. See
> [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the limitations, which are
> substantial and stated plainly.

---

## Results

Next-day inactivity risk, 5-fold stratified cross-validation over 600 simulated
users (`procrastination-analyzer evaluate`):

| Predictor | ROC-AUC | Avg. precision | Brier ↓ | Calibration error ↓ |
|---|---|---|---|---|
| Oracle (Bayes ceiling) | 0.814 ± 0.026 | 0.729 | 0.169 | 0.077 |
| **Gradient boosting** | **0.757 ± 0.043** | 0.658 | **0.200** | **0.104** |
| **Heuristic (rule-based)** | **0.757 ± 0.029** | **0.703** | 0.201 | 0.104 |
| Logistic regression | 0.748 ± 0.051 | 0.629 | 0.209 | 0.120 |
| Base rate (no skill) | 0.500 ± 0.000 | 0.397 | 0.239 | 0.005 |

Two things are worth reading carefully here:

- **The learned model does not beat the hand-tuned heuristic.** Gradient boosting ties
  it on ROC-AUC and loses on average precision. This is reported rather than buried;
  a model that ties a transparent baseline is not worth the deployment cost, and the
  heuristic remains the pipeline default for exactly that reason.
- **The oracle row bounds what is achievable.** Labels are Bernoulli draws from a known
  probability, so no predictor can exceed ~0.814. Both leading predictors capture about
  82% of the available signal above chance — the remaining gap is mostly irreducible
  noise, not headroom a better model would close.

Rule-based pattern classification recovers the simulator's latent personas with
**95.0% accuracy** over 200 held-out users (per-persona recall: consistent, deadline-driven,
fatigued and nocturnal at 1.00; avoidant at 0.75, which is where the residual error sits).

Reproduce both tables with:

```bash
procrastination-analyzer evaluate --patterns
```

---

## Install and run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/ritwick14999/procrastination-analyzer.git
cd procrastination-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
```

Then pick an entry point:

```bash
make app          # Streamlit dashboard at http://localhost:8501
make cli          # analyse the packaged sample dataset
make evaluate     # reproduce the results tables above
make check        # lint + typecheck + tests, exactly what CI runs
```

Working in VS Code? See [`docs/VSCODE.md`](docs/VSCODE.md) — the repo ships
`launch.json`, test discovery, and a devcontainer.

### Command line

```bash
# Analyse any CSV with a timestamp column (auto-detected)
procrastination-analyzer analyze mylog.csv

# Machine-readable output, or a shareable report
procrastination-analyzer analyze mylog.csv --format json
procrastination-analyzer analyze mylog.csv --format markdown -o report.md

# Score a live user: recency features need to know what "now" is
procrastination-analyzer analyze mylog.csv --now "2026-09-03 18:00"

# Train a model, then use it instead of the heuristic
procrastination-analyzer train -o artifacts/risk.joblib
procrastination-analyzer analyze mylog.csv --model artifacts/risk.joblib

# Generate labelled synthetic data
procrastination-analyzer simulate --persona avoidant --days 90 -o cohort.csv
```

### Python API

```python
from procrastination_analyzer import analyze

result = analyze(["2025-01-06 09:15", "2025-01-08 23:40", "2025-01-13 22:05"])

print(result.pattern.pattern.value)   # 'Avoidance-driven'
print(result.risk, result.risk_band)  # 0.61 High
print(result.pattern.evidence)        # why the classifier decided that
print(result.warnings)                # data-quality caveats, if any
```

Any of these work as input: a DataFrame, a Series, a list of strings, or a CSV path.
Mixed UTC offsets, unparseable rows and duplicate timestamps are handled and *reported*
rather than silently dropped.

---

## How it works

```
timestamps ─► schema.py ─► features.py ─► patterns.py ─┬─► retrieval.py ─► report.py
              validate     15 scale-      rule engine  │   TF-IDF          markdown
              normalise    invariant      + evidence   └─► risk.py         / JSON
              deduplicate  features                        calibrated ML
```

| Module | Responsibility |
|---|---|
| `schema.py` | Validates and normalises input into a sorted, deduplicated `EventFrame`. Handles mixed timezones. |
| `features.py` | 15 scale-invariant features: gap structure, session/burst structure, circadian rhythm, trend. |
| `patterns.py` | Additive rule engine producing a pattern, a confidence, and the evidence that fired. |
| `risk.py` | Calibrated logistic / gradient-boosting models with permutation importance and per-prediction attribution. |
| `simulate.py` | Generative persona model producing labelled cohorts from a seed. |
| `evaluate.py` | Cross-validated comparison against the base rate, the heuristic, and the Bayes ceiling. |
| `retrieval.py` | Fit-once TF-IDF index over 122 deduplicated suggestion snippets. |
| `config.py` | Every threshold in the pipeline, validated on construction. |

### Design decisions worth calling out

**Features are scale-invariant by construction.** Every feature is a rate or a share,
never a raw count. A user with six months of data must not score worse than an
identically-behaved user with one month. This is enforced by test, not by intention:
`test_share_features_are_exactly_stable` asserts that a repeating rhythm produces
identical features at 4 weeks and 32 weeks.

**Weekend hours are discounted from gaps.** A Friday-evening to Monday-morning silence
is a weekend, not avoidance. Gap length is measured in *active* hours, so ordinary
Mon–Fri workers are not flagged. Toggleable via `exclude_weekend_from_gaps`.

**Clock time is treated as circular.** 23:00 and 01:00 are two hours apart, not
twenty-two. Rhythm irregularity uses circular dispersion, and the "late night" band
wraps past midnight, so a consistent after-midnight worker reads as *regular and
nocturnal* rather than *maximally erratic*.

**Sessions are disjoint.** Bursts are detected over segmented sessions, so a single
three-hour sitting counts once rather than as many overlapping sub-windows.

**Calibration is measured, not assumed.** The risk number is shown to a human, so a
predicted 0.7 should mean roughly a 70% chance. ROC-AUC cannot detect miscalibration —
it is invariant to any monotone rescaling — so Brier score and expected calibration
error are reported alongside it, and both models are wrapped in Platt scaling.

**Thin data is labelled as thin.** Below 20 events, or under two weeks of span, the
output carries explicit low-confidence warnings instead of quietly presenting a number
with the same visual weight as a well-supported one.

---

## Development

```bash
make dev        # install with dev + app extras
make check      # ruff + mypy + pytest, the full CI gate
make cov        # coverage report
```

135 tests, 96% line coverage. CI runs lint, format check, `mypy --strict`-adjacent
settings, the test matrix on Python 3.10/3.11/3.12, and an end-to-end CLI smoke test.

A meaningful share of the suite is **regression tests for specific defects**, each
naming the behaviour it protects — scale drift, overlapping burst windows, weekend
gaps, midnight wrapping, a risk term that was structurally always zero, and a
sub-day record scoring as exemplary consistency. See
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#defects-found-and-fixed) for the details.

## Documentation

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — features, validation design, defects
  found, and an honest account of what the numbers do and do not establish.
- [`docs/VSCODE.md`](docs/VSCODE.md) — local setup, debugging, and running the app.

## License

MIT.
