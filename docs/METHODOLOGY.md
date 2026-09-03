# Methodology

This document states what the system measures, how it was validated, and — at
some length, because it is the part most easily overstated — what the results do
**not** establish.

---

## 1. Problem framing

Given a sequence of activity timestamps for one person, produce:

1. a **behavioural pattern** label with supporting evidence,
2. an **avoidance score** in [0, 1],
3. a **next-day inactivity risk** in [0, 1], and
4. relevant **suggestions** retrieved from a curated corpus.

Only *timing* is observed. There is no task content, no difficulty, no stated
intent, no self-report. That constraint is the single most important thing about
this project: it bounds what can honestly be claimed. Timing data can show that
someone works in irregular late-night bursts after multi-day silences. It cannot
show *why*, and "procrastination" is a claim about why. The labels here are names
for timing shapes, not psychological findings.

---

## 2. Feature design

### 2.1 The scale-invariance requirement

Every feature must be a **rate** or a **share**, never a raw count.

The motivating failure: a score built from counts (`0.55 * long_gaps + 0.35 *
bursts + ...`) makes the output a function of how much data the user happens to
have. Two users with *identical* weekly rhythms score differently if one has
logged six months and the other one month. The score then measures record length
wearing the costume of a behavioural finding.

This is enforced by test rather than by intent. `TestScaleInvariance` asserts
that shares are bit-identical between a 4-week and a 32-week record of the same
repeating rhythm, and that the gap rate *converges* rather than drifts.

### 2.2 The feature set

| Group | Features | Rationale |
|---|---|---|
| Coverage | `coverage`, `events_per_active_day` | Consistency independent of record length. |
| Gaps | `long_gap_rate`, `extended_gap_rate`, `max_gap_hours`, `median_gap_hours` | Silence structure. Median resists a single outlier. |
| Sessions | `n_sessions`, `burst_event_share`, `mean_session_hours`, `mean_session_events` | Work shape: steady vs. crammed. |
| Circadian | `late_night_share`, `evening_share`, `working_hours_share`, `weekend_share`, `rhythm_irregularity` | When work happens, and how predictably. |
| Trend | `recency_trend`, `hours_since_last_event` | Direction of travel. |

Three choices in this set are non-obvious and worth defending.

**Weekend-adjusted gaps.** Gap length is measured in *active* hours, with weekend
hours subtracted. A Friday 18:00 → Monday 09:00 silence spans 63 clock hours but
only ~15 working hours. Counting the raw 63 flags every ordinary Monday-to-Friday
professional as an avoider — a false positive rate that would be near 100% on the
target population. Configurable via `exclude_weekend_from_gaps`.

**Circular rhythm dispersion.** Clock hours live on a circle: 23:00 and 01:00 are
two hours apart, not twenty-two. Linear variance over the hour column reports a
perfectly punctual midnight worker as maximally erratic. `rhythm_irregularity`
instead maps hours onto the unit circle and uses `1 - R̄` (one minus the mean
resultant length), giving 0 for a fixed hour and approaching 1 for uniform spread.
The "late night" band wraps past midnight for the same reason.

**Disjoint sessions.** Events are segmented into sessions by an inactivity
timeout, and bursts are counted over those sessions. Sliding windows over raw
event indices count one long sitting many times: 20 events in a single stretch
yields 16 overlapping "5-events-in-2-hours" windows, which then inflates every
downstream score. Session segmentation makes it one session.

---

## 3. Validation design

### 3.1 The labelling problem, and an honest response

There is no labelled real-world dataset for this task, and self-reported
procrastination is a weak and biased target. Rather than assert that the
heuristics work, the project makes the ground truth explicit: a generative model
(`simulate.py`) samples a latent persona, generates a timestamp log from it, and
retains the persona as the label.

**This is a testbed for the estimator, not evidence about human behaviour.** It
answers "can this pipeline recover structure that is genuinely present?" — a real
and necessary question, and the one that catches implementation bugs. It does not
answer "do people behave like these personas?" Section 6 is explicit about the
gap.

The personas are deliberately **overlapping**, not separable. Perfectly separable
classes would make classification trivial and the accuracy number meaningless. The
avoidant and nocturnal personas in particular share clock hours and differ mainly
in regularity, which is precisely the discrimination that is hard and that a naive
rule set gets wrong.

Record lengths are jittered across the cohort (±20 days) *specifically* so that a
scale-dependent feature would show up as degraded performance. The evaluation is
designed to be able to fail in the way the original code failed.

### 3.2 Risk-model evaluation

Target: did the day after the observation window see any activity? Labels are
Bernoulli draws from a persona-conditioned probability that is also modulated by
how the record actually ended, so the label correlates with observable behaviour
rather than being pure persona noise.

Protocol: 5-fold stratified cross-validation, 600 simulated users. Four metrics:

- **ROC-AUC** — ranking quality, threshold-independent.
- **Average precision** — more informative under class imbalance.
- **Brier score** — squared error of the probabilities themselves.
- **Expected calibration error** — mean |predicted − observed| across bins.

Both AUC *and* calibration metrics are reported because they answer different
questions. AUC is invariant to any monotone rescaling of the scores, so a model
that ranks perfectly while being wildly overconfident scores 1.0. Since the risk
value is shown to a person as a probability, that invariance is exactly the wrong
property to optimise alone.

### 3.3 Baselines

A model is only interesting relative to what it beats, so every run reports:

1. **Base rate** — predict the training positive rate for everyone. Floor.
2. **Heuristic** — the transparent rule-based score. The bar a learned model must
   clear to justify its complexity.
3. **Oracle** — the true generating probability. The Bayes ceiling; because labels
   are stochastic draws, no predictor can do better, and quoting a model's AUC
   without this context makes 0.76 look like failure when it is near-optimal.

---

## 4. Results and interpretation

| Predictor | ROC-AUC | Avg. precision | Brier ↓ | ECE ↓ |
|---|---|---|---|---|
| Oracle (Bayes ceiling) | 0.814 ± 0.026 | 0.729 | 0.169 | 0.077 |
| Gradient boosting | 0.757 ± 0.043 | 0.658 | 0.200 | 0.104 |
| Heuristic | 0.757 ± 0.029 | 0.703 | 0.201 | 0.104 |
| Logistic regression | 0.748 ± 0.051 | 0.629 | 0.209 | 0.120 |
| Base rate | 0.500 ± 0.000 | 0.397 | 0.239 | 0.005 |

**The learned models do not beat the heuristic.** Gradient boosting ties it on AUC
and is worse on average precision. Reporting this rather than quietly dropping the
baseline is the point: the honest conclusion is that on this data, at this sample
size, a well-designed heuristic is competitive with a learned model, and the
heuristic stays the pipeline default because it is inspectable and needs no
artifact to ship.

**Both leading predictors sit close to the achievable ceiling.** Normalising against
chance, they capture (0.757 − 0.5) / (0.814 − 0.5) ≈ **82%** of the available signal.
Most of the residual is irreducible label noise, not modelling headroom.

**Where the remaining error lives.** Pattern classification reaches 95.0% accuracy,
with per-persona recall of 1.00 for consistent, deadline-driven, fatigued and
nocturnal, and **0.75 for avoidant** — avoidant users leak into the deadline-driven
class, which makes sense: an avoidant user who eventually crams looks, in the
window observed, exactly like someone who planned to cram.

Rule thresholds were tuned against one cohort (seed 11) and then checked on five
unseen seeds. Held-out accuracy averaged 0.974 (sd 0.018) against 0.955 on the
tuning seed itself — that is, performance was *better* on cohorts the thresholds
were never exposed to, which is evidence the hand-tuning did not overfit.

---

## 5. Defects found and fixed

Rewriting the analysis surfaced six substantive defects. Five were in the original
implementation; the sixth surfaced while dogfooding the rewrite against this
repository's own git history. Each now has a named regression test.

| # | Defect | Consequence | Test |
|---|---|---|---|
| 1 | `predict_risk` computed inactivity as `last_ts − last_week.max()` | Structurally **always zero** — `last_ts` *is* that maximum. The term contributed nothing to any prediction. | `test_inactivity_actually_moves_the_estimate` |
| 2 | Scores summed raw counts normalised by `max(1, n/25)` | Scores tracked record length, not behaviour; identical rhythms scored 0.34 at 2 weeks and 0.67 at 32. | `TestScaleInvariance` |
| 3 | Bursts counted over overlapping sliding windows | One 20-event sitting reported as **16 bursts**, inflating every downstream score. | `test_one_sitting_is_one_session` |
| 4 | Long gaps measured in raw clock hours | Every Mon–Fri worker flagged for weekend silence. | `test_weekend_silence_is_not_a_long_gap` |
| 5 | Late night tested `hour >= 23`; irregularity used linear variance | After-midnight work invisible; punctual midnight workers scored maximally irregular. | `test_after_midnight_counts_as_late_night`, `test_regular_midnight_worker_is_not_irregular` |
| 6 | Coverage and gap-freedom vouched for consistency regardless of span | A record confined to one afternoon trivially covers every day it spans, so nine commits in an hour classified as *Consistent / low-procrastination*. | `test_single_afternoon_burst_is_not_consistent` |

Defect 6 is worth noting for how it was found: not by reasoning about the code, but by running the finished tool on this repository's own commit history and reading the output critically. Defects 1 and 2 are the consequential ones. Defect 2 in particular is the kind that
never announces itself: the system produced plausible-looking numbers throughout,
and only measuring against ground truth with varying record lengths revealed that a
core output was substantially an artifact of data volume.

Two further issues were fixed in the retrieval layer: the TF-IDF vectorizer was
refit on every query (and fit on corpus **+ query**, leaking query terms into the
IDF statistics, so scores depended on what had been asked), and the corpus of 220
snippets contained only 122 distinct texts.

---

## 6. Limitations

Stated directly, because the failure mode of this genre of project is overclaiming.

**External validity is not established.** Simulator performance measures whether the
pipeline recovers structure the simulator inserted. It says nothing about real
human behaviour. The personas encode my assumptions; validating against them
partly tests my assumptions against themselves. Real validation needs real logs
with an independent outcome measure, and this project does not have that.

**The construct is not validated.** "Avoidance-driven" is a name for a timing shape
— long gaps, late irregular returns — not a measured psychological state. Whether
that shape corresponds to what psychologists mean by avoidance is an open question
this project does not address, and the labels should be read as descriptions of
data, not of people.

**Timing is confounded.** Irregular late-night activity is consistent with
procrastination, and equally consistent with caregiving, shift work, a different
timezone from the commit clock, chronic illness, or simply preferring to work at
night. The system cannot distinguish these, and the "nocturnal but consistent"
class exists specifically so that a stable late schedule is not automatically
pathologised.

**Event streams are proxies, not work.** Commits measure commits. A researcher who
thinks for six hours and commits once is not idle. Any conclusion drawn from this
inherits every bias of the underlying event source.

**Thresholds are judgement calls.** The 24-hour gap boundary, the 2-hour session
timeout, the 5-event burst minimum: defensible, but not derived from data. They
live in `config.py` and are swept-able, and results are sensitive to them.

**Sample size.** 600 simulated users with 16 features gives adequate but not
generous power; the ±0.04 fold-to-fold standard deviation on AUC means differences
under ~0.05 between predictors should not be treated as real. This is exactly why
the heuristic/GBM comparison is reported as a tie rather than a win.

---

## 7. What would make this stronger

In rough order of scientific value:

1. **Real data with a real outcome.** Public git histories plus a defensible outcome
   (did the contributor commit in the next 7 days?) would give genuine external
   validity. This is the single biggest gap.
2. **Sequence models.** The current features collapse a time series into summary
   statistics. A temporal model over the daily activity sequence could capture
   momentum and periodicity that summary statistics discard.
3. **Per-user baselines.** Risk is currently estimated against a population. "Quiet
   *for this person*" is a stronger signal than "quiet in absolute terms".
4. **Sensitivity analysis.** Sweep the `config.py` thresholds and report how much
   conclusions move — currently unknown, and a real weakness.
5. **Fairness review before any deployment.** Circadian features correlate with
   timezone, caregiving load, disability and shift work. Any use of this to
   *evaluate* rather than *inform* a person would need that analysis first, and
   nothing here should be used to assess someone else.
