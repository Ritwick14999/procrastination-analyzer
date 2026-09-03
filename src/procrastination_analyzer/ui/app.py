"""Streamlit dashboard.

Run with::

    streamlit run src/procrastination_analyzer/ui/app.py

The app is a thin presentation layer: all analysis goes through
:func:`procrastination_analyzer.pipeline.analyze`, so the dashboard and the CLI
can never disagree about what a number means. No file path here is relative to
the working directory, so the app runs from anywhere.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run path/to/app.py` to work from a source checkout without
# the package being pip-installed first.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_SRC))

from procrastination_analyzer.config import DEFAULT_CONFIG  # noqa: E402
from procrastination_analyzer.features import segment_sessions  # noqa: E402
from procrastination_analyzer.pipeline import analyze  # noqa: E402
from procrastination_analyzer.report import render_markdown  # noqa: E402
from procrastination_analyzer.retrieval import load_index  # noqa: E402
from procrastination_analyzer.schema import (  # noqa: E402
    InsufficientDataError,
    build_event_frame,
)
from procrastination_analyzer.simulate import PERSONAS, simulate_user  # noqa: E402
from procrastination_analyzer.ui.visualize import (  # noqa: E402
    plot_daily_activity,
    plot_hour_distribution,
    plot_session_sizes,
    plot_weekday_hour_heatmap,
)

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_commits.csv"

st.set_page_config(
    page_title="Procrastination Pattern Analyzer",
    page_icon="🕒",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      div[data-testid="stMetricValue"] { font-size: 1.55rem; }
      .caveat { font-size: 0.85rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Procrastination Pattern Analyzer")
st.caption(
    "Exploratory analysis of *when* work happens. Heuristic and model outputs are "
    "descriptive signals about timing — not a psychological or clinical assessment."
)


# --------------------------------------------------------------------------
# Sidebar: data source and options
# --------------------------------------------------------------------------
st.sidebar.header("Data source")
source = st.sidebar.radio(
    "Choose input", ["Sample data", "Upload CSV", "Simulated persona", "Paste timestamps"]
)

st.sidebar.header("Suggestions")
top_k = st.sidebar.slider("How many", 2, 8, 4)

_index = load_index()
category = st.sidebar.selectbox("Category", ["(auto)", *_index.categories])

st.sidebar.header("Thresholds")
long_gap_h = st.sidebar.slider(
    "Long-gap threshold (active hours)", 6.0, 72.0, float(DEFAULT_CONFIG.long_gap_h), 2.0
)
exclude_weekend = st.sidebar.checkbox(
    "Discount weekend hours from gaps",
    value=DEFAULT_CONFIG.exclude_weekend_from_gaps,
    help="A Friday-to-Monday silence is a normal weekend, not avoidance.",
)
config = DEFAULT_CONFIG.with_overrides(
    long_gap_h=long_gap_h,
    extended_gap_h=max(long_gap_h * 2, long_gap_h + 1.0),
    exclude_weekend_from_gaps=exclude_weekend,
)


@st.cache_data(show_spinner=False)
def _read_csv(payload: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV. Cached on raw bytes so reruns are free."""
    from io import BytesIO

    return pd.read_csv(BytesIO(payload))


def _load_input() -> pd.Series | None:
    """Return a raw timestamp series from the selected source, or None."""
    if source == "Sample data":
        return pd.read_csv(SAMPLE_CSV)["timestamp"]

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV containing a timestamp column", type=["csv"])
        if uploaded is None:
            return None
        frame = _read_csv(uploaded.getvalue())
        column = st.selectbox("Which column holds the timestamps?", list(frame.columns))
        return frame[column]

    if source == "Simulated persona":
        persona = st.sidebar.selectbox("Persona", sorted(PERSONAS))
        days = st.sidebar.slider("Days to simulate", 21, 180, 60, 1)
        seed = st.sidebar.number_input("Seed", value=42, step=1)
        import numpy as np

        events, _meta = simulate_user(persona, days=int(days), rng=np.random.default_rng(int(seed)))
        return events.ts

    default = "\n".join(
        [
            "2025-01-06 09:15",
            "2025-01-06 09:50",
            "2025-01-08 23:40",
            "2025-01-13 22:05",
        ]
    )
    text = st.text_area("One timestamp per line", value=default, height=180)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return pd.Series(lines) if lines else None


raw = _load_input()
if raw is None:
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

try:
    events = build_event_frame(raw, min_events=config.min_events_required)
    result = analyze(
        events,
        config=config,
        top_k_suggestions=top_k,
        category_override=None if category == "(auto)" else category,
    )
except InsufficientDataError as exc:
    st.error(str(exc))
    st.stop()
except (ValueError, KeyError) as exc:
    st.error(f"Could not analyse this input: {exc}")
    st.stop()

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
for warning in result.warnings:
    st.warning(warning, icon="⚠️")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pattern", result.pattern.pattern.value.split(" (")[0])
c2.metric("Confidence", f"{result.pattern.confidence:.2f}")
c3.metric("Avoidance score", f"{result.avoidance:.2f}")
c4.metric("Next-day risk", f"{result.risk:.2f}", result.risk_band)

st.caption(result.pattern.pattern.description())
st.progress(min(1.0, float(result.risk)))

left, right = st.columns([3, 2])
with left:
    st.subheader("Why this classification")
    for item in result.pattern.evidence or ["No dominant signal."]:
        st.markdown(f"- {item}")
with right:
    st.subheader("Pattern scores")
    st.bar_chart(pd.Series(result.pattern.scores).sort_values(ascending=False))

st.divider()

# --------------------------------------------------------------------------
# Rhythm visualisations
# --------------------------------------------------------------------------
st.subheader("Your rhythm")
v1, v2 = st.columns(2)
with v1:
    st.pyplot(plot_hour_distribution(events.ts, config.late_night_hour))
    st.pyplot(plot_weekday_hour_heatmap(events.ts))
with v2:
    st.pyplot(plot_daily_activity(events.ts))
    st.pyplot(plot_session_sizes(segment_sessions(events, config)))

with st.expander("All extracted features"):
    st.dataframe(
        pd.DataFrame(sorted(result.features.to_dict().items()), columns=["feature", "value"]),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --------------------------------------------------------------------------
# Suggestions and export
# --------------------------------------------------------------------------
st.subheader("Suggestions")
st.caption(
    "Retrieved by TF-IDF similarity against the detected pattern. Generic advice, "
    "surfaced by relevance — not personalised guidance."
)
for s in result.suggestions:
    with st.expander(f"{s.get('title', 'Suggestion')} · relevance {s.get('score', 0):.3f}"):
        st.write(s.get("text", ""))
        if s.get("tags"):
            st.caption("Tags: " + ", ".join(s["tags"]))

st.divider()
st.subheader("Export")
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
e1, e2 = st.columns(2)
e1.download_button(
    "Download Markdown report",
    render_markdown(result),
    file_name=f"procrastination-report-{stamp}.md",
    mime="text/markdown",
)
e2.download_button(
    "Download JSON",
    __import__("json").dumps(result.to_dict(), indent=2),
    file_name=f"procrastination-analysis-{stamp}.json",
    mime="application/json",
)
