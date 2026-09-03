"""Matplotlib figures for the dashboard.

Figures are built with the explicit object-oriented API (``Figure``/``Axes``)
rather than the implicit ``pyplot`` state machine. Under Streamlit the module
is re-executed on every interaction, and the original ``plt.figure()`` calls
leaked a new figure into the global registry each rerun — a slow memory leak
that eventually triggers matplotlib's "more than 20 figures" warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "plot_hour_distribution",
    "plot_daily_activity",
    "plot_weekday_hour_heatmap",
    "plot_session_sizes",
]

_ACCENT = "#4C78A8"
_MUTED = "#B0B7C3"


def _style(ax: object, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a consistent, low-chrome style to an axis."""
    ax.set_title(title, fontsize=11, loc="left")  # type: ignore[attr-defined]
    ax.set_xlabel(xlabel, fontsize=9)  # type: ignore[attr-defined]
    ax.set_ylabel(ylabel, fontsize=9)  # type: ignore[attr-defined]
    ax.spines["top"].set_visible(False)  # type: ignore[attr-defined]
    ax.spines["right"].set_visible(False)  # type: ignore[attr-defined]
    ax.tick_params(labelsize=8)  # type: ignore[attr-defined]


def plot_hour_distribution(ts: pd.Series, late_night_hour: int = 23) -> Figure:
    """Activity count by hour of day, highlighting the late-night band."""
    counts = ts.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    fig = Figure(figsize=(6, 3), dpi=110)
    ax = fig.add_subplot(111)
    colors = [_ACCENT if (h >= late_night_hour or h < 5) else _MUTED for h in counts.index]
    ax.bar(counts.index, counts.to_numpy(), color=colors)
    ax.set_xticks(range(0, 24, 3))
    _style(ax, "Activity by hour (late night highlighted)", "Hour of day", "Events")
    fig.tight_layout()
    return fig


def plot_daily_activity(ts: pd.Series) -> Figure:
    """Events per calendar day across the full span, including zero days.

    Reindexing over the complete date range matters: the original grouped only
    on days that had events, so the line silently skipped inactive days and made
    long gaps invisible — exactly the signal the analysis is about.
    """
    daily = ts.dt.normalize().value_counts().sort_index()
    if len(daily) > 1:
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0)

    fig = Figure(figsize=(6, 3), dpi=110)
    ax = fig.add_subplot(111)
    ax.fill_between(daily.index, daily.to_numpy(), color=_ACCENT, alpha=0.25)
    ax.plot(daily.index, daily.to_numpy(), color=_ACCENT, linewidth=1.4)
    _style(ax, "Daily activity (inactive days shown as zero)", "Date", "Events")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_weekday_hour_heatmap(ts: pd.Series) -> Figure:
    """Weekday x hour density — the clearest single view of a work rhythm."""
    grid = (
        pd.crosstab(ts.dt.weekday, ts.dt.hour)
        .reindex(index=range(7), columns=range(24), fill_value=0)
        .fillna(0)
    )
    fig = Figure(figsize=(6, 2.8), dpi=110)
    ax = fig.add_subplot(111)
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
    ax.set_xticks(range(0, 24, 3))
    _style(ax, "When work happens", "Hour of day", "")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Events")
    fig.tight_layout()
    return fig


def plot_session_sizes(sessions: pd.DataFrame) -> Figure:
    """Session size vs duration, with burst sessions marked."""
    fig = Figure(figsize=(6, 3), dpi=110)
    ax = fig.add_subplot(111)
    if sessions.empty:
        ax.text(0.5, 0.5, "No sessions", ha="center", va="center")
        return fig

    bursts = sessions[sessions["is_burst"]]
    normal = sessions[~sessions["is_burst"]]
    ax.scatter(normal["span_hours"], normal["n_events"], s=28, color=_MUTED, label="Session")
    if not bursts.empty:
        ax.scatter(bursts["span_hours"], bursts["n_events"], s=42, color=_ACCENT, label="Burst")
        ax.legend(fontsize=8, frameon=False)
    _style(ax, "Working sessions", "Session length (hours)", "Events in session")
    fig.tight_layout()
    return fig
