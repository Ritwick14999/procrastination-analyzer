"""Report rendering.

Reports embed the configuration and the data-quality warnings alongside the
scores, so a saved report stays interpretable and reproducible after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pipeline import AnalysisResult

__all__ = ["render_markdown", "write_report"]


def render_markdown(result: AnalysisResult) -> str:
    """Render an analysis as a self-contained Markdown document."""
    r = result
    lines: list[str] = [
        "# Procrastination Pattern Analysis",
        "",
        f"_Generated {r.generated_at}. Risk source: `{r.risk_source}`._",
        "",
        "> This is an exploratory analysis of activity timing. It is not a "
        "psychological assessment and carries no clinical meaning.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pattern | **{r.pattern.pattern.value}** |",
        f"| Classifier confidence | {r.pattern.confidence:.2f} |",
        f"| Avoidance score | {r.avoidance:.3f} / 1.00 |",
        f"| Next-day risk | {r.risk:.3f} ({r.risk_band}) |",
        f"| Events analysed | {r.features.n_events} over {r.features.span_days:.1f} days |",
        "",
        f"{r.pattern.pattern.description()}",
        "",
    ]

    if r.warnings:
        lines += ["## Data quality", ""]
        lines += [f"- {w}" for w in r.warnings]
        lines.append("")

    lines += ["## Why this classification", ""]
    lines += [f"- {e}" for e in r.pattern.evidence] or ["- No dominant signal."]
    lines.append("")

    if r.risk_drivers:
        lines += [
            "## What drives the risk estimate",
            "",
            "Log-odds contributions from the linear model; positive raises risk.",
            "",
            "| Feature | Contribution |",
            "|---|---|",
        ]
        lines += [f"| `{n}` | {v:+.3f} |" for n, v in r.risk_drivers]
        lines.append("")

    lines += ["## Behavioural features", "", "| Feature | Value |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in r.features.to_dict().items()]
    lines.append("")

    lines += ["## Suggestions", ""]
    for s in r.suggestions:
        tags = ", ".join(s.get("tags", []))
        lines += [
            f"### {s.get('title', 'Suggestion')}",
            "",
            s.get("text", ""),
            "",
            f"_Category: {s.get('category')} · relevance: {s.get('score', 0):.3f}"
            + (f" · tags: {tags}_" if tags else "_"),
            "",
        ]

    lines += [
        "## Configuration",
        "",
        "Thresholds used for this run, recorded so the numbers can be reproduced.",
        "",
        "```json",
        json.dumps(r.config_used, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(
    result: AnalysisResult,
    output_path: str | Path,
    *,
    fmt: str = "markdown",
) -> Path:
    """Write a report to disk, creating parent directories as needed.

    Args:
        result: The analysis to render.
        output_path: Destination file.
        fmt: ``"markdown"`` or ``"json"``.

    Returns:
        The path written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "markdown":
        path.write_text(render_markdown(result), encoding="utf-8")
    elif fmt == "json":
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unknown report format {fmt!r}; expected 'markdown' or 'json'")
    return path
