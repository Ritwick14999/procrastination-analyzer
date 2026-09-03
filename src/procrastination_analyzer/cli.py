"""Command-line interface.

Exposes the whole pipeline without Streamlit, which matters for three reasons:
it makes the project scriptable, it lets CI exercise the real entry points, and
it means the analysis can be reproduced without a browser.

Subcommands::

    analyze    Analyse a CSV of timestamps
    simulate   Generate a synthetic labelled cohort
    evaluate   Cross-validate the risk models against the heuristic baseline
    train      Fit and persist a risk model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__


def _add_analyze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Analyse a CSV of activity timestamps")
    p.add_argument("input", type=Path, help="CSV file containing a timestamp column")
    p.add_argument("--column", default=None, help="Timestamp column name (auto-detected)")
    p.add_argument("--model", type=Path, default=None, help="Trained .joblib risk model")
    p.add_argument("-k", "--suggestions", type=int, default=4, help="Suggestions to retrieve")
    p.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text", help="Output format"
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Write to this file")
    p.add_argument(
        "--now",
        default=None,
        help="Reference time for recency features (ISO 8601). Defaults to the last event.",
    )


def _add_simulate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("simulate", help="Generate a labelled synthetic cohort")
    p.add_argument("--persona", default=None, help="Single persona to generate")
    p.add_argument("-n", "--n-per-persona", type=int, default=20)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=20240)
    p.add_argument("-o", "--output", type=Path, default=None, help="Write CSV here")


def _add_evaluate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("evaluate", help="Cross-validate risk models vs the heuristic")
    p.add_argument("-n", "--n-per-persona", type=int, default=120)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20240)
    p.add_argument("--patterns", action="store_true", help="Also score the pattern rules")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("-o", "--output", type=Path, default=None)


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train and persist a risk model")
    p.add_argument("--kind", choices=["logistic", "gradient_boosting"], default="gradient_boosting")
    p.add_argument("-n", "--n-per-persona", type=int, default=120)
    p.add_argument("--seed", type=int, default=20240)
    p.add_argument("-o", "--output", type=Path, default=Path("artifacts/risk_model.joblib"))


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="procrastination-analyzer",
        description="Behavioural analytics over activity timestamps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  procrastination-analyzer analyze log.csv\n"
            "  procrastination-analyzer analyze log.csv --format markdown -o report.md\n"
            "  procrastination-analyzer evaluate --patterns\n"
            "  procrastination-analyzer train -o artifacts/risk.joblib\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_analyze(sub)
    _add_simulate(sub)
    _add_evaluate(sub)
    _add_train(sub)
    return parser


def _format_analysis_text(result: object) -> str:
    """Render an analysis as a compact terminal summary."""
    from .pipeline import AnalysisResult

    assert isinstance(result, AnalysisResult)
    r = result
    width = 62
    out: list[str] = [
        "=" * width,
        " PROCRASTINATION PATTERN ANALYSIS",
        "=" * width,
        f" Pattern      : {r.pattern.pattern.value}  (confidence {r.pattern.confidence:.2f})",
        f" Avoidance    : {r.avoidance:.3f} / 1.000",
        f" Next-day risk: {r.risk:.3f}  [{r.risk_band}]   via {r.risk_source}",
        f" Data         : {r.features.n_events} events over {r.features.span_days:.1f} days",
        "-" * width,
        " WHY",
    ]
    out += [f"   - {e}" for e in r.pattern.evidence] or ["   - No dominant signal."]

    if r.risk_drivers:
        out += ["-" * width, " RISK DRIVERS (log-odds)"]
        out += [f"   {n:<26} {v:+.3f}" for n, v in r.risk_drivers]

    if r.warnings:
        out += ["-" * width, " DATA QUALITY"]
        out += [f"   ! {w}" for w in r.warnings]

    out += ["-" * width, " SUGGESTIONS"]
    for s in r.suggestions:
        out.append(f"   [{s.get('score', 0):.3f}] {s.get('title', 'Suggestion')}")
        out.append(f"          {s.get('text', '')}")
    out.append("=" * width)
    return "\n".join(out)


def _emit(text: str, output: Path | None) -> None:
    """Write to a file or stdout."""
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}", file=sys.stderr)


def _run_analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    from .pipeline import analyze_file
    from .report import render_markdown
    from .risk import RiskModel

    model = RiskModel.load(args.model) if args.model else None
    reference = pd.Timestamp(args.now) if args.now else None

    result = analyze_file(
        args.input,
        column=args.column,
        model=model,
        reference_time=reference,
        top_k_suggestions=args.suggestions,
    )

    if args.format == "json":
        _emit(json.dumps(result.to_dict(), indent=2), args.output)
    elif args.format == "markdown":
        _emit(render_markdown(result), args.output)
    else:
        _emit(_format_analysis_text(result), args.output)
    return 0


def _run_simulate(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .simulate import PERSONAS, simulate_cohort, simulate_user

    rows: list[dict] = []
    if args.persona:
        if args.persona not in PERSONAS:
            print(f"Unknown persona {args.persona!r}. Known: {sorted(PERSONAS)}", file=sys.stderr)
            return 2
        events, meta = simulate_user(
            args.persona, days=args.days, rng=np.random.default_rng(args.seed)
        )
        rows = [{"timestamp": ts, "persona": meta["persona"], "user_id": 0} for ts in events.ts]
    else:
        cohort = simulate_cohort(n_per_persona=args.n_per_persona, days=args.days, seed=args.seed)
        for uid, (events, meta) in enumerate(cohort):
            rows.extend(
                {"timestamp": ts, "persona": meta["persona"], "user_id": uid} for ts in events.ts
            )

    frame = pd.DataFrame(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame)} events to {args.output}", file=sys.stderr)
    else:
        print(frame.to_csv(index=False))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import build_dataset, evaluate_pattern_rules, evaluate_risk_models

    dataset = build_dataset(n_per_persona=args.n_per_persona, seed=args.seed)
    report = evaluate_risk_models(
        dataset.features,
        dataset.labels,
        n_folds=args.folds,
        seed=0,
        oracle_probs=dataset.oracle_probs,
    )

    payload: dict = {
        "risk": {
            "n_samples": report.n_samples,
            "n_folds": report.n_folds,
            "base_rate": round(report.base_rate, 4),
            "predictors": {k: v.to_dict() for k, v in report.means.items()},
            "feature_importance": report.feature_importance,
            "logistic_coefficients": report.logistic_coefficients,
        }
    }
    text_parts = [report.to_markdown()]

    if args.patterns:
        rules = evaluate_pattern_rules(n_per_persona=max(20, args.n_per_persona // 3))
        payload["patterns"] = rules.to_dict()
        text_parts += [
            "",
            f"Pattern rules: accuracy {rules.accuracy:.3f} over {rules.n} users "
            f"(mean confidence {rules.mean_confidence:.3f}).",
            "",
            "| Persona | Recall |",
            "|---|---|",
        ]
        text_parts += [f"| {k} | {v:.3f} |" for k, v in sorted(rules.per_persona_recall.items())]

    if args.format == "json":
        _emit(json.dumps(payload, indent=2), args.output)
    else:
        _emit("\n".join(text_parts), args.output)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from .evaluate import build_dataset
    from .risk import RiskModel

    dataset = build_dataset(n_per_persona=args.n_per_persona, seed=args.seed)
    model = RiskModel.fit(
        dataset.features, dataset.labels.astype(bool), kind=args.kind, seed=args.seed
    )
    path = model.save(args.output)
    print(
        f"Trained {args.kind} on {model.n_training_samples} simulated users "
        f"(base rate {model.base_rate:.3f}) -> {path}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "analyze": _run_analyze,
        "simulate": _run_simulate,
        "evaluate": _run_evaluate,
        "train": _run_train,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
