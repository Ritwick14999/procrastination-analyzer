"""Generative model of activity logs with known ground-truth labels.

The honest problem with this project — and with most behavioural-analytics
side projects — is that there is no labelled data, so any claim that the
heuristics "work" is unfalsifiable. This module fixes that by *defining* the
behaviour generatively: we sample a latent persona, generate a timestamp log
from it, and keep the persona as the label.

That gives a measurable target. What it does **not** give is external validity:
performance here measures whether the pipeline can recover the structure this
simulator puts in, not whether real people behave this way. See
``docs/METHODOLOGY.md`` for the full statement of that limitation. The simulator
is a testbed for the estimator, not evidence about human behaviour.

Every generator takes an explicit :class:`numpy.random.Generator` so that all
results are reproducible from a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import EventFrame, build_event_frame

__all__ = ["Persona", "PersonaSpec", "PERSONAS", "simulate_user", "simulate_cohort"]


class Persona(str):
    """Ground-truth behavioural label used by the simulator."""

    AVOIDANT = "avoidant"
    DEADLINE_DRIVEN = "deadline_driven"
    FATIGUED = "fatigued"
    NOCTURNAL = "nocturnal"
    CONSISTENT = "consistent"


@dataclass(frozen=True)
class PersonaSpec:
    """Parameters of the generative process for one persona.

    Args:
        name: Ground-truth label.
        active_day_prob: Probability that any given calendar day sees work.
        session_hour_mean: Mean start hour of a working session (circular).
        session_hour_sd: Spread of session start hours, in hours.
        events_per_session: (low, high) inclusive range of events per session.
        session_minutes: (low, high) range for how long a session runs.
        weekend_factor: Multiplier on ``active_day_prob`` at weekends.
        binge_prob: Probability a session becomes an unusually long cram.
        slip_prob: Probability of dropping a scheduled day entirely, on top of
            ``active_day_prob``. Models genuine avoidance rather than schedule.
        next_day_inactive_base: Ground-truth probability that the day after the
            record ends sees no activity. This is the regression/classification
            target for the risk models.
    """

    name: str
    active_day_prob: float
    session_hour_mean: float
    session_hour_sd: float
    events_per_session: tuple[int, int]
    session_minutes: tuple[int, int]
    weekend_factor: float
    binge_prob: float
    slip_prob: float
    next_day_inactive_base: float


#: The persona library. Values are chosen to be *distinguishable but
#: overlapping* — perfectly separable classes would make the evaluation
#: meaningless, since any classifier would score 1.0.
PERSONAS: dict[str, PersonaSpec] = {
    Persona.CONSISTENT: PersonaSpec(
        name=Persona.CONSISTENT,
        active_day_prob=0.85,
        session_hour_mean=10.0,
        session_hour_sd=1.6,
        events_per_session=(2, 5),
        session_minutes=(40, 150),
        weekend_factor=0.35,
        binge_prob=0.03,
        slip_prob=0.03,
        next_day_inactive_base=0.15,
    ),
    Persona.AVOIDANT: PersonaSpec(
        name=Persona.AVOIDANT,
        active_day_prob=0.28,
        session_hour_mean=21.5,
        session_hour_sd=3.4,
        events_per_session=(1, 4),
        session_minutes=(20, 110),
        weekend_factor=0.8,
        binge_prob=0.16,
        slip_prob=0.30,
        next_day_inactive_base=0.70,
    ),
    Persona.DEADLINE_DRIVEN: PersonaSpec(
        name=Persona.DEADLINE_DRIVEN,
        active_day_prob=0.40,
        session_hour_mean=16.0,
        session_hour_sd=3.0,
        events_per_session=(5, 12),
        session_minutes=(45, 160),
        weekend_factor=0.9,
        binge_prob=0.55,
        slip_prob=0.16,
        next_day_inactive_base=0.50,
    ),
    Persona.FATIGUED: PersonaSpec(
        name=Persona.FATIGUED,
        active_day_prob=0.62,
        session_hour_mean=19.5,
        session_hour_sd=1.7,
        events_per_session=(2, 5),
        session_minutes=(30, 120),
        weekend_factor=0.5,
        binge_prob=0.07,
        slip_prob=0.12,
        next_day_inactive_base=0.36,
    ),
    Persona.NOCTURNAL: PersonaSpec(
        name=Persona.NOCTURNAL,
        active_day_prob=0.72,
        session_hour_mean=23.6,
        session_hour_sd=1.2,
        events_per_session=(3, 7),
        session_minutes=(50, 170),
        weekend_factor=0.85,
        binge_prob=0.10,
        slip_prob=0.07,
        next_day_inactive_base=0.24,
    ),
}


def _sample_hour(spec: PersonaSpec, rng: np.random.Generator) -> float:
    """Draw a session start hour, wrapped onto the 24h clock."""
    return float(rng.normal(spec.session_hour_mean, spec.session_hour_sd) % 24.0)


def simulate_user(
    persona: str,
    *,
    days: int = 60,
    rng: np.random.Generator | None = None,
    start: pd.Timestamp | None = None,
) -> tuple[EventFrame, dict[str, object]]:
    """Generate one user's activity log from a persona.

    Args:
        persona: Key into :data:`PERSONAS`.
        days: Length of the observation window in calendar days.
        rng: Seeded generator. One is created if omitted (non-reproducible).
        start: First calendar day of the window.

    Returns:
        ``(events, metadata)`` where metadata carries the ground-truth persona
        and the sampled ``next_day_inactive`` outcome used as the risk label.

    Raises:
        KeyError: if ``persona`` is not a known persona.
    """
    if persona not in PERSONAS:
        raise KeyError(f"Unknown persona {persona!r}. Known: {sorted(PERSONAS)}")
    spec = PERSONAS[persona]
    rng = rng if rng is not None else np.random.default_rng()
    start = start if start is not None else pd.Timestamp("2025-01-06")  # a Monday

    timestamps: list[pd.Timestamp] = []
    active_flags: list[bool] = []

    for day_offset in range(days):
        day = start + pd.Timedelta(days=day_offset)
        prob = spec.active_day_prob
        if day.weekday() >= 5:
            prob *= spec.weekend_factor
        # An independent "slip" models avoidance on top of the base schedule.
        active = bool(rng.random() < prob and rng.random() > spec.slip_prob)
        active_flags.append(active)
        if not active:
            continue

        hour = _sample_hour(spec, rng)
        session_start = day + pd.Timedelta(hours=hour)

        low, high = spec.events_per_session
        n_events = int(rng.integers(low, high + 1))
        lo_m, hi_m = spec.session_minutes
        span_minutes = float(rng.uniform(lo_m, hi_m))
        if rng.random() < spec.binge_prob:
            n_events = int(n_events * rng.uniform(1.8, 3.0))
            span_minutes *= rng.uniform(1.1, 1.7)

        # Events within a session arrive at irregular offsets, not on a grid.
        offsets = np.sort(rng.uniform(0.0, span_minutes, size=n_events))
        timestamps.extend(session_start + pd.Timedelta(minutes=float(m)) for m in offsets)

    # Ground-truth risk label: does the day *after* the window see activity?
    # Modulated by how the record actually ended, so the label is correlated
    # with observable behaviour rather than being pure persona noise.
    trailing = active_flags[-7:] if len(active_flags) >= 7 else active_flags
    recent_activity_rate = float(np.mean(trailing)) if trailing else 0.0
    p_inactive = float(
        np.clip(spec.next_day_inactive_base + 0.30 * (0.5 - recent_activity_rate), 0.02, 0.98)
    )
    next_day_inactive = bool(rng.random() < p_inactive)

    metadata: dict[str, object] = {
        "persona": persona,
        "days": days,
        "n_active_days": int(sum(active_flags)),
        "recent_activity_rate": round(recent_activity_rate, 3),
        "p_next_day_inactive": round(p_inactive, 3),
        "next_day_inactive": next_day_inactive,
        "reference_time": start + pd.Timedelta(days=days),
    }

    if len(timestamps) < 3:
        # Degenerate draw (a very avoidant user over a short window). Retry with
        # a longer window rather than returning something unanalysable.
        return simulate_user(persona, days=days * 2, rng=rng, start=start)

    return build_event_frame(timestamps), metadata


def simulate_cohort(
    n_per_persona: int = 60,
    *,
    days: int = 60,
    seed: int = 20240,
    personas: Sequence[str] | None = None,
    jitter_days: int = 20,
) -> list[tuple[EventFrame, dict[str, object]]]:
    """Generate a balanced, reproducible cohort across personas.

    Args:
        n_per_persona: Users to generate per persona.
        days: Base observation window.
        seed: Master seed; every user gets a distinct child stream.
        personas: Restrict to a subset of personas.
        jitter_days: Randomly vary each user's window by up to this many days,
            so that record length varies across the cohort. This is deliberate:
            it is what makes the evaluation able to detect scale-dependent
            features, which is how the original scoring bug would have surfaced.

    Returns:
        List of ``(events, metadata)`` pairs.
    """
    keys = list(personas) if personas is not None else list(PERSONAS)
    master = np.random.default_rng(seed)
    cohort: list[tuple[EventFrame, dict[str, object]]] = []

    for persona in keys:
        for _ in range(n_per_persona):
            rng = np.random.default_rng(master.integers(0, 2**32 - 1))
            window = days + int(rng.integers(-jitter_days, jitter_days + 1))
            window = max(21, window)
            events, meta = simulate_user(persona, days=window, rng=rng)
            cohort.append((events, meta))

    return cohort
