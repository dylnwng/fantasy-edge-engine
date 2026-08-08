"""Concentration risk across a roster: bye collisions, NFL-team stacking,
and injury-report exposure.

Every check here is **purely descriptive**. Nothing in this module adjusts
a score, and none of these values may be turned into a number in the UI --
they are context, and context is surfaced rather than baked in (the
project's central invariant, and the one most likely to erode when
someone asks for a single "roster risk" figure).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from edge_engine.roster.models import Player

# Measured, not assumed: scripts/measure_team_correlation.py found
# same-team point residuals correlate at r=0.0405 across 2018-2024, against
# a cross-team control of r=0.0087. Real, and much smaller than intuition
# suggests. Quoted in the user-facing note so the manager calibrates
# instead of over-weighting a stack.
MEASURED_TEAM_CORRELATION = 0.0405

# Below this many players on one NFL team, a "stack" isn't worth mentioning.
_STACK_THRESHOLD = 3


@dataclass(frozen=True)
class ByeCollision:
    week: int
    position: str
    players: tuple[str, ...]


def find_bye_collisions(rostered: list[Player], min_players: int = 2) -> list[ByeCollision]:
    """Weeks where `min_players` or more players at the SAME position share
    a bye. Same-position is the point: three players on bye in week 9 is
    only a problem if they compete for the same slot."""
    collisions: list[ByeCollision] = []
    by_key: dict[tuple[int, str], list[str]] = {}

    for p in rostered:
        if p.bye_week is None:
            continue
        by_key.setdefault((p.bye_week, p.position), []).append(p.name)

    for (week, position), names in sorted(by_key.items()):
        if len(names) >= min_players:
            collisions.append(ByeCollision(week=week, position=position, players=tuple(sorted(names))))
    return collisions


def team_stacking_notes(rostered: list[Player]) -> list[str]:
    """Counts of rostered players sharing an NFL team, reported as a
    variance note with the measured correlation included in the text.

    Showing how weak the signal is *is* the point of showing it: a manager
    told "you're stacked on KC" with no magnitude will over-correct."""
    counts = Counter(p.team for p in rostered if p.team)
    notes: list[str] = []
    for team, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n >= _STACK_THRESHOLD:
            notes.append(
                f"{n} rostered players share an NFL team ({team}). Measured same-team "
                f"scoring correlation is only r={MEASURED_TEAM_CORRELATION:.4f} "
                "(cross-team control r=0.0087), so this is a mild variance note, not a "
                "meaningful concentration risk. No score is adjusted for it."
            )
    return notes


def injury_exposure_notes(
    rostered: list[Player], injuries: pd.DataFrame, season: int, week: int
) -> tuple[list[str], list[str]]:
    """Returns (notes, data_gaps). Counts rostered players carrying an
    official game-status designation this week, by designation.

    Descriptive only. It does not infer a return timeline, and absence
    from the report is never treated as evidence of recovery -- the same
    rule blocker-trajectory follows in injury_context.py."""
    if injuries.empty:
        return [], [f"No injury report data for {season} week {week}; injury exposure not computed."]

    this_week = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
    if this_week.empty:
        return [], [
            f"Injury report for {season} week {week} is empty (not yet published, most "
            "likely); injury exposure not computed."
        ]

    rostered_ids = {p.player_id for p in rostered if p.player_id}
    names_by_id = {p.player_id: p.name for p in rostered if p.player_id}
    flagged = this_week[this_week["gsis_id"].isin(rostered_ids)]
    if flagged.empty:
        return ["No rostered players carry an injury designation this week."], []

    notes: list[str] = []
    for status, group in flagged.groupby("report_status"):
        names = sorted(names_by_id.get(gid, gid) for gid in group["gsis_id"].unique())
        notes.append(f"{len(names)} rostered player(s) listed {status}: {', '.join(names)}")

    unresolved = sum(1 for p in rostered if not p.player_id)
    gaps = (
        [f"{unresolved} rostered player(s) have no resolved player id and were excluded "
         "from the injury-exposure count."]
        if unresolved
        else []
    )
    return notes, gaps
