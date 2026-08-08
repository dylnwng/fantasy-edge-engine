"""Assembles a TeamInsightReport from the divergence, scarcity and
exposure modules.

Output contract note: there is no `team_grade`, no roster score, and no
field that combines a projection with a context item. The PRD is explicit
that a single 0-100 roster number is the most-requested and
least-defensible thing that could go here, and it would violate the
context-is-never-numeric invariant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from edge_engine.insights.divergence import (
    DEFAULT_WINDOW,
    PlayerDivergence,
    compute_divergence,
    explain,
)
from edge_engine.insights.exposure import (
    ByeCollision,
    find_bye_collisions,
    injury_exposure_notes,
    team_stacking_notes,
)
from edge_engine.insights.scarcity import PositionScarcity, compute_scarcity, remaining_weeks
from edge_engine.roster.models import Player


@dataclass(frozen=True)
class PlayerInsight:
    player_id: str | None
    name: str
    position: str
    usage_z: float | None
    points_z: float | None
    divergence: float | None
    signal: str  # "undervalued" | "overvalued" | "neutral" | "no_data"
    context: list[str] = field(default_factory=list)  # text only, never numeric


@dataclass(frozen=True)
class TeamInsightReport:
    season: int
    week: int
    window: int
    players: list[PlayerInsight]
    scarcity: list[PositionScarcity]
    bye_collisions: list[ByeCollision]
    exposure_notes: list[str]
    data_gaps: list[str]


def build_report(
    rostered: list[Player],
    player_week: pd.DataFrame,
    points: pd.Series,
    scored: pd.DataFrame,
    lineup_slots: dict[str, int],
    injuries: pd.DataFrame,
    season: int,
    week: int,
    window: int = DEFAULT_WINDOW,
) -> TeamInsightReport:
    gaps: list[str] = []

    if not rostered:
        # A genuinely empty roster (an undrafted league in the offseason,
        # most often) must announce itself. A table of zeros looks like a
        # broken tool, and "no signal is a designed state, not an empty
        # result" applies to the whole report, not just the neutral band.
        gaps.append(
            "The active roster source returned 0 rostered players — nothing to analyze. "
            "In the offseason this is expected (the league hasn't drafted yet)."
        )

    divergences, div_gaps = compute_divergence(player_week, points, season, week, window)
    gaps.extend(div_gaps)
    by_id: dict[str, PlayerDivergence] = {d.player_id: d for d in divergences}

    byes_known = any(p.bye_week is not None for p in rostered)
    weeks, week_gaps = remaining_weeks(week, byes_known)
    gaps.extend(week_gaps)

    scarcity, scarcity_gaps = compute_scarcity(rostered, lineup_slots, scored, weeks)
    gaps.extend(scarcity_gaps)

    collisions = find_bye_collisions(rostered)
    notes = list(team_stacking_notes(rostered))
    injury_notes, injury_gaps = injury_exposure_notes(rostered, injuries, season, week)
    notes.extend(injury_notes)
    gaps.extend(injury_gaps)

    players: list[PlayerInsight] = []
    unresolved = 0
    for p in rostered:
        d = by_id.get(p.player_id) if p.player_id else None
        context: list[str] = []
        if p.bye_week is not None:
            context.append(f"Bye week {p.bye_week}")
        if p.match_status != "matched":
            context.append(f"Unresolved against nflverse data ({p.match_status})")

        if d is None:
            unresolved += 1
            players.append(
                PlayerInsight(
                    player_id=p.player_id, name=p.name, position=p.position,
                    usage_z=None, points_z=None, divergence=None,
                    signal="no_data",
                    context=[*context, "No usage data in the window; divergence unavailable."],
                )
            )
            continue

        players.append(
            PlayerInsight(
                player_id=p.player_id, name=p.name, position=p.position,
                usage_z=d.usage_z, points_z=d.points_z, divergence=d.divergence,
                signal=d.signal, context=[*context, explain(d)],
            )
        )

    if unresolved:
        gaps.append(
            f"{unresolved} of {len(rostered)} rostered player(s) had no usage data in "
            f"{season} weeks {max(1, week - window + 1)}-{week} and show no divergence."
        )

    players.sort(key=lambda pi: (pi.divergence is None, -abs(pi.divergence or 0.0)))

    return TeamInsightReport(
        season=season, week=week, window=window,
        players=players, scarcity=scarcity, bye_collisions=collisions,
        exposure_notes=notes, data_gaps=gaps,
    )
